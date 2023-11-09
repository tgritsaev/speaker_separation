import random
from tqdm import tqdm

import PIL
import pandas as pd
import numpy as np
import torch
from torchvision.transforms import ToTensor
import pyloudnorm as pyln

from src.base import BaseTrainer
from src.logger.utils import plot_spectrogram_to_buf
from src.utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizer,
        config,
        device,
        dataloaders,
        lr_scheduler=None,
        len_epoch=None,
        skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, optimizer, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler = lr_scheduler
        self.log_step = 100

        self.scaler = torch.cuda.amp.GradScaler()

        self.train_metrics = MetricTracker("loss", "grad norm", *[m.name for m in self.metrics], writer=self.writer)
        self.evaluation_metrics = MetricTracker(*[m.name for m in self.metrics if not m.ignore_on_eval], writer=self.writer)

        self.meter = pyln.Meter(config["preprocessing"]["sr"])

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["y_wav", "x_wav", "target_wav"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["trainer"]["grad_norm_clip"])

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            outputs = self.model(**batch)
            batch.update(outputs)
            if is_train:
                batch["loss"] = self.criterion(**batch)
        if is_train:
            self.scaler.scale(batch["loss"]).backward()
            self.scaler.unscale_(self.optimizer)
            self._clip_grad_norm()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            metrics.update("loss", batch["loss"].item())

        # outputs = self.model(**batch)
        # batch.update(outputs)
        # if is_train:
        #     batch["loss"] = self.criterion(**batch)
        #     self._clip_grad_norm()
        #     batch["loss"].backward()
        #     self.optimizer.step()
        #     if self.lr_scheduler is not None:
        #         self.lr_scheduler.step()
        #     metrics.update("loss", batch["loss"].item())

        wavs = batch["s1"]
        normalized_s = torch.zeros_like(batch["s1"], device=wavs.device)
        for i in range(wavs.shape[0]):
            tensor_wav = torch.nan_to_num(wavs[i])
            normalized_s[i] = 20 * tensor_wav / tensor_wav.norm()
        batch.update({"normalized_s": normalized_s})

        for metric in self.metrics:
            metrics.update(metric.name, metric(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(dataloader), desc=part, total=len(dataloader)):
                batch = self.process_batch(batch, is_train=False, metrics=self.evaluation_metrics)
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_predictions(False, **batch)
            # self._log_spectrogram(batch["spectrogram"])
            self._log_scalars(self.evaluation_metrics)

        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(self, is_train, **batch):
        y_wav = batch["y_wav"]
        x_wav = batch["x_wav"]
        target_wav = batch["target_wav"]
        normalized_s = batch["normalized_s"]
        if self.writer is None:
            return
        batch_size = y_wav.shape[0]
        examples_to_log = min(2, batch_size)
        ids = np.random.choice(batch_size, examples_to_log, replace=False)

        def get_wandb_audio(tensor):
            return self.writer.wandb.Audio(tensor.detach().to(torch.float32).cpu().numpy(), sample_rate=16000)

        def get_i_tensors_for_metrics(i, **batch):
            out = {}
            for key, value in batch.items():
                if torch.is_tensor(value) and key != "loss":
                    out[key] = value[i].unsqueeze(0)
            return out

        rows = {}
        for i in ids:
            rows[i] = {
                "norm_pred": get_wandb_audio(normalized_s[i]),
                "mixed": get_wandb_audio(y_wav[i]),
                "ref": get_wandb_audio(x_wav[i]),
                "target": get_wandb_audio(target_wav[i]),
            }
            for metric in self.metrics:
                if not is_train and metric.ignore_on_eval:
                    continue
                kwargs = get_i_tensors_for_metrics(i, **batch)
                rows[i].update({metric.name: metric(**kwargs)})

        self.writer.add_table("predictions", pd.DataFrame.from_dict(rows, orient="index"))

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack([torch.norm(torch.nan_to_num(p.grad.detach(), nan=0), norm_type).cpu() for p in parameters]),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(tqdm(self.train_dataloader, desc="train", total=self.len_epoch)):
            try:
                batch = self.process_batch(batch, is_train=True, metrics=self.train_metrics)
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug("Train Epoch: {} {} Loss: {:.6f}".format(epoch, self._progress(batch_idx), batch["loss"].item()))
                self.writer.add_scalar("learning rate", self.lr_scheduler.get_last_lr()[0])
                self._log_predictions(True, **batch)
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()

            if batch_idx % 25 == 0:
                torch.cuda.empty_cache()

        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log
