import argparse
import json
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F

import src.model as ss_module_model
import hw_asr.model as asr_module_model
import src.metric as module_metric
from src.trainer import Trainer
from src.utils import MetricTracker
from src.utils.object_loading import get_dataloaders
from src.utils.parse_config import ConfigParser


def main(config, args):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup data_loader instances
    dataloader = get_dataloaders(config)["test"]

    def load_model(arch, checkpoint):
        # build model architecture
        if "ss" in arch:
            model = config.init_obj(config[arch], ss_module_model)
        else:
            model = config.init_obj(config[arch], asr_module_model, n_class=len(text_encoder))
        logger.info(model)

        logger.info("Loading checkpoint...")
        checkpoint = torch.load(checkpoint, map_location=device)
        state_dict = checkpoint["state_dict"]
        if config["n_gpu"] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)
        logger.info("Checkpoint has been loaded.")

        # prepare ss_model for testing
        model = model.to(device)
        model.eval()
        return model

    ss_model = load_model("ss_arch", args.ss_checkpoint)
    if args.asr_checkpoint is not None:
        # text_encoder
        text_encoder = config.get_text_encoder()
        asr_model = load_model("asr_arch", args.asr_checkpoint)

    segmentation = False
    metrics = []
    for metric_dict in config["metrics"]:
        if "Segmented" in metric_dict["type"]:
            segmentation = True
            metrics.append(config.init_obj(metric_dict, module_metric))
        elif "WER" in metric_dict["type"] or "CER" in metric_dict["type"]:
            if args.asr_checkpoint is not None:
                metrics.append(config.init_obj(metric_dict, module_metric, text_encoder=text_encoder))
        else:
            metrics.append(config.init_obj(metric_dict, module_metric))
    metrics_tracker = MetricTracker(*[m.name for m in metrics])

    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader)):
            batch = Trainer.move_batch_to_device(batch, device)

            # basic metrics
            outputs = ss_model(**batch)
            batch.update(outputs)

            wav = batch["s1"]
            normalized_s = torch.zeros_like(batch["s1"], device=wav.device)
            for i in range(wav.shape[0]):
                tensor_wav = torch.nan_to_num(wav[i], nan=0)
                normalized_s[i] = (20 * tensor_wav / tensor_wav.norm()).to(torch.float32)
            batch.update({"normalized_s": normalized_s})

            # ASR
            if args.asr_checkpoint is not None:

                def insert_logits(pref, wav):
                    _, spectrogram = dataloader.dataset.process_wave(wav.cpu())
                    batch["spectrogram"] = spectrogram.to(device)
                    batch["spectrogram_length"] = torch.Tensor([spectrogram.shape[1]]).to(device)
                    batch[pref + "log_probs"] = F.log_softmax(asr_model(**batch)["logits"], dim=-1)

                insert_logits("pred_", normalized_s)
                insert_logits("target_", batch["target_wav"])
                batch["lengths"] = [len(batch["text"][0])]

            # Segmented
            if segmentation:
                assert dataloader.batch_size == 1, "Yoy can use only `batch_size=1`!"

                window_len = int(args.second * config["preprocessing"]["sr"])
                wav_len = len(wav)
                segmented_wavs = []
                for left in range(0, wav_len, window_len):
                    segmented_batch = {}
                    right = left + window_len
                    if right > window_len:
                        break
                    for key in ["x_wav", "y_wav"]:
                        segmented_batch[key] = batch[key][0, left:right].unsqueeze(0).to(device)
                    segmented_batch["x_wav_len"] = torch.Tensor([right - left]).to(device)

                    segmented_wav = torch.nan_to_num(ss_model(**segmented_batch)["s1"], nan=0)
                    segmented_wavs.append((20 * segmented_wav / segmented_wav.norm()).to(torch.float32))
                    print(segmented_wav.shape)
                batch.update({"segmented_s": torch.concatenate(segmented_wavs, dim=1)})
                batch.update({"cut_target_wav": batch["target_wav"][0, (wav_len // window_len) * window_len]})
                print(batch["cut_target_wav"].shape)

            for metric in metrics:
                metrics_tracker.update(metric.name, metric(**batch))

    for metric in metrics:
        name = metric.name
        line = f"{name}: {metrics_tracker.avg(name)}"
        logger.info(line)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    # args.add_argument(
    #     "-o",
    #     "--output",
    #     default="output.json",
    #     type=str,
    #     help="File to write results (.json)",
    # )
    args.add_argument(
        "-c",
        "--config",
        default="test_model/config.json",
        type=str,
        help="Path to config",
    )
    args.add_argument(
        "--ss_checkpoint",
        default="test_model/ss_checkpoint.pth",
        type=str,
        help="Path to speech separation checkpoint",
    )
    args.add_argument(
        "--asr_checkpoint",
        default=None,
        type=str,
        help="Path to audio speech recognition checkpoint",
    )
    args.add_argument(
        "--test_data_folder",
        default=None,
        type=str,
        help="Path to custom test data folder",
    )
    args.add_argument(
        "-s",
        "--second",
        default=0.1,
        type=float,
        help="Window length in seconds for segmented speech separation",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )
    args = args.parse_args()

    with Path(args.config).open() as f:
        config = ConfigParser(json.load(f))

    # if `--test-data-folder` was provided, set it as a default test set
    if args.test_data_folder is not None:
        test_data_folder = Path(args.test_data_folder).absolute().resolve()
        assert test_data_folder.exists()
        config.config["data"] = {
            "test": {
                "batch_size": 1,
                "num_workers": args.jobs,
                "datasets": [
                    {
                        "type": "CustomDirAudioDataset",
                        "args": {
                            "mix_dir": str(test_data_folder) + "/mix",
                            "ref_dir": str(test_data_folder) + "/refs",
                            "target_dir": str(test_data_folder) + "/targets",
                            "limit": 1,
                        },
                    }
                ],
            }
        }

    main(config, args)
