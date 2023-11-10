import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

import src.model as module_model
import src.metric as module_metric
from src.trainer import Trainer
from src.utils import MetricTracker
from src.utils.object_loading import get_dataloaders
from src.utils.parse_config import ConfigParser


def main(config):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    def load_model(arch, checkpoint):
        # build ss_model architecture
        model = config.init_obj(config[arch], module_model, n_class=len(text_encoder))
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

    ss_model = load_model("ss_arch", config.ss_checkpoint)
    if config.asr_checkpoint:
        # text_encoder
        text_encoder = config.get_text_encoder()
        asr_model = load_model("asr_arch", config.asr_checkpoint)

    metrics = [config.init_obj(metric_dict, module_metric) for metric_dict in config["metrics"]]
    metrics_tracker = MetricTracker(*[m.name for m in metrics])

    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloaders["test"])):
            batch = Trainer.move_batch_to_device(batch, device)
            output = ss_model(**batch)
            batch.update(output)
            for metric in metrics:
                metrics_tracker.update(metric.name, metric(**batch))

    for metric in metrics:
        line = f"{metric.name}: {metric.avg()}"
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
        help="Path to test data folder",
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
                            "audio_dir": str(test_data_folder),
                            "transcription_dir": str(test_data_folder),
                        },
                    }
                ],
            }
        }

    main(config)
