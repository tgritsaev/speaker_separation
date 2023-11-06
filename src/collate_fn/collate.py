import torch
import logging
from typing import List

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    y_wav, x_wav, x_wav_len, target_wav, speaker_id = [], [], [], [], []

    def get_x_wav_length(item):
        return item["x_wav"].shape[1]

    max_x_wav_length = get_x_wav_length(max(dataset_items, key=get_x_wav_length))
    for item in dataset_items:
        y_wav.append(item["y_wav"])
        cur_x_wav_len = get_x_wav_length(item)
        x_wav.append(torch.nn.functional.pad(item["x_wav"], (1, max_x_wav_length - cur_x_wav_len)))
        x_wav_len.append(cur_x_wav_len)
        target_wav.append(item["target_wav"])
        speaker_id.append(item["speaker_id"])

    return {
        "y_wav": torch.cat(y_wav),
        "x_wav": torch.cat(x_wav),
        "x_wav_len": torch.Tensor(x_wav_len),
        "target_wav": torch.cat(target_wav),
        "speaker_id": torch.LongTensor(speaker_id),
    }
