import torch
import torch.nn.functional as F
import logging
from typing import List

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    y_wav, x_wav, x_wav_len, target_wav, speaker_id = [], [], [], [], []

    def get_max_length(key):
        return max(dataset_items, key=lambda item: item[key].shape[1]).shape[1]

    def pad_to_len(wav, len):
        return F.pad(wav, (1, len - wav.shape[1]))

    max_x_wav_length = get_max_length("x_wav")
    max_y_target_wav_length = max(get_max_length("y_wav"), get_max_length("target_wav"))
    for item in dataset_items:
        y_wav.append(pad_to_len(item["y_wav"], max_y_target_wav_length))
        x_wav.append(pad_to_len(item["x_wav"], max_x_wav_length))
        target_wav.append(pad_to_len(item["target_wav"], max_y_target_wav_length))
        x_wav_len.append(item["x_wav"].shape[1])
        speaker_id.append(item["speaker_id"])

    return {
        "y_wav": torch.cat(y_wav),
        "x_wav": torch.cat(x_wav),
        "x_wav_len": torch.Tensor(x_wav_len),
        "target_wav": torch.cat(target_wav),
        "speaker_id": torch.LongTensor(speaker_id),
    }
