import torch
import torch.nn.functional as F
from typing import List


def ss_collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    y_wav, x_wav, x_wav_len, target_wav, speaker_id, text = [], [], [], [], [], []

    def get_max_length(key_):
        return max(dataset_items, key=lambda item: item[key_].shape[1])[key_].shape[1]

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
        text.append(item["text"] if "text" in item.keys() else "")

    return {
        "y_wav": torch.cat(y_wav),
        "x_wav": torch.cat(x_wav),
        "x_wav_len": torch.Tensor(x_wav_len),
        "target_wav": torch.cat(target_wav),
        "speaker_id": torch.LongTensor(speaker_id),
        "text": text,
    }
