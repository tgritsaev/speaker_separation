import torch
import logging
from typing import List

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    # TODO: your code here
    feature_length_dim = dataset_items[0]['spectrogram'].shape[1]
    time_dim = max(dataset_items, key=lambda item: item['spectrogram'].shape[2])['spectrogram'].shape[2]
    spectrogram = torch.zeros((len(dataset_items), feature_length_dim, time_dim))
    spectrogram_length = []

    text_length_dim = max(dataset_items, key=lambda item: item['text_encoded'].shape[1])['text_encoded'].shape[1]
    text_encoded = torch.zeros((len(dataset_items), text_length_dim))
    text_encoded_length = []
    text = []

    audio_path = []
    audio = []
    for i, item in enumerate(dataset_items):
        cur_time_dim = item['spectrogram'].shape[2]
        spectrogram[i] = torch.cat([item['spectrogram'][0], torch.zeros((feature_length_dim, time_dim - cur_time_dim))], axis=1)
        spectrogram_length.append(cur_time_dim)

        cur_text_length_dim = item['text_encoded'].shape[1]
        text_encoded[i] = torch.cat([item['text_encoded'][0], torch.zeros(text_length_dim - cur_text_length_dim)])
        text_encoded_length.append(cur_text_length_dim)
        text.append(item['text'])

        audio_path.append(item['audio_path'])
        audio.append(item['audio'])

    return {
        'spectrogram': spectrogram,
        'spectrogram_length': torch.Tensor(spectrogram_length).to(torch.int32),
        'text_encoded': text_encoded,
        'text_encoded_length': torch.Tensor(text_encoded_length).to(torch.int32),
        'text': text,
        'audio_path': audio_path,
        'audio': audio,
    }
