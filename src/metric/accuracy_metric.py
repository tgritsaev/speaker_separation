import torch
from src.base.base_metric import BaseMetric


class AccuracyMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, speaker_pred, speaker_id, **kwargs):
        accuracy = torch.sum(speaker_pred.argmax(dim=-1) == speaker_id.to(speaker_pred.device)) / speaker_id.shape[0]
        return accuracy
