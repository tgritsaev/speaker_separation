import torch
from src.base.base_metric import BaseMetric


class SISDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = 1e-6

    def __call__(self, s1, target_wav, **kwargs):
        alpha = (target_wav * s1).sum() / (torch.linalg.norm(target_wav) ** 2)
        return -20 * torch.log10(torch.linalg.norm(alpha * target_wav) / (torch.linalg.norm(alpha * target_wav - s1) + self.eps) + self.eps)
