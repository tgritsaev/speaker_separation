from typing import List

import torch
from torch import Tensor

from src.base.base_metric import BaseMetric
from src.base.base_text_encoder import BaseTextEncoder
from src.metric.utils import calc_cer


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.name = name

    def __call__(self, pred_log_probs: Tensor, target_log_probs: Tensor, text: List[str], **kwargs):
        cers = []
        if "target" in self.name:
            predictions = torch.argmax(target_log_probs.cpu(), dim=-1).numpy()
        else:
            predictions = torch.argmax(pred_log_probs.cpu(), dim=-1).numpy()
        for log_prob_vec, target_text in zip(predictions, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec)
            else:
                pred_text = self.text_encoder.decode(log_prob_vec)
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)
