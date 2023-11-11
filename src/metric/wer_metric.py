from typing import List

import torch
from torch import Tensor

from src.base.base_metric import BaseMetric
from src.base.base_text_encoder import BaseTextEncoder
from src.metric.utils import calc_wer


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.name = name

    def __call__(self, pred_log_probs: Tensor, target_log_probs: Tensor, pred_lengths, target_lengths, text: List[str], **kwargs):
        wers = []
        if "target" in self.name:
            predictions = torch.argmax(target_log_probs.cpu(), dim=-1).numpy()
            lengths = pred_lengths
        else:
            predictions = torch.argmax(pred_log_probs.cpu(), dim=-1).numpy()
            lengths = target_lengths
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec)[:length]
            else:
                pred_text = self.text_encoder.decode(log_prob_vec)[:length]
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)
