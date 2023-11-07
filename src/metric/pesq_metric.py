from src.base.base_metric import BaseMetric
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality


class PESQMetric(BaseMetric):
    def __init__(self, fs=16000, mode="nb"):
        super().__init__()
        self.metric = PerceptualEvaluationSpeechQuality(fs, mode)

    def __call__(self, preds, target):
        return self.metric(preds, target)
