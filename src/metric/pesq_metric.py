from src.base.base_metric import BaseMetric
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality


class PESQMetric(BaseMetric):
    def __init__(self, fs=16000, mode="wb", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = PerceptualEvaluationSpeechQuality(fs, mode)

    def __call__(self, normalized_s, target_wav, **kwargs):
        return self.metric(normalized_s, target_wav)
