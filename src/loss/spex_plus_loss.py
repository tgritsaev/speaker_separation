import torch
import torch.nn as nn


def si_sdr(estimated, target):
    eps = 1e-6
    alpha = (target * estimated).sum() / (torch.linalg.norm(target) ** 2)
    return 20 * torch.log10(torch.linalg.norm(alpha * target) / (torch.linalg.norm(alpha * target - estimated) + eps) + eps)


# https://www.isca-speech.org/archive/pdfs/interspeech_2020/ge20_interspeech.pdf, 2.4. Multi-task learning
class SpExPlusLoss(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, s1, s2, s3, speaker_pred, target_wav, speaker_id, **kwargs):
        s1 = s1 - s1.mean(dim=-1, keepdim=True)
        s2 = s2 - s2.mean(dim=-1, keepdim=True)
        s3 = s3 - s3.mean(dim=-1, keepdim=True)
        target_wav = target_wav - target_wav.mean(dim=-1, keepdim=True)

        batch_size = target_wav.shape[0]
        loss_si_sdr = (
            -(1 - self.alpha - self.beta) * si_sdr(s1, target_wav).sum() - self.alpha * si_sdr(s2, target_wav).sum() - self.beta * si_sdr(s3, target_wav).sum()
        ) / batch_size
        ce = self.ce_loss(speaker_pred, speaker_id.to(speaker_pred.device))
        return loss_si_sdr + self.gamma * ce
