import torch
from torch import nn
import torch.nn.functional as F
from src.base import BaseModel

# https://www.isca-speech.org/archive/pdfs/interspeech_2020/ge20_interspeech.pdf


class SpeechEncoder(nn.Module):
    def __init__(self, L1, L2, L3, channels_cnt):
        super().__init__()
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.stride = L1 // 2
        self.short = nn.Sequential(nn.Conv1d(1, channels_cnt, L1, self.stride), nn.ReLU())
        self.middle = nn.Sequential(nn.Conv1d(1, channels_cnt, L2, self.stride), nn.ReLU())
        self.long = nn.Sequential(nn.Conv1d(1, channels_cnt, L3, self.stride), nn.ReLU())

    def forward(self, x, return_other=False):
        # B x W
        x = torch.unsqueeze(x, 1)
        x1 = self.short(x)
        # B x N x final_len
        final_len = x1.shape[-1]
        len1 = x.shape[-1]
        len2 = (final_len - 1) * self.stride + self.L2
        len3 = (final_len - 1) * self.stride + self.L3

        x2 = self.middle(F.pad(x, (0, len2 - len1), "constant", 0))
        x3 = self.long(F.pad(x, (0, len3 - len1), "constant", 0))

        if return_other:
            return torch.cat([x1, x2, x3], 1), [x1, x2, x3]
        return torch.cat([x1, x2, x3], 1)
        # B x 3N x final_len


class Norm(nn.Module):
    def __init__(self, channels_cnt):
        super().__init__()
        self.ln = nn.LayerNorm(channels_cnt)

    def forward(self, x):
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class ResNetBlock(nn.Module):
    mul = 2

    def __init__(self, channels_cnt):
        super().__init__()
        self.part1 = nn.Sequential(
            nn.Conv1d(channels_cnt, ResNetBlock.mul * channels_cnt, 1, bias=False),
            nn.BatchNorm1d(ResNetBlock.mul * channels_cnt),
            nn.PReLU(),
            nn.Conv1d(ResNetBlock.mul * channels_cnt, channels_cnt, 1, bias=False),
            nn.BatchNorm1d(channels_cnt),
        )
        self.part2 = nn.Sequential(nn.PReLU(), nn.MaxPool1d(3))

    def forward(self, x):
        x = self.part1(x) + x
        x = self.part2(x)
        return x


class SpeakerEncoder(nn.Module):
    mul = 3

    def __init__(self, channels_cnt, ResNetBlock_cnt, L1, stride, speakers_cnt):
        super().__init__()
        self.ResNetBlock_cnt = ResNetBlock_cnt
        self.L1 = L1
        self.stride = stride
        self.norm = Norm(SpeakerEncoder.mul * channels_cnt)
        self.conv1 = nn.Conv1d(SpeakerEncoder.mul * channels_cnt, channels_cnt, 1)
        self.resnet_blocks = nn.Sequential(*(ResNetBlock(channels_cnt) for _ in range(ResNetBlock_cnt)))
        self.conv2 = nn.Conv1d(channels_cnt, channels_cnt, 1)
        self.classification = nn.Linear(channels_cnt, speakers_cnt)

    def forward(self, x, len):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.resnet_blocks(x)
        x = self.conv2(x)

        final_len = (len - self.L1) // self.stride + 1
        for _ in range(self.ResNetBlock_cnt):
            final_len //= 3
        speaker_embedding = torch.sum(x, -1) / final_len.view(-1, 1).to(x.device)

        return self.classification(speaker_embedding), speaker_embedding


class GlobalLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super(GlobalLayerNorm, self).__init__()
        self.eps = eps
        self.beta = nn.Parameter(torch.zeros(dim, 1))
        self.gamma = nn.Parameter(torch.ones(dim, 1))

    def forward(self, x):
        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
        x = self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
        return x


class TCN(nn.Module):
    mul = 2

    def __init__(self, channels_cnt, kernel_size, speaker_channels_cnt, dilation):
        super().__init__()
        conv_padding = (dilation * (kernel_size - 1)) // 2
        self.seq = nn.Sequential(
            nn.Conv1d(channels_cnt + speaker_channels_cnt, TCN.mul * channels_cnt, 1),
            nn.PReLU(),
            GlobalLayerNorm(TCN.mul * channels_cnt),
            nn.Conv1d(TCN.mul * channels_cnt, TCN.mul * channels_cnt, kernel_size, padding=conv_padding, dilation=dilation, groups=TCN.mul * channels_cnt),
            nn.PReLU(),
            GlobalLayerNorm(TCN.mul * channels_cnt),
            nn.Conv1d(TCN.mul * channels_cnt, channels_cnt, 1),
        )

    def forward(self, x, speaker_embedding):
        if speaker_embedding is None:
            return x + self.seq(x)
        repeated_speaker_embedding = torch.unsqueeze(speaker_embedding, -1).repeat(1, 1, x.shape[-1])
        return x + self.seq(torch.concat([x, repeated_speaker_embedding], dim=1))


class StackedTCNs(nn.Module):
    def __init__(self, channels_cnt, speaker_channels_cnt, TCN_cnt):
        super().__init__()
        self.TCN_cnt = TCN_cnt
        tcns = [TCN(channels_cnt, 3, speaker_channels_cnt, 1)] + [TCN(channels_cnt, 3, 0, 2**i) for i in range(1, TCN_cnt)]
        self.tcns = nn.ModuleList(tcns)

    def forward(self, x, speaker_embedding):
        for i, tcn in enumerate(self.tcns):
            x = tcn(x, speaker_embedding) if i == 0 else tcn(x, None)
        return x


class SpeakerExtractor(nn.Module):
    mul1 = 3
    mul2 = 2

    def __init__(self, channels_cnt, speaker_channels_cnt, TCN_cnt):
        super().__init__()
        self.TCN_cnt = TCN_cnt
        self.norm = Norm(SpeakerExtractor.mul1 * channels_cnt)
        self.conv1 = nn.Conv1d(SpeakerExtractor.mul1 * channels_cnt, channels_cnt, 1)
        self.stacked_TCNs = nn.ModuleList([StackedTCNs(channels_cnt, speaker_channels_cnt, TCN_cnt) for _ in range(4)])
        self.convs = nn.ModuleList([nn.Sequential(nn.Conv1d(channels_cnt, channels_cnt, 1), nn.ReLU()) for _ in range(3)])

    def forward(self, x, speaker_embedding):
        x = self.norm(x)
        x = self.conv1(x)
        for stacked_TCNs in self.stacked_TCNs:
            x = stacked_TCNs(x, speaker_embedding)

        extracted_speech = [conv(x) for conv in self.convs]
        return extracted_speech


class SpeechDecoder(nn.Module):
    def __init__(self, L1, L2, L3, channels_cnt):
        super().__init__()
        self.short = nn.ConvTranspose1d(channels_cnt, 1, L1, L1 // 2)
        self.middle = nn.ConvTranspose1d(channels_cnt, 1, L2, L1 // 2)
        self.long = nn.ConvTranspose1d(channels_cnt, 1, L3, L1 // 2)

    def forward(self, x_short, x_middle, x_long):
        return self.short(x_short).squeeze(), self.middle(x_middle).squeeze(), self.long(x_long).squeeze()


class SpExPlusModel(BaseModel):
    def __init__(self, L1, L2, L3, N, ResNetBlock_cnt, TCN_cnt, speakers_cnt):
        super().__init__()
        self.speech_encoder = SpeechEncoder(L1, L2, L3, N)
        self.speaker_encoder = SpeakerEncoder(N, ResNetBlock_cnt, L1, self.speech_encoder.stride, speakers_cnt)
        self.speaker_extractor = SpeakerExtractor(N, N, TCN_cnt)
        self.speech_decoder = SpeechDecoder(L1, L2, L3, N)

    def forward(self, y_wav, x_wav, x_wav_len, **kwargs):
        y, ys = self.speech_encoder(y_wav, True)
        x = self.speech_encoder(x_wav)

        speaker_preds, speaker_embedding = self.speaker_encoder(x, x_wav_len)

        extracted_speech = self.speaker_extractor(y, speaker_embedding)
        s_short, s_middle, s_long = self.speech_decoder(*[ys[i] * extracted_speech[i] for i in range(len(ys))])
        ylen = y_wav.shape[-1]

        return {"speaker_pred": speaker_preds, "s1": F.pad(s_short[:, :ylen], (0, ylen)), "s2": s_middle[:, :ylen], "s3": s_long[:, :ylen]}
