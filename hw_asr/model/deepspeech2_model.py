import torch
from torch import nn

from hw_asr.base import BaseModel


class RNNwBatchNorm(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_dropout):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, dropout=rnn_dropout, batch_first=False, bidirectional=True)
        self.norm = nn.BatchNorm1d(hidden_size)

    def forward(self, x, h=None):
        # N x T x input_size
        x, h = self.rnn(x, h)
        # T x N x (2 * hidden_size)
        x = x.view(x.shape[0], x.shape[1], 2, -1).sum(2)
        # T x N x hidden_size
        t_dim, n_dim = x.shape[0], x.shape[1]
        x = x.view((t_dim * n_dim, -1))
        x = self.norm(x)
        x = x.view((t_dim, n_dim, -1)).contiguous()
        return x, h


# https://proceedings.mlr.press/v48/amodei16.pdf
class DeepSpeech2Model(BaseModel):
    def __init__(self, n_feats, n_rnn_layers, rnn_hidden_size, rnn_dropout, n_class):
        assert n_rnn_layers >= 1
        super().__init__(n_feats, n_class)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, padding=(20, 5), kernel_size=(41, 11), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, padding=(10, 5), kernel_size=(21, 11), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=96, padding=(10, 5), kernel_size=(21, 11), stride=(2, 1)),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )

        rnn_input_size = (n_feats + 2 * 20 - 41) // 2 + 1
        rnn_input_size = (rnn_input_size + 2 * 10 - 21) // 2 + 1
        rnn_input_size = (rnn_input_size + 2 * 10 - 21) // 2 + 1
        rnn_input_size *= 96
        self.rnns = nn.Sequential(
            RNNwBatchNorm(rnn_input_size, rnn_hidden_size, rnn_dropout),
            *(RNNwBatchNorm(rnn_hidden_size, rnn_hidden_size, rnn_dropout) for _ in range(n_rnn_layers - 1))
        )

        self.fc = nn.Linear(in_features=rnn_hidden_size, out_features=n_class)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, spectrogram, spectrogram_length, **batch):
        # N x big_F x big_T
        x = self.conv(spectrogram.unsqueeze(1))
        # N x C x F x T
        x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        # N x (C * F) x T
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        # T x N x (C * F)
        h = None
        for rnn in self.rnns:
            x, h = rnn(x, h)
        # T x N x rnn_hidden_size
        t_dim, n_dim = x.shape[0], x.shape[1]
        x = x.view((t_dim * n_dim, -1))
        x = self.fc(x)
        x = x.view((t_dim, n_dim, -1)).transpose(0, 1)
        # N x T x n_class
        return {"logits": x}

    def transform_input_lengths(self, input_lengths):
        t_dim = input_lengths.max()

        t_dim = (t_dim + 2 * 5 - 11) // 2 + 1
        t_dim = (t_dim + 2 * 5 - 11) // 2 + 1
        t_dim = (t_dim + 2 * 5 - 11) + 1

        return torch.zeros_like(input_lengths).fill_(t_dim)