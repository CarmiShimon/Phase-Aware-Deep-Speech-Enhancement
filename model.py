import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class BasicConvBlock(nn.Module):

    def __init__(self, channels, kernel, stride, padding):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = torch.relu(x)
        out = self.bn1(out)
        out = self.conv1(out)
        out += self.shortcut(x)
        return out


class MagNet(nn.Module):

    def __init__(self, input_size=257, num_channels=[1536]*15):
        super().__init__()
        self.linear = nn.Linear(in_features=257, out_features=1536)
        self.conv_layers = self._make_layers(BasicConvBlock, num_channels, kernel=5, stride=1, padding=2)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(in_features=1536, out_features=257)


    def _make_layers(self, BasicConvBlock, num_channels, kernel, stride, padding):
        layers = []
        for channels in num_channels:
            layers += [BasicConvBlock(channels, kernel, stride, padding)]
        return nn.Sequential(*layers)

    def forward(self, x):
        inp = x
        x = torch.transpose(x, 1, 2)  # swap between freq. and time
        x = self.linear(x)
        x = torch.transpose(x, 1, 2)  # swap between freq. and time
        x = self.conv_layers(x)
        x = torch.transpose(x, 1, 2)  # swap between freq. and time
        x = self.linear2(x)
        x = self.sigmoid(x)
        recon_mag = torch.transpose(inp, 1, 2) * x
        return recon_mag


class PhaseNet(nn.Module):

    def __init__(self, input_size=257*3, num_channels=[1024]*6):
        super().__init__()
        self.linear = nn.Linear(in_features=257*3, out_features=1024)
        self.conv_layers = self._make_layers(BasicConvBlock, num_channels, kernel=5, stride=1, padding=2)
        self.linear2 = nn.Linear(in_features=1024, out_features=514)

    def _make_layers(self, BasicConvBlock, num_channels, kernel, stride, padding):
        layers = []
        for channels in num_channels:
            layers += [BasicConvBlock(channels, kernel, stride, padding)]
        return nn.Sequential(*layers)

    def forward(self, cos_x, sin_x, estim_mag):
        x = torch.cat(tensors=(cos_x, sin_x, estim_mag.transpose(1, 2)), axis=1)
        x = torch.transpose(x, 1, 2)  # swap between freq. and time
        x = self.linear(x)
        x = torch.transpose(x, 1, 2)  # swap between freq. and time
        x = self.conv_layers(x)
        x = torch.transpose(x, 1, 2)  # swap between freq. and time
        x = self.linear2(x).transpose(1, 2) + torch.cat(tensors=(cos_x, sin_x), dim=1)
        x = torch.cat((x[:, :257, :].unsqueeze(-1), x[:, 257:, :].unsqueeze(-1)), dim=-1)
        return nn.functional.normalize(x, p=2.0, dim=-1)  # norm2


class PASE(nn.Module):

    def __init__(self, n_fft=512, hop_length=128):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mag = MagNet()
        self.phase = PhaseNet()

    def forward(self, x):
        stft = torch.stft(x, self.n_fft, self.hop_length)
        mag = torch.sqrt(stft[..., 0] ** 2 + stft[..., 1] ** 2)
        phase = torch.atan2(stft[..., 1], stft[..., 0])
        cos_x, sin_x = torch.cos(phase), torch.sin(phase)
        estim_mag = self.mag(mag)  # real part of stft
        estim_phase = self.phase(cos_x, sin_x, estim_mag)  # imag part of stft
        estim_phase = estim_phase.reshape(-1, 257, 5, 2)
        # estim_stft = torch.cat((torch.cos(torch.transpose(estim_phase, 1, 2)).unsqueeze(-1) * estim_mag.unsqueeze(-1), torch.sin(torch.transpose(estim_phase, 1, 2)).unsqueeze(-1) * estim_mag.unsqueeze(-1)), dim=-1).transpose(1, 2)
        estim_stft = torch.cat(((estim_phase[..., 0] * torch.transpose(estim_mag, 1, 2)).unsqueeze(-1), (estim_phase[..., 1] * torch.transpose(estim_mag, 1, 2)).unsqueeze(-1)), dim=-1)
        recon_signal = torch.istft(estim_stft, self.n_fft, self.hop_length)

        return recon_signal
