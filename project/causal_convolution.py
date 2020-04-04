import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class TemporalBlock(nn.Module):
    """
    Block of two dilated causal convolutions
    with a residual connection
    """
    def __init__(
            self, in_channels, 
            out_channels, kernel_size, dilation):
        super().__init__()

        stride = 1
        padding = (kernel_size - 1) * dilation
        self.T = padding

        self.conv1 = weight_norm(
                nn.Conv1d(
                    in_channels, out_channels,
                    kernel_size, stride, padding, dilation))
        self.conv2 = weight_norm(
                nn.Conv1d(
                    out_channels, out_channels,
                    kernel_size, stride, padding, dilation))

        self.resample = nn.Conv1d(
            in_channels, out_channels, kernel_size=1
        ) if in_channels != out_channels else nn.Sequential()
        self.leaky = nn.LeakyReLU(.2, inplace=True)

    def forward(self, X):
        H = self.leaky(self.conv1(X)[...,:-self.T])
        H = self.conv2(H)[..., :-self.T]
        return self.leaky(H + self.resample(X))


class DCConvStack(nn.Module):
    """
    Stack of temporal blocks, namely
    exponentially dilated causal convolutions
    """
    def __init__(self, channels, kernel_size, depth=3):
        """
        Args:
        ----
        channels    list of integers of length `depth+1`;
                    channels[0] - number of net's input channels,
                    channels[1:] - numbers of output channels of its layers

        """
        super().__init__()
        dilation = 1
        main_list = []
        for i in range(depth):
            main_list.append(
                TemporalBlock(
                    channels[i], channels[i+1],
                    kernel_size, dilation))

        self.main = nn.Sequential(*main_list)

    def forward(self, X):
        return self.main(X)
