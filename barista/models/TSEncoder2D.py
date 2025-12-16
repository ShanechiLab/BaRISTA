## Source code based on publicly released dilated CNN models as found in
##  SimTS model: https://github.com/xingyu617/SimTS_Representation_Learning/blob/main/models/dilation.py
## and
##  TS2Vec repo: https://github.com/zhihanyue/ts2vec/blob/main/models/dilated_conv.py

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def init_weights(m):
    """
    Relevant reading material:
        https://pytorch.org/docs/stable/nn.init.html
        https://github.com/pytorch/vision/blob/309bd7a1512ad9ff0e9729fbdad043cb3472e4cb/torchvision/models/densenet.py#L203
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Linear):
        nn.init.constant_(m.bias, 0)


class SamePadConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
    ):
        """Padded convolution to ensure same sized input and output."""
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            (1, kernel_size),
            padding=(0, padding),
            stride=(1, stride),
            dilation=(1, dilation),
            groups=groups,
        )

        init_weights(self.conv)

        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, :, : -self.remove]
        return out


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dilation,
        final=False,
        enable_checkpointing=False,
    ):
        """
        Convolutional block implementation.

        Consists of two convolution layers followed by a residual stream.

        Args:
            in_channels: int. Input channel count.
            out_channels: int. Output channel count.
            kernel_size: int. Convolution kernel size.
            stride: int. Convolution stride size.
            dilation: int. Convolution dilation amount.
            final: bool. This is the final convolutional block in the stack. Only relevant for
                using a projection head for the residual stream.
            enable_checkpointing: bool. Enable checkpointing of the intermediate weights if
                desired. Default False.
        """
        super().__init__()

        self.enable_checkpointing = enable_checkpointing

        self.conv1 = SamePadConv(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
        )

        self.conv2 = SamePadConv(
            out_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
        )

        self.projector = (
            nn.Conv2d(
                in_channels, out_channels, kernel_size=(1, 1), stride=(1, stride**2),
            )
            if in_channels != out_channels or final or stride != 1
            else None
        )
        if self.projector is not None:
            init_weights(self.projector)

    def _forward_mini_block(self, x: torch.tensor, block_num: int):
        x = self.conv1(x) if block_num == 1 else self.conv2(x)
        x = F.layer_norm(x, (x.shape[-1],))
        x = F.gelu(x)
        return x

    def forward(self, x: torch.tensor):
        residual = x if self.projector is None else self.projector(x)

        if self.enable_checkpointing:
            x = checkpoint(self._forward_mini_block, x, 1, use_reentrant=False)
            x = checkpoint(self._forward_mini_block, x, 2, use_reentrant=False)
        else:
            x = self._forward_mini_block(x, block_num=1)
            x = self._forward_mini_block(x, block_num=2)

        return x + residual


class DilatedConvEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        kernel_size,
        stride=1,
        enable_checkpointing=False,
    ):
        """Dilated CNN implementation. See ConvBlock for argument definitions."""
        super().__init__()

        self.enable_checkpointing = enable_checkpointing

        self.net = nn.ModuleList(
            [
                ConvBlock(
                    channels[i - 1] if i > 0 else in_channels,
                    channels[i],
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=2**i,
                    final=(i == len(channels) - 1),
                    enable_checkpointing=enable_checkpointing,
                )
                for i in range(len(channels))
            ]
        )

    def forward(self, x: torch.tensor):
        for layer in self.net:
            x = layer(x)
        return x


class TSEncoder2D(nn.Module):
    def __init__(
        self,
        input_dims,
        output_dims,
        hidden_dims=64,
        depth=10,
        kernel_size=3,
        stride=1,
        enable_checkpointing=False,
    ):
        """
        Original source implementation:
            TS2Vec Encoder: https://github.com/zhihanyue/ts2vec/blob/main/models/encoder.py

        See ConvBlock function for argument definitions.
        """
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.enable_checkpointing = enable_checkpointing

        self.feature_extractor = DilatedConvEncoder(
            input_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=kernel_size,
            stride=stride,
            enable_checkpointing=self.enable_checkpointing,
        )

    def forward(self, x: torch.tensor):
        """
        Args:
            x: torch.tensor of shape (1, 1, B * T * D, N) with time (N) along the last axis.
                Note: the additional (1, 1) for the first two axies is to use 2D convs for
                1D convolution operations.
                Note: B=Batch, T=Number of segments, D=Channels.

        Returns:
            Temporal encoded version of the input tensor of shape  (1, 1, B * T * D, N)
        """
        return self.feature_extractor(x)
