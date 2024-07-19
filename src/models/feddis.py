import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.functional import adjust_gamma
from typing import Sequence
from monai.networks.blocks import Convolution
from .layers.blocks import conv_bn_relu
from .losses import FedDisLoss


class Encoder(nn.Sequential):
    def __init__(self, dimensions: int, in_channels: int, channels: Sequence[int], strides: int,
                 kernel_size=5, norm='batch', act='leakyrelu', name_prefix='_'):
        """
        :param dimensions:
        :param in_channels:
        :param channels:
        :param strides:
        :param kernel_size:
        :param norm:
        :param act:
        :param name_prefix:
        """
        super(Encoder, self).__init__()

        layer_channels = in_channels
        for i, c in enumerate(channels):
            self.add_module(name_prefix + "encode_%i" % i,
                            Convolution(spatial_dims=dimensions, in_channels=layer_channels, out_channels=c,
                                        strides=strides, kernel_size=kernel_size, norm=norm, act=act))
            layer_channels = c


class Decoder(nn.Sequential):
    def __init__(self, dimensions: int, in_channels: int, channels: Sequence[int], out_ch: int, strides: int,
                 kernel_size: int = 5, norm: str = 'batch', act: str = 'leakyrelu', act_final: str = 'sigmoid',
                 bottleneck: bool = False, skip: bool = False, add_final: bool = True, name_prefix: str = '_'):
        """
        :param dimensions:
        :param in_channels:
        :param channels:
        :param strides:
        :param kernel_size:
        :param norm:
        :param act:
        :param act_final:
        :param bottleneck:
        :param skip:
        :param add_final:
        :param name_prefix:
        """
        super(Decoder, self).__init__()

        decode_channel_list = list(channels[-1::-1])
        layer_channels = in_channels
        for i, c in enumerate(decode_channel_list):
            if i == 0 and bottleneck:
                continue

            self.add_module(name_prefix + "decode_%i" % i,
                            Convolution(spatial_dims=dimensions, in_channels=layer_channels, out_channels=c,
                                        strides=strides, kernel_size=kernel_size, norm=norm, act=act, is_transposed=True))
            layer_channels = c

        if add_final:
            self.add_module(name_prefix + "decode_final",
                            Convolution(spatial_dims=dimensions, in_channels=layer_channels, out_channels=out_ch,
                                        strides=1, kernel_size=1, act=act_final))

class GlobalFedDis(nn.Module):

    def __init__(self, dimensions=2, in_channels=3, channels=[128, 256, 512], out_ch=2, strides=2,
                 kernel_size=5, norm='batch', act='leakyrelu', act_final='softmax', bottleneck=False, skip=False):
        """
        :param dimensions:
        :param in_channels:
        :param channels:
        :param strides:
        :param kernel_size:
        :param norm:
        :param act:
        :param act_final:
        :param bottleneck:
        :param skip:
        """
        super().__init__()

        self.encode = Encoder(dimensions=dimensions, in_channels=in_channels, channels=channels, strides=strides,
                              kernel_size=kernel_size, norm=norm, act=act)

        self.decode = Decoder(dimensions=dimensions, in_channels=channels[-1], channels=channels, out_ch=in_channels,
                              strides=strides, kernel_size=kernel_size, norm=norm, act=act, act_final="tanh",
                              bottleneck=bottleneck, skip=skip, add_final=True)

        self.decoder_seg = Decoder(dimensions=dimensions, in_channels=channels[-1], channels=channels, out_ch=out_ch,
                              strides=strides, kernel_size=kernel_size, norm=norm, act=act, act_final=act_final,
                              bottleneck=bottleneck, skip=skip, add_final=True)

    def forward(self,  x: torch):
        z = self.encode(x)
        x_ = self.decode(z)
        seg = self.decoder_seg(z)

        return seg, x_, z, None, None, None, None

    def get_criterion(self, train=True):
        return FedDisLoss(train)


class LocalFedDis(nn.Module):

    def __init__(self, dimensions=2, in_channels=3, channels=[128, 256, 512], out_ch=2, strides=2,
                 channels_app=[32, 64, 128], kernel_size=5, norm='batch', act='leakyrelu', act_final='softmax',
                 bottleneck=False, skip=False):
        """
        :param dimensions:
        :param in_channels:
        :param channels:
        :param out_ch:
        :param strides:
        :param channels_app:
        :param kernel_size:
        :param norm:
        :param act:
        :param act_final:
        :param bottleneck:
        :param skip:
        """
        super().__init__()

        self.encode_content = Encoder(dimensions=dimensions, in_channels=in_channels, channels=channels, strides=strides,
                                    kernel_size=kernel_size, norm=norm, act=act)
        self.encode_style = Encoder(dimensions=dimensions, in_channels=in_channels, channels=channels_app,
                                  strides=strides, kernel_size=kernel_size, norm=norm, act=act)

        self.decode_content = Decoder(dimensions=dimensions, in_channels=channels[-1], channels=channels,
                              out_ch=in_channels, strides=strides, kernel_size=kernel_size, norm=norm, act=act,
                              act_final=act, bottleneck=bottleneck, skip=skip, add_final=True)
        
        self.decode_style = Decoder(dimensions=dimensions, in_channels=channels_app[-1], channels=channels_app,
                              out_ch=in_channels, strides=strides, kernel_size=kernel_size, norm=norm, act=act,
                              act_final=act, bottleneck=bottleneck, skip=skip, add_final=True)

        self.decode_content_seg = Decoder(dimensions=dimensions, in_channels=channels[-1], channels=channels,
                              out_ch=out_ch, strides=strides, kernel_size=kernel_size, norm=norm, act=act,
                              act_final=act_final, bottleneck=bottleneck, skip=skip, add_final=True)

        self.merger = nn.Sequential(nn.Conv2d(in_channels*2, in_channels, 1, stride=1),
                                     nn.Tanh())
        
        self.projector = nn.Conv2d(channels[-1], channels_app[-1], 1, stride=1)

    def forward(self,  x: torch):
        # gamma = np.random.uniform(low=0.5, high=2.0)
        # gamma_x = adjust_gamma(x.clone(), gamma)

        z_s = self.encode_content(x)
        z_a = self.encode_style(x)

        # z_s_shift = self.encode_content(gamma_x)

        x_s = self.decode_content(z_s)
        seg = self.decode_content_seg(z_s)

        x_a = self.decode_style(z_a)

        x_ = self.merger(torch.cat((x_s, x_a), 1))

        z_s_proj = self.projector(z_s)

        return seg, x_, z_s, z_a, None, z_s_proj, x

    def get_criterion(self, train=True):
        return FedDisLoss(train)