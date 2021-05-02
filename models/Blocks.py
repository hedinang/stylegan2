"""
-------------------------------------------------
   File Name:    Blocks.py
   Author:       Zhonghao Huang
   Date:         2019/12/13
   Description:  Modified from:
                 https://github.com/akanimax/pro_gan_pytorch
                 https://github.com/lernapparat/lernapparat
                 https://github.com/NVlabs/stylegan
-------------------------------------------------
"""

import torch
import torch.nn as nn

from models.op import fused_leaky_relu
from models.CustomLayers import EqualizedModConv2d, Upsample, EqualizedConv2d, BlurLayer, StddevLayer, View, EqualizedLinear
from collections import OrderedDict


class ToRGB(nn.Module):
    def __init__(self, dlatent_size, in_channel, num_channels, resample_kernel=None):
        super(ToRGB, self).__init__()

        if resample_kernel is None:
            resample_kernel = [1, 3, 3, 1]

        self.upsample = Upsample(resample_kernel)
        self.conv = EqualizedModConv2d(dlatent_size=dlatent_size, in_channel=in_channel,
                                       out_channel=num_channels, kernel=1, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(
            1, num_channels, 1, 1), requires_grad=True)

    def forward(self, x, dlatents_in_range, y=None):
        x = self.conv(x, dlatents_in_range)
        out = x + self.bias  # act='linear'

        if y is not None:  # architecture='skip'
            y = self.upsample(y)
            out = out + y

        return out


class ModConvLayer(nn.Module):
    def __init__(self, dlatent_size, in_channel, out_channel, kernel, up=False, down=False, use_noise=True):
        super(ModConvLayer, self).__init__()

        self.conv = EqualizedModConv2d(dlatent_size=dlatent_size,
                                       in_channel=in_channel, out_channel=out_channel,
                                       kernel=kernel, up=up, down=down)
        self.bias = nn.Parameter(torch.zeros(out_channel), requires_grad=True)

        self.use_noise = use_noise
        if self.use_noise:
            self.noise_strength = nn.Parameter(
                torch.zeros(1), requires_grad=True)

    def forward(self, x, dlatents_in_range, noise_input=None):
        x = self.conv(x, dlatents_in_range)

        if self.use_noise:
            if noise_input is None:
                batch, _, height, width = x.shape
                noise_input = x.new_empty(batch, 1, height, width).normal_()
            print('noise input ', noise_input.shape)
            x += self.noise_strength * noise_input

        out = fused_leaky_relu(x, self.bias)  # act='lrelu'

        return out


class InputBlock(nn.Module):
    def __init__(self, dlatent_size, num_channels, in_fmaps, out_fmaps, use_noise):
        super(InputBlock, self).__init__()

        self.const = nn.Parameter(torch.randn(
            1, in_fmaps, 4, 4), requires_grad=True)
        self.conv = ModConvLayer(dlatent_size=dlatent_size,
                                 in_channel=in_fmaps,
                                 out_channel=out_fmaps,
                                 kernel=3, use_noise=use_noise)
        self.to_rgb = ToRGB(dlatent_size=dlatent_size,
                            in_channel=out_fmaps,
                            num_channels=num_channels)

    def forward(self, dlatents_in):
        x = self.const.repeat(dlatents_in.shape[0], 1, 1, 1)
        x = self.conv(x, dlatents_in[:, 0])
        y = self.to_rgb(x, dlatents_in[:, 1])

        return x, y


class GSynthesisBlock(nn.Module):
    """
    Building blocks for main layers
    """

    def __init__(self, dlatent_size, num_channels, res, in_fmaps, out_fmaps, use_noise):
        super(GSynthesisBlock, self).__init__()

        self.res = res

        self.conv0_up = ModConvLayer(dlatent_size=dlatent_size,
                                     in_channel=in_fmaps,
                                     out_channel=out_fmaps,
                                     kernel=3, up=True, use_noise=use_noise)
        self.conv1 = ModConvLayer(dlatent_size=dlatent_size,
                                  in_channel=out_fmaps,
                                  out_channel=out_fmaps,
                                  kernel=3, use_noise=use_noise)
        self.to_rgb = ToRGB(dlatent_size=dlatent_size,
                            in_channel=out_fmaps,
                            num_channels=num_channels)

    def forward(self, x, dlatents_in, y):
        x = self.conv0_up(x, dlatents_in[:, self.res * 2 - 5])
        x = self.conv1(x, dlatents_in[:, self.res * 2 - 4])

        # architecture='skip'
        y = self.to_rgb(x, dlatents_in[:, self.res * 2 - 3], y)

        return x, y


class DiscriminatorBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, gain, use_wscale, activation_layer, blur_kernel):
        super().__init__(OrderedDict([
            ('conv0', EqualizedConv2d(in_channels, in_channels,
             kernel_size=3, gain=gain, use_wscale=use_wscale)),
            # out channels nf(res-1)
            ('act0', activation_layer),
            ('blur', BlurLayer(kernel=blur_kernel)),
            ('conv1_down', EqualizedConv2d(in_channels, out_channels, kernel_size=3,
                                           gain=gain, use_wscale=use_wscale, downscale=True)),
            ('act1', activation_layer)]))


class DiscriminatorTop(nn.Sequential):
    def __init__(self,
                 mbstd_group_size,
                 mbstd_num_features,
                 in_channels,
                 intermediate_channels,
                 gain, use_wscale,
                 activation_layer,
                 resolution=4,
                 in_channels2=None,
                 output_features=1,
                 last_gain=1):
        """
        :param mbstd_group_size:
        :param mbstd_num_features:
        :param in_channels:
        :param intermediate_channels:
        :param gain:
        :param use_wscale:
        :param activation_layer:
        :param resolution:
        :param in_channels2:
        :param output_features:
        :param last_gain:
        """

        layers = []
        if mbstd_group_size > 1:
            layers.append(('stddev_layer', StddevLayer(
                mbstd_group_size, mbstd_num_features)))

        if in_channels2 is None:
            in_channels2 = in_channels

        layers.append(('conv', EqualizedConv2d(in_channels + mbstd_num_features, in_channels2, kernel_size=3,
                                               gain=gain, use_wscale=use_wscale)))
        layers.append(('act0', activation_layer))
        layers.append(('view', View(-1)))
        layers.append(('dense0', EqualizedLinear(in_channels2 * resolution * resolution, intermediate_channels,
                                                 gain=gain, use_wscale=use_wscale)))
        layers.append(('act1', activation_layer))
        layers.append(('dense1', EqualizedLinear(intermediate_channels, output_features,
                                                 gain=last_gain, use_wscale=use_wscale)))

        super().__init__(OrderedDict(layers))
