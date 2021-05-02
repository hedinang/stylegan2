"""
-------------------------------------------------
   File Name:    CustomLayers.py
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
import torch.nn.functional as F
import numpy as np
from models.op import fused_leaky_relu, upfirdn2d


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k
class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)
class StddevLayer(nn.Module):
    def __init__(self, group_size=4, num_new_features=1):
        super().__init__()
        self.group_size = group_size
        self.num_new_features = num_new_features

    def forward(self, x):
        b, c, h, w = x.shape
        group_size = min(self.group_size, b)
        y = x.reshape([group_size, -1, self.num_new_features,
                       c // self.num_new_features, h, w])
        y = y - y.mean(0, keepdim=True)
        y = (y ** 2).mean(0, keepdim=True)
        y = (y + 1e-8) ** 0.5
        y = y.mean([3, 4, 5], keepdim=True).squeeze(3)  # don't keep the meaned-out channels
        y = y.expand(group_size, -1, -1, h, w).clone().reshape(b, self.num_new_features, h, w)
        z = torch.cat([x, y], dim=1)
        return z
class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor,
                        down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1,
                        down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out

class Upscale2d(nn.Module):
    @staticmethod
    def upscale2d(x, factor=2, gain=1):
        assert x.dim() == 4
        if gain != 1:
            x = x * gain
        if factor != 1:
            shape = x.shape
            x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1, -1, -1, factor, -1, factor)
            x = x.contiguous().view(shape[0], shape[1], factor * shape[2], factor * shape[3])
        return x

    def __init__(self, factor=2, gain=1):
        super().__init__()
        assert isinstance(factor, int) and factor >= 1
        self.gain = gain
        self.factor = factor

    def forward(self, x):
        return self.upscale2d(x, factor=self.factor, gain=self.gain)


class Downscale2d(nn.Module):
    def __init__(self, factor=2, gain=1):
        super().__init__()
        assert isinstance(factor, int) and factor >= 1
        self.factor = factor
        self.gain = gain
        if factor == 2:
            f = [np.sqrt(gain) / factor] * factor
            self.blur = BlurLayer(kernel=f, normalize=False, stride=factor)
        else:
            self.blur = None

    def forward(self, x):
        assert x.dim() == 4
        # 2x2, float32 => downscale using _blur2d().
        if self.blur is not None and x.dtype == torch.float32:
            return self.blur(x)

        # Apply gain.
        if self.gain != 1:
            x = x * self.gain

        # No-op => early exit.
        if self.factor == 1:
            return x

        # Large factor => downscale using tf.nn.avg_pool().
        # NOTE: Requires tf_config['graph_options.place_pruned_graph']=True to work.
        return F.avg_pool2d(x, self.factor)

class EqualizedLinear(nn.Module):
    """Linear layer with equalized learning rate and custom learning rate multiplier."""

    def __init__(self, in_dim, out_dim, bias=True, bias_init=0., activation=None,
                 gain=1., use_wscale=True, lrmul=1.):
        super(EqualizedLinear, self).__init__()

        # Equalized learning rate and custom learning rate multiplier.
        he_std = gain * in_dim ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul

        self.weight = torch.nn.Parameter(torch.randn(
            out_dim, in_dim) * init_std, requires_grad=True)

        if bias:
            self.bias = nn.Parameter(torch.zeros(
                out_dim).fill_(bias_init), requires_grad=True)
            self.b_mul = lrmul
        else:
            self.bias = None

        self.activation = activation

    def forward(self, x):
        if self.activation == 'lrelu':  # act='lrelu'
            out = F.linear(x, self.weight * self.w_mul)
            out = fused_leaky_relu(out, self.bias * self.b_mul)
        else:
            out = F.linear(x, self.weight * self.w_mul,
                           bias=self.bias * self.b_mul)

        return out


class EqualizedModConv2d(nn.Module):
    def __init__(self, dlatent_size, in_channel, out_channel, kernel,
                 up=False, down=False, demodulate=True, resample_kernel=None,
                 gain=1., use_wscale=True, lrmul=1.):
        """
        """
        super(EqualizedModConv2d, self).__init__()

        assert not (up and down)
        assert kernel >= 1 and kernel % 2 == 1

        if resample_kernel is None:
            resample_kernel = [1, 3, 3, 1]

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.up = up
        self.down = down
        self.demodulate = demodulate
        self.kernel = kernel

        if up:
            factor = 2
            p = (len(resample_kernel) - factor) - (kernel - 1)
            self.blur = Blur(resample_kernel, pad=(
                (p + 1) // 2 + factor - 1, p // 2 + 1), upsample_factor=factor)

        if down:
            factor = 2
            p = (len(resample_kernel) - factor) + (kernel - 1)
            self.blur = Blur(resample_kernel, pad=((p + 1) // 2, p // 2))

        self.mod = EqualizedLinear(
            in_dim=dlatent_size, out_dim=in_channel, bias_init=1.)

        he_std = gain * (in_channel * kernel ** 2) ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul

        self.weight = torch.nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel, kernel) * init_std, requires_grad=True)

    def forward(self, x, y):
        batch, in_channel, height, width = x.shape

        # Modulate
        s = self.mod(y).view(batch, 1, in_channel, 1, 1)
        ww = self.w_mul * self.weight * s

        # Demodulate
        if self.demodulate:
            # [BO] Scaling factor.
            d = torch.rsqrt(ww.pow(2).sum([2, 3, 4]) + 1e-8)
            # [BOIkk] Scale output feature maps.
            ww *= d.view(batch, self.out_channel, 1, 1, 1)

        weight = ww.view(batch * self.out_channel,
                         in_channel, self.kernel, self.kernel)

        if self.up:
            x = x.view(1, batch * in_channel, height, width)
            weight = weight.view(batch, self.out_channel,
                                 in_channel, self.kernel, self.kernel)
            weight = weight.transpose(1, 2).reshape(batch * in_channel, self.out_channel,
                                                    self.kernel, self.kernel)
            out = F.conv_transpose2d(
                x, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)
        elif self.down:
            x = self.blur(x)
            _, _, height, width = x.shape
            x = x.view(1, batch * in_channel, height, width)
            out = F.conv2d(x, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
        else:
            x = x.view(1, batch * in_channel, height, width)
            out = F.conv2d(x, weight, padding=self.kernel // 2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.up}, downsample={self.down})'
        )

class EqualizedConv2d(nn.Module):
    """Conv layer with equalized learning rate and custom learning rate multiplier."""

    def __init__(self, input_channels, output_channels, kernel_size, stride=1, gain=2 ** 0.5, use_wscale=False,
                 lrmul=1, bias=True, intermediate=None, upscale=False, downscale=False):
        super().__init__()
        if upscale:
            self.upscale = Upscale2d()
        else:
            self.upscale = None
        if downscale:
            self.downscale = Downscale2d()
        else:
            self.downscale = None
        he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5)  # He init
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(
            torch.randn(output_channels, input_channels, kernel_size, kernel_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
            self.b_mul = lrmul
        else:
            self.bias = None
        self.intermediate = intermediate

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul

        have_convolution = False
        if self.upscale is not None and min(x.shape[2:]) * 2 >= 128:
            # this is the fused upscale + conv from StyleGAN, sadly this seems incompatible with the non-fused way
            # this really needs to be cleaned up and go into the conv...
            w = self.weight * self.w_mul
            w = w.permute(1, 0, 2, 3)
            # probably applying a conv on w would be more efficient. also this quadruples the weight (average)?!
            w = F.pad(w, [1, 1, 1, 1])
            w = w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]
            x = F.conv_transpose2d(x, w, stride=2, padding=(w.size(-1) - 1) // 2)
            have_convolution = True
        elif self.upscale is not None:
            x = self.upscale(x)

        downscale = self.downscale
        intermediate = self.intermediate
        if downscale is not None and min(x.shape[2:]) >= 128:
            w = self.weight * self.w_mul
            w = F.pad(w, [1, 1, 1, 1])
            # in contrast to upscale, this is a mean...
            w = (w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]) * 0.25  # avg_pool?
            x = F.conv2d(x, w, stride=2, padding=(w.size(-1) - 1) // 2)
            have_convolution = True
            downscale = None
        elif downscale is not None:
            assert intermediate is None
            intermediate = downscale

        if not have_convolution and intermediate is None:
            return F.conv2d(x, self.weight * self.w_mul, bias, padding=self.kernel_size // 2)
        elif not have_convolution:
            x = F.conv2d(x, self.weight * self.w_mul, None, padding=self.kernel_size // 2)

        if intermediate is not None:
            x = intermediate(x)

        if bias is not None:
            x = x + bias.view(1, -1, 1, 1)
        return x
class BlurLayer(nn.Module):
    def __init__(self, kernel=None, normalize=True, flip=False, stride=1):
        super(BlurLayer, self).__init__()
        if kernel is None:
            kernel = [1, 2, 1]
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel[None, None]
        if normalize:
            kernel = kernel / kernel.sum()
        if flip:
            kernel = kernel[:, :, ::-1, ::-1]
        self.register_buffer('kernel', kernel)
        self.stride = stride

    def forward(self, x):
        # expand kernel channels
        kernel = self.kernel.expand(x.size(1), -1, -1, -1)
        x = F.conv2d(
            x,
            kernel,
            stride=self.stride,
            padding=int((self.kernel.size(2) - 1) / 2),
            groups=x.size(1)
        )
        return x

if __name__ == '__main__':
    print('Done.')
