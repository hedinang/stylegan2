# StyleGAN2.pytorch (Work In Progress)

## \[:star:\] Please head over to [StyleGAN.pytorch](https://github.com/huangzh13/StyleGAN.pytorch) for my stylegan pytorch implementation.

This repository contains the unofficial PyTorch implementation of the following paper:

> **Analyzing and Improving the Image Quality of StyleGAN**<br>
Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, Timo Aila<br>
> Paper: http://arxiv.org/abs/1912.04958<br><br>
> Abstract: *The style-based GAN architecture (StyleGAN) yields state-of-the-art results in data-driven unconditional generative image modeling. We expose and analyze several of its characteristic artifacts, and propose changes in both model architecture and training methods to address them. In particular, we redesign generator normalization, revisit progressive growing, and regularize the generator to encourage good conditioning in the mapping from latent vectors to images. In addition to improving image quality, this path length regularizer yields the additional benefit that the generator becomes significantly easier to invert. This makes it possible to reliably detect if an image is generated by a particular network. We furthermore visualize how well the generator utilizes its output resolution, and identify a capacity problem, motivating us to train larger models for additional quality improvements. Overall, our improved model redefines the state of the art in unconditional image modeling, both in terms of existing distribution quality metrics as well as perceived image quality.*

## Features

## How to use

## Generated samples

## Reference

- **stylegan2[official]**: https://github.com/NVlabs/stylegan2
- **stylegan2-pytorch**: https://github.com/rosinality/stylegan2-pytorch
- **pro_gan_pytorch**: https://github.com/akanimax/pro_gan_pytorch
- **pytorch_style_gan**: https://github.com/lernapparat/lernapparat

## Thanks

Please feel free to open PRs / issues / suggestions here.

## Due Credit
This code heavily uses NVIDIA's original 
[StyleGAN2](https://github.com/NVlabs/stylegan2) code. We accredit and acknowledge their work here. The 
[Original License](/LICENSE_ORIGINAL.txt) is located in the base directory (file named `LICENSE_ORIGINAL.txt`).