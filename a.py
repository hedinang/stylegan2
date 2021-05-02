from models.GAN import Generator
from torchsummary import summary
import torch
g = Generator(1024)
device = torch.device('cuda')
g = g.to(device)
# input = torch.rand((2, 512)).to(device)
# noise_sample = torch.randn(2, 512, device=device)
# output = g(noise_sample)
# print('aa')
summary(g, (512,))
# gen = Generator()
# gen = gen.to(device)
# test_latents_in = torch.randn(4, 512).to(device)
# test_imgs_out = gen(test_latents_in)

# print('Done.')
