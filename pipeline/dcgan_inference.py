import matplotlib.pyplot as plt
import numpy as np
import torch
from GAN_models import Discriminator, Generator

device = 'cpu'
nz = 100
fixed_noise = torch.randn(1, nz, 1, 1, device=device)

netG = Generator(ngpu=0)

netG.load_state_dict(torch.load("../models/gen.pkl"))
netG.eval()

fake_img = netG(fixed_noise).detach().cpu()

plt.figure()
plt.imshow(np.transpose(fake_img.squeeze(),(1,2,0)))
plt.show()