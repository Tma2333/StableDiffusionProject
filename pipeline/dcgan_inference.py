import matplotlib.pyplot as plt
import numpy as np
import torch
from GAN_models import Discriminator, Generator

device = 'cpu'
netG = Generator(ngpu=0)

for i in range(50):
    fixed_noise = torch.randn(1, netG.nz, 1, 1, device=device)


    netG.load_state_dict(torch.load("../models/gen.pkl", map_location=torch.device('cpu')))
    netG.eval()

    fake_img = netG(fixed_noise).detach().cpu()

    plt.figure()
    plt.imshow(np.transpose(fake_img.squeeze(),(1,2,0)))


    plt.savefig(f"../data/gan_samples/gan_{i}.png")
#plt.show()