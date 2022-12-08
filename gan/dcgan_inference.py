import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from GAN_models import Discriminator, Generator
from torchsummary import summary
from PIL import Image

#from torchsummary import summary
device = 'cuda'
netG = Generator(ngpu=1).to(device)
disc = Discriminator(ngpu=1).to(device)
print(disc)
summary(netG, ( netG.nz, 1, 1))
num_samples = 5000
import time
t1 = time.time()
with torch.no_grad():
    fixed_noise = torch.randn(num_samples, netG.nz, 1, 1, device=device)


gen_chpt = "../models/gen.pkl"
saved = torch.load(gen_chpt)
netG.load_state_dict(saved['model_state_dict'])
#optimizerG.load_state_dict(saved['optimizer_state_dict'])


#netG.load_state_dict(torch.load("../models/gen.pkl", map_location=torch.device('cpu')))
netG.eval()

fake_imgs = netG(fixed_noise).detach().cpu()
#fake_imgs = torch.nn.Sigmoid()(fake_imgs).detach().cpu()

for i, img in enumerate(tqdm(fake_imgs)):
    A = np.transpose(img.squeeze(),(1,2,0)).numpy()
    im = Image.fromarray(((A+1)/2 * 255).astype(np.uint8))
    im.save(f"../data/gan_samples/sample_imgs/gan_{i}.png")    
t2 = time.time()
print(t2-t1)

#plt.show()