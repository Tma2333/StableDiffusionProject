## GAN code and models are based on the following tutorial
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

from __future__ import print_function
#%matplotlib inline
import argparse
import time
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import glob
from PIL import Image


from GAN_models import Generator, Discriminator


# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
dataroot = "../data/cats"







# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)






def make_img_path_list(use_dir_num):
    '''
    '''
    train_img_list = []
    train_img_list += glob.glob(os.path.join(dataroot+"/cat_faces/dataset-part1/","*.png"))
    train_img_list += glob.glob(os.path.join(dataroot+"/cat_faces/dataset-part2/","*.png"))
    train_img_list += glob.glob(os.path.join(dataroot+"/cat_faces/dataset-part3/","*.png"))
    return train_img_list
    for i in range(use_dir_num):
        use_dir = dataroot+f"/CAT_0{i}"
        paths = glob.glob(os.path.join(use_dir,"*.jpg"))
        train_img_list+=paths
        print("num_img",len(train_img_list))
    
    
    for path_tuple in os.walk(dataroot+"/cat_breeds/"):
        use_dir = glob.glob(os.path.join(dataroot+"/cat_breeds/"+path_tuple[0],"*.jpg"))
        train_img_list+=paths
        print("num_img",len(train_img_list))

    train_img_list+= glob.glob(os.path.join(dataroot+"/dog vs cat/dataset/training_set/cats/","*.jpg"))
    print("num_img",len(train_img_list))
    train_img_list += glob.glob(os.path.join(dataroot+"/dog vs cat/dataset/test_set/cats/","*.jpg"))
    print("num_img",len(train_img_list))
    return train_img_list


                           
class GAN_Dataset(data.Dataset):
    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        img = self.transform(img)
        #img = img.unsqueeze(0)
        return img
    


# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.






def train(netG, netD, optimizerG, optimizerD, dataloader, checkpoint=False, num_epochs=5): 

    fixed_noise = torch.randn(64, netG.nz, 1, 1, device=device)

    epoch = 0
    gen_chpt = "../models/gen_checkpoint.pkl"
    disc_chpt = "../models/disc_checkpoint.pkl"
    if checkpoint:
        saved = torch.load(gen_chpt)
        netG.load_state_dict(saved['model_state_dict'])
        optimizerG.load_state_dict(saved['optimizer_state_dict'])
        epoch = saved['epoch']
        errG = saved['loss']
        G_losses = saved['loss_list'].tolist()

        saved = torch.load(disc_chpt)
        netD.load_state_dict(saved['model_state_dict'])
        optimizerD.load_state_dict(saved['optimizer_state_dict'])
        errD = saved['loss']

        D_losses = saved['loss_list'].tolist()
        print(f"Starting from epoch: {epoch}, loss: errD={errD.item()}, errG={errG.item()}")

    else:
        netG.apply(weights_init)
        netD.apply(weights_init)
    # Lists to keep track of progress
        G_losses = []
        D_losses = []
    print("Starting Training Loop...", device)
    # For each epoch
    criterion = nn.BCEWithLogitsLoss()
    netD.train()
    netG.train()
    while epoch < num_epochs:
        t1 = time.time()
        epoch+=1
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data.to(device)
            b_size = data.shape[0]
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, netG.nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
        if (epoch % 10 == 0) or (epoch == num_epochs-1) :
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            plt.figure()
            plt.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True), (1,2,0)))
            plt.tight_layout()
            plt.savefig(f"../data/gan_samples/epochs_{epoch}.png")

        

        
        if epoch % 10 == 0:
            save_model([gen_chpt, disc_chpt], epoch, [netG, netD],
             [optimizerG, optimizerD], [errG, errD], [G_losses, D_losses])

        diff = time.time()-t1
        print(f"Time for one epoch: {diff}")
            
    generator_filepath = "../models/gen.pkl"
    discriminator_filepath = "../models/disc.pkl"
    save_model([generator_filepath, discriminator_filepath], epoch, [netG, netD],
             [optimizerG, optimizerG], [errG, errD], [G_losses, D_losses] )

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../data/losses.png")



## Model saving
def save_model(paths, epoch, models,  optims, losses, losses_lists):
    for path, model, optim, loss, loss_list in zip(paths, models, optims, losses, losses_lists):
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'loss': loss,
                    'loss_list': np.array(loss_list)
                    }, path)

def get_dataloader(image_size, batch_size):
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               transforms.RandomAdjustSharpness(sharpness_factor=2),
                               transforms.RandomVerticalFlip(0.5) ## Flip image 50% of the time
])

    dataset = GAN_Dataset(
    file_list=train_img_list,
    transform=transform)


    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=5)
  
    
    return dataloader

if __name__=='__main__':
    ngpu = 1

    batch_size = 128
    image_size = 64


    num_epochs = 300
    lr = 0.00005
    beta1 = 0.5

    train_img_list = make_img_path_list(7)
    
    

    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    netG = Generator(ngpu).to(device)
    netD = Discriminator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))
        netG = nn.DataParallel(netG, list(range(ngpu)))


    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    dataloader = get_dataloader(image_size, batch_size)
    train(
        netG, netD, optimizerG, optimizerD, dataloader, True, num_epochs
    )
