import glob

import torchvision.transforms as transforms
from PIL import Image
import torch
import os

def image_loader(image_name, device, imsize):
    
    loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


    image = Image.open(image_name)
    print(image_name, image.mode)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)




def get_train_test(folder_path, imsize, device, max_imgs):
    imgs = glob.glob(os.path.join(folder_path,"*.jpg"))[:max_imgs]

    N = len(imgs)
    train, test = imgs[:(N*2)//3], imgs[(N*2)//3:]

    train = torch.concat([image_loader(path, device, imsize) for path in train], 0)
    test = torch.concat([image_loader(path, device, imsize) for path in test], 0)
    return train, test
    


def batch_loader(filepaths, device, imsize):

    imgs = []
    for fp in filepaths:
        img = image_loader(fp, device,  imsize)
        imgs.append(img)
    return torch.cat(imgs,dim=0)