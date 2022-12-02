import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torch

from datafetching import *
from model import get_input_optimizer, get_loss
from image_blender import Normalization, imshow, get_vgg19, train
from lossFunctions import ContentLoss, StyleLoss
import numpy as np


def get_style_model_and_losses_w_references(cnn, input_imgs, reference_imgs,  device, content_weight, style_weight):


    """
    Point of model is to preserve content equal to original input, but blend the images in style
    """
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    original_input = input_imgs.clone().detach()
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)
    model_copy = nn.Sequential(normalization)
    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)
        model_copy.add_module(name, layer)
        if name in content_layers:
            # add content loss:
            
            targets = model_copy(original_input).detach()
            
            content_loss = ContentLoss(targets, content_weight)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            targets = model_copy(reference_imgs).detach()
            style_loss = StyleLoss(targets, style_weight)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def run_style_transfer(cnn, input_imgs, reference_imgs, num_steps=300,
                       style_weight=1000000, content_weight=1, device='cpu'):
    """Run the style transfer."""
    print('Building the style transfer model..', input_imgs.shape)
    model, style_losses, content_losses = get_style_model_and_losses_w_references(cnn, input_imgs,
         reference_imgs, device, content_weight, style_weight)

    print("input img", input_imgs.shape)
    output = train(model, style_losses, content_losses,
        input_imgs, num_steps, lr=0.05)
    return output








if __name__=='__main__':
    imsize = (224, 224) # if torch.cuda.is_available() else 128  # use small size if no gpu
    device= 'cuda'
  
    model = get_vgg19(device)
    content_weight=50
    style_weight=100000000

    #imgs_vg = batch_loader(glob.glob(os.path.join("../data/Arles/","*.jpg"))[:n], device, imsize)
    n = 5
    imgs_ref = batch_loader(["../data/pikene_munch.jpg", "../data/scream.jpg"], device, imsize)
    
    folder = "narcissus_seed132"
    paths = glob.glob(os.path.join(f"../data/{folder}/","*guidance.png"))
    paths.sort(key = lambda x: int("".join([c for c in x if c.isnumeric()])))
    imgs_samples = batch_loader(paths, device, imsize)
    
   


    
    output = run_style_transfer(model, input_imgs=imgs_samples, reference_imgs=imgs_ref, content_weight=content_weight, style_weight=style_weight, 
                    num_steps=50, device=device)
    for i, out in enumerate(output):
        try:
            os.mkdir(f"../data/copier_{folder}")
        except:
            pass
        fpath = f"../data/copier_{folder}/Output_{i}.png"
        A = np.transpose(out.detach().cpu().squeeze(),(1,2,0)).numpy()
        im = Image.fromarray((A * 255).astype(np.uint8))
        im.save(fpath)  