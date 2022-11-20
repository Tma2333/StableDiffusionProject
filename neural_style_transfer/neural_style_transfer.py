"""
Based on https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
"""

import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import torch


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import glob

from evaluate import evaluate


from model import *
from datafetching import *



def run_style_transfer(cnn, content_img, style_imgs, input_img, content_layers, style_layers, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
         style_imgs, content_img, device)

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()
            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


def get_loss(style_losses, content_losses, style_weight, content_weight):
    style_score = 0
    content_score = 0

    for sl in style_losses:
        style_score += sl.loss
    for cl in content_losses:
        content_score += cl.loss

    style_score *= style_weight
    content_score *= content_weight
    return style_score, content_score

def van_gogh_analysis(output_image, content_image, device, imsize):
    #https://www.kaggle.com/code/amyjang/monet-cyclegan-tutorial/data
    #https://www.kaggle.com/datasets/ipythonx/van-gogh-paintings
    
 
    folder_path = "../data/VincentVanGogh/Arles/"

    imgs_vg = glob.glob(os.path.join(folder_path,"*.jpg"))

    image_vg = image_loader(folder_path+"A Pair of Leather Clogs.jpg", device, imsize)
    
    image_original = content_image
    
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    
    model_vg =  models.vgg19(pretrained=True).features.to(device).eval()
    model_vg, style_losses_vg, content_losses_vg = get_style_model_and_losses(model_vg,
        cnn_normalization_mean, cnn_normalization_std, image_vg, image_vg)
    model_orig =  models.vgg19(pretrained=True).features.to(device).eval()
    model_orig, style_losses_orig, content_losses_orig = get_style_model_and_losses(model_orig,
        cnn_normalization_mean, cnn_normalization_std, image_original, image_original)
    model_output =  models.vgg19(pretrained=True).features.to(device).eval()
    model_output, style_losses_output, content_losses_output = get_style_model_and_losses(model_output,
        cnn_normalization_mean, cnn_normalization_std, output_image, output_image)
        
    losses_vg, losses_orig, losses_output = [], [], []
    losses_vg_content, losses_orig_content, losses_output_content = [], [], []
    for i, content_path_vg in enumerate(imgs_vg):
        if i == 50:
            break
        content_img_vg = image_loader(content_path_vg, device, imsize)

        #Send Van Gogh image to all 3 models
        model_output(content_img_vg)
        model_vg(content_img_vg)
        model_orig(content_img_vg)


        style_loss, content_loss = get_loss(style_losses_vg, content_losses_vg, 1, 1)

        print("van gogh vs van gogh", style_loss.item(), content_loss.item())
        losses_vg.append(style_loss.item())
        losses_vg_content.append(content_loss.item())

      
        style_loss, content_loss = get_loss(style_losses_orig, content_losses_orig, 1, 1)
        print("orig vs van gogh", style_loss.item(), content_loss.item())
        losses_orig.append(style_loss.item())
        losses_orig_content.append(content_loss.item())

      
        style_loss, content_loss = get_loss(style_losses_output, content_losses_output, 1, 1)
        print("output_image vs van gogh", style_loss.item(), content_loss.item())
        losses_output.append(style_loss.item())
        losses_output_content.append(content_loss.item())

    plt.figure()
    plt.title("Style loss")
    plt.plot(losses_vg, label = "van gogh vs. van gogh")
    plt.plot(losses_orig, label="orginal vs. van gogh")
    plt.plot(losses_output, label="output vs. van gogh")
    plt.legend()

    plt.savefig("../data/Style_loss.png")
    plt.figure()
    plt.title("Content loss")

    plt.plot(losses_vg_content, label = "van gogh vs. van gogh")
    plt.plot(losses_orig_content, label="original vs. van gogh")
    plt.plot(losses_output_content, label="output vs. van gogh")

    plt.legend()

    plt.savefig("../data/Content_loss.png")

    plt.show()




if __name__=='__main__':
    imsize = (224, 224)# if torch.cuda.is_available() else 128  # use small size if no gpu
    device= 'cpu'
  
    model = get_vgg19(device)


    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


    style_train, style_test = get_train_test("../data/VincentVanGogh/Arles/", imsize, device, max_imgs=5)
    reference_image = image_loader("../data/VincentVanGogh/Arles/Wheat Stacks with Reaper.jpg", device, imsize)
    content_img =  image_loader("../data/louvre.png", device, imsize)
    generated_image = torch.clone(content_img).detach()
    output = run_style_transfer(model, content_img, style_train, generated_image,
             content_layers_default, style_layers_default, num_steps=100)
    imshow(output)

    evaluate(content_img, output, reference_image, style_test, device)


    van_gogh_analysis(output, content_img, device, imsize)

    