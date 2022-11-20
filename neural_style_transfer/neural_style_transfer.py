"""
Based on https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
"""

import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import torch


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import glob

def get_vgg19(device):

    model =  models.vgg19(pretrained=True).features.to(device).eval()

    for param in model.parameters():
        param.requires_grad = False
    return model






content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']



def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost as defined by the formula above.
    """
    ### START CODE HERE
    
    #(≈1 line)
    J = alpha*J_content + beta*J_style
    
    ### START CODE HERE

    return J



# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


imsize = (224, 224)# if torch.cuda.is_available() else 128  # use small size if no gpu
unloader = transforms.ToPILImage()  # reconvert into PIL image


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    #plt.pause(0.001)
    plt.show()

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name, device):
    image = Image.open(image_name)

    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)






class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()/torch.norm(target)

    def forward(self, input):
        
        self.loss = F.mse_loss(input/torch.norm(input), self.target)
        return input



def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix

    G = G.div(torch.norm(G))
    return G#G.div(a * b * c * d)




class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

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

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses





def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img], lr=0.5)
    return optimizer





def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

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


# plt.figure()
# imshow(output, title='Output Image')

# # sphinx_gallery_thumbnail_number = 4
# #plt.ioff()
# plt.show()

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

def van_gogh_analysis(output_image, content_image, device):
    #https://www.kaggle.com/code/amyjang/monet-cyclegan-tutorial/data
    #https://www.kaggle.com/datasets/ipythonx/van-gogh-paintings
    
 
    folder_path = "../data/VincentVanGogh/Arles/"

    imgs_vg = glob.glob(os.path.join(folder_path,"*.jpg"))

    image_vg = image_loader(folder_path+"A Pair of Leather Clogs.jpg", device)
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
        content_img_vg = image_loader(content_path_vg, device)

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
    device= 'cpu'
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    model = get_vgg19(device)

    style_img = image_loader("../data/VincentVanGogh/Arles/Wheat Stacks with Reaper.jpg", device)

    content_img =  image_loader("../data/louvre.png", device)

    generated_image = torch.clone(content_img).detach()
    output = run_style_transfer(model, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, generated_image, num_steps=100)
    imshow(output)

    van_gogh_analysis(output, content_img, device)

    