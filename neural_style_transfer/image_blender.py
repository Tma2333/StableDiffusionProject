import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torch
import numpy as np

from datafetching import *
from model import get_input_optimizer, get_loss
from lossFunctions import ContentLoss, StyleLoss_blending
def get_vgg19(device):

    model =  models.vgg19(pretrained=True).features.to(device).eval()

    for param in model.parameters():
        param.requires_grad = False
    return model





# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).clone().detach().view(-1, 1, 1)
        self.std = torch.tensor(std).clone().detach().view(-1, 1,1)

    def forward(self, imgs):
        # normalize img
        return imgs
        imgs =(imgs-self.mean)/self.std
        return imgs





def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    unloader = transforms.ToPILImage()  # reconvert into PIL image

    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    #plt.pause(0.001)
    plt.savefig(title+".png")
    plt.show()








def get_style_model_and_losses(cnn, input_imgs, device, content_weight, style_weight):


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
            targets = model(original_input).detach()
            
            content_loss = ContentLoss(targets, content_weight)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            targets = model(original_input).detach()
            style_loss = StyleLoss_blending(targets, style_weight)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss_blending):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def train(model, style_losses, content_losses, input, num_steps, lr=0.1):
    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    model.requires_grad_(False)
    input.requires_grad_(True)

    optimizer = get_input_optimizer(input, lr=lr)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input.clamp_(0, 1)

            optimizer.zero_grad()
            model(input)
            style_score, content_score = get_loss(style_losses, content_losses)

            loss = style_score + content_score
            loss.backward()
            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        #torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input.clamp_(0, 1)

    return input


def run_style_transfer(cnn, input_imgs,  num_steps=300,
                       style_weight=1000000, content_weight=1, device='cpu'):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
         input_imgs, device, content_weight, style_weight)
    output = train(model, style_losses, content_losses,
        input_imgs, num_steps)
    return output
    


if __name__=='__main__':
    imsize = (224, 224)# if torch.cuda.is_available() else 128  # use small size if no gpu
    device= 'cuda'
  
    model = get_vgg19(device)
    content_weight=50
    style_weight=10000000
    folder = "icarus_seed132"
    paths = glob.glob(os.path.join(f"../data/{folder}/","*.png"))
    paths.sort(key = lambda x: int("".join([c for c in x if c.isnumeric()])))
    imgs_samples = torch.cat([
        image_loader(path, device, imsize) for path in paths
    ], 0)
        
    # img1 = image_loader("../data/Arles/A Pair of Leather Clogs.jpg", device, imsize)
    # img2 = image_loader("../data/samples/redhood_CB.png", device, imsize)
    # img3 = image_loader("../data/Arles/Still Life Vase with Oleanders and Books.jpg", device, imsize)
    # img4 = image_loader("../data/spooky.jpg", device, imsize)
    # input_imgs = torch.cat([img1, img3], 0)

    #style_train, style_test = get_train_test("../data/Arles/", imsize, device, max_imgs=7)

    
    output = run_style_transfer(model, imgs_samples, content_weight=content_weight, style_weight=style_weight, 
                    num_steps=100, device=device)

    for i, out in enumerate(output):
        fpath = f"../data/image_blender_output/Output_{i}.png"
        A = np.transpose(out.detach().cpu().squeeze(),(1,2,0)).numpy()
        im = Image.fromarray((A * 255).astype(np.uint8))
        im.save(fpath)    
        #imshow(out, fpath )

