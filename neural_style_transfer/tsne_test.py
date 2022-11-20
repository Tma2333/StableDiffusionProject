## T-SNE for feature visualization

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import torch
import glob
import os
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.manifold._t_sne import TSNE
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap


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


def get_vgg19(device):

    model =  models.vgg19(pretrained=True).features.to(device).eval()

    for param in model.parameters():
        param.requires_grad = False
    return model





def get_features(model, input, device, output_name='conv_4'):

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


    normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)


    x = normalization(input)
    i=0
    for layer in model.children():
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

        x = layer(x)

        if name == output_name:
            # add content loss:
            for i in range(x.shape[0]):
                x[i] = x[i].div(x[i].norm())
            x_clone = x.clone()
            activations = x.detach().numpy().reshape(x.shape[0], -1)
            a, b, c, d = x_clone.size() 

            features = x_clone.view(a, b, c * d) 
            G = torch.matmul(features, torch.permute(features, (0, 2, 1)))
            G = G.detach().numpy().reshape(G.shape[0], -1)
            return activations,  G
            

  


def batch_loader(filepaths, imsize):

    imgs = []
    for fp in filepaths:
        img = image_loader(fp, imsize)
        imgs.append(img)
    return torch.cat(imgs,dim=0)

def image_loader(image_name, imsize):
    loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

    image = Image.open(image_name)

    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

if __name__=='__main__':

    device = 'cpu'
    model = get_vgg19(device)
    imsize = (224, 224)# if torch.cuda.is_available() else 128  # use small size if no gpu
    unloader = transforms.ToPILImage() 
    img= image_loader("../data/picasso.jpeg", imsize)

    n = 20
    imgs_monet = batch_loader(glob.glob(os.path.join("../data/monet/","*.jpg"))[:n], imsize)

    imgs_vg = batch_loader(glob.glob(os.path.join("../data/VincentVanGogh/Arles/","*.jpg"))[:n], imsize)
    
    outputs_vg, outputs_G_vg = get_features(model, imgs_vg, device, 'conv_5')
    outputs_monet, outputs_G_monet = get_features(model, imgs_monet, device, 'conv_5')

    outputs = np.concatenate((outputs_G_vg, outputs_G_monet), 0)

    two_d = True


    labels = np.concatenate(
        (np.zeros((n,)),np.ones((n,)))
    )
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(outputs)


    df = pd.DataFrame()
    df["y"] = labels
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 2),
                data=df).set(title="T-SNE projection")
    plt.show()

    tsne = TSNE(n_components=3, verbose=1, random_state=123)
    z = tsne.fit_transform(outputs)

    x = z[:,0]
    y = z[:,1]
    z_axis = z[:,2]

        # axes instance
    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    
    ax.scatter(x[:n], y[:n], z_axis[:n], c = 'r', s=40, marker='o', alpha=1)
    ax.scatter(x[n:], y[n:], z_axis[n:], c = 'b', s=40, marker='o', alpha=1)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # legend
    #plt.legend(labels, bbox_to_anchor=(1.05, 1), loc=2)
    plt.show()