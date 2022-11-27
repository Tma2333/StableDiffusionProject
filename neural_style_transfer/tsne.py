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
from datafetching import batch_loader
from model import get_vgg19, Normalization, get_resnet50
from torchsummary import summary



def get_featuresResnet( input, device, output_name='Sequential_3'):

    model = get_resnet50(device)
    # if device=='cpu':
    summary(model,input[0].shape, device=device)
    #raise Exception
    # cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    # cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


    # normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)


    #x = normalization(input)
    x = input
    i=0
    for layer in model.children():
        #print(layer.type)
        name = layer.type
        if isinstance(layer, nn.Sequential):
            print(layer[0])
            i += 1
            name = 'Sequential_{}'.format(i)
                
        x = layer(x)

        if name == output_name:
            for i in range(x.shape[0]):
                x[i] = x[i].div(x[i].norm())
            x_clone = x.clone()
            activations = x.detach().numpy().reshape(x.shape[0], -1)
            a, b, c, d = x_clone.size() 

            features = x_clone.view(a, b, c * d) 
            G = torch.matmul(features, torch.permute(features, (0, 2, 1)))
            G = G.detach().numpy().reshape(G.shape[0], -1)
            
            return activations,  G
            
            


def get_features( input, device, output_name='conv_4'):

    model = get_vgg19(device)

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
            

  





def plot_tsne(inputs, labels, n_classes=2, n_components=2, title="tsne.png"):

    tsne = TSNE(n_components=n_components, verbose=1, random_state=123, n_iter=10000, perplexity=5)
    z = tsne.fit_transform(inputs)

    x = z[:,0]
    y = z[:,1]
    fig = plt.figure(figsize=(6,6))

    if n_components == 3:
        z_axis = z[:,2]

        # axes instance

        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
        n = len(x)//len(labels)
        colors = ['r', 'b', 'g', 'y']
        for i in range(n_classes):
            ax.scatter(x[i*n:(i+1)*n], y[i*n:(i+1)*n], z_axis[i*n:(i+1)*n], c = colors[i], s=40, marker='o', alpha=1)
            #ax.scatter(x[n:], y[n:], z_axis[n:], c = 'b', s=40, marker='o', alpha=1)
        ax.set_xlabel('X ')
        ax.set_ylabel('Y ')
        ax.set_zlabel('Z ')

        plt.legend(labels)

    else:
        df = pd.DataFrame()
        df["y"] = labels
        df["comp-1"] = x
        df["comp-2"] = y

        sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                #palette=sns.color_palette("hls", 2),
                data=df).set(title="T-SNE projection")
    plt.savefig("../data/tsne/"+title)
    plt.show()



if __name__=='__main__':

    device = 'cpu'
    imsize = (224, 224)# if torch.cuda.is_available() else 128  # use small size if no gpu
    #unloader = transforms.ToPILImage() 
    #img= image_loader("../data/picasso.jpeg", imsize)

    n = 5
    imgs_samples = batch_loader(glob.glob(os.path.join("../data/samples/","*.png"))[:n], device, imsize)

    imgs_vg = batch_loader(glob.glob(os.path.join("../data/Arles/","*.jpg"))[:n],device,  imsize)
    
    outputs_vg, outputs_G_vg = get_featuresResnet(imgs_vg, device)
    outputs_samples, outputs_G_samples = get_featuresResnet(imgs_samples, device)

    outputs = np.concatenate((outputs_vg, outputs_samples), 0)



    labels = ["vg"]*n + ["samples"]*n
    plot_tsne(outputs, labels,2,  2, "tsne_2dim")
    plot_tsne(outputs, labels,2,  3, "tsne_3dim")

    