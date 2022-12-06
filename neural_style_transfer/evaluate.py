"""
Run forward on test style images, get style loss for 
    - input image
    - reference image
    - original image


"""
from model import *
from tsne import plot_tsne, get_features, get_featuresResnet
from datafetching import batch_loader
import glob
import os
import numpy as np

def evaluateOnTestSet(original_img, new_img, reference_image, style_test_set, device , content_weight, style_weight):
    model = get_vgg19(device)
    model, style_losses, content_losses = get_style_model_and_losses(model,
         style_test_set, original_img, device, content_weight, style_weight)

    model(original_img)
    style_loss, content_loss = get_loss(style_losses, content_losses)
    
    print("original img", style_loss.item(), content_loss.item())

    model(new_img)
    style_loss, content_loss = get_loss(style_losses, content_losses)
    print("new img", style_loss.item(), content_loss.item())

    model(reference_image)
    style_loss, content_loss = get_loss(style_losses, content_losses)
    print("reference img", style_loss.item(), content_loss.item())




def evaluate_on_tsne(original_imgs, output_imgs, device, imsize, title="tsne"):
    original = batch_loader(original_imgs, device, imsize)
    output = batch_loader(output_imgs, device, imsize)

    n = len(original)
    #vg = batch_loader(glob.glob(os.path.join("../data/Arles/","*.jpg"))[:n], device, imsize)

    
    labels = ["Original"]*n + ["After NST"]*n
    activations, tsne_inputs = get_featuresResnet(
        torch.cat((
            original, output
        ), 0), device
    )
    activations_train, tsne_inputs_train = get_features(
        torch.cat((
            original, output
        ), 0), device
    )
    print(title, tsne_inputs.shape[1])
    print("Variance before T-SNE (VGG19): Original", np.std(tsne_inputs_train[:n]))
    print("Variance before T-SNE (VGG19): Learned", np.std(tsne_inputs_train[n:]))
    
    print("Variance before T-SNE (Resnet): Original", np.std(tsne_inputs[:n]))
    print("Variance before T-SNE (Resnet): Learned", np.std(tsne_inputs[n:]))

    print("Activations")
    print("Variance before T-SNE (VGG19): Original", np.std(activations_train[:n]))
    print("Variance before T-SNE (VGG19): Learned", np.std(activations_train[n:]))
    
    print("Variance before T-SNE (Resnet): Original", np.std(activations[:n]))
    print("Variance before T-SNE (Resnet): Learned", np.std(activations[n:]))
    plot_tsne(tsne_inputs, labels,2 , 2, title+"_2dim", perplexity=n)
    #plot_tsne(tsne_inputs, labels,2 , 3, title+"_3dim")

if __name__=='__main__':
    device = 'cpu'
    imsize = (224, 224)

    orig = glob.glob(os.path.join("../data/samples2/","*o.png"))
    output = glob.glob(os.path.join("../data/image_copier_output/","*.png"))

    for folder in ["icarus_seed10", "narcissus_seed69", "match_sticks_seed4952", "match_sticks_seed5", "match_sticks_seed10"]:
        orig = glob.glob(os.path.join(f"../data/{folder}/","*guidance.png"))
        output = glob.glob(os.path.join(f"../data/copier_{folder}/","*.png"))
        evaluate_on_tsne(orig, output, device, imsize, folder)

