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




    _, orig_res = get_featuresResnet(
        original, device
    )
    _, out_res = get_featuresResnet(
        output, device
    )
    _, orig_vgg = get_features(
        original, device
    )
    _, out_vgg = get_features(
        output, device
    )
    activations_train, tsne_inputs_train = get_features(
        torch.cat((
            original, output
        ), 0), device
    )
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

    print(out_vgg.shape)
    print("Variance before T-SNE (VGG19): Original", np.sum(np.std(orig_vgg, axis=0)))
    print("Variance before T-SNE (VGG19): Learned", np.sum(np.std(out_vgg, axis=0)))
    
    print("Variance before T-SNE (Resnet): Original", np.sum(np.std(orig_res, axis=0)))
    print("Variance before T-SNE (Resnet): Learned", np.sum(np.std(out_res, axis=0)))
    print("Perc. diff", (np.sum(np.std(orig_res, axis=0))-np.sum(np.std(out_res, axis=0)))/np.sum(np.std(orig_res, axis=0))*100.) 
    print("Perc. diff train", (np.sum(np.std(orig_vgg, axis=0))-np.sum(np.std(out_vgg, axis=0)))/np.sum(np.std(orig_vgg, axis=0))*100.) 
    # print("Activations")
    # print("Variance before T-SNE (VGG19): Original", np.std(activations_train[:n]))
    # print("Variance before T-SNE (VGG19): Learned", np.std(activations_train[n:]))
    
    # print("Variance before T-SNE (Resnet): Original", np.std(activations[:n]))
    # print("Variance before T-SNE (Resnet): Learned", np.std(activations[n:]))
    plot_tsne(tsne_inputs, labels,2 , 2, title+"_2dim", perplexity=n)
    #plot_tsne(tsne_inputs, labels,2 , 3, title+"_3dim")

if __name__=='__main__':
    device = 'cpu'
    imsize = (224, 224)

    
    for folder in ["icarus_seed10", "narcissus_seed69", "match_sticks_seed4952"]:
        orig = glob.glob(os.path.join(f"../data/{folder}/","*guidance.png"))
        output = glob.glob(os.path.join(f"../data/blender_{folder}/","*.png"))
        print(orig, output)
        evaluate_on_tsne(orig, output, device, imsize, folder)

