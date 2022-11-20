"""
Run forward on test style images, get style loss for 
    - input image
    - reference image
    - original image


"""
from model import *


def evaluate(original_img, new_img, reference_image, style_test_set, device , content_weight, style_weight):
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




