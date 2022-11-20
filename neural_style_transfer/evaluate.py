"""
Run forward on test style images, get style loss for 
    - input image
    - reference image
    - original image


"""
from model import *


def evaluate(original_img, new_img, reference_image, style_test_set, device ):
    model = get_vgg19(device)
    model, style_losses, content_losses = get_style_model_and_losses(model,
         style_test_set, original_img, device)

    model(original_img)
    style_loss, content_loss = get_loss(style_losses, content_losses, 1, 1)
    print("original img", style_loss.item(), content_loss.item())

    model(new_img)
    style_loss, content_loss = get_loss(style_losses, content_losses, 1, 1)
    print("new img", style_loss.item(), content_loss.item())

    model(reference_image)
    style_loss, content_loss = get_loss(style_losses, content_losses, 1, 1)
    print("reference img", style_loss.item(), content_loss.item())




