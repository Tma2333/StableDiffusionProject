import typing as T

import torch
from torch import autocast
from tqdm.auto import tqdm
import matplotlib.pyplot as plt 


def run_stable_diffusion(prompt: T.List[str], 
                         vae, 
                         tokenizer, 
                         text_encoder, 
                         unet, 
                         scheduler, 
                         height: int = 512,
                         width: int= 512,                  
                         num_inference_steps: int = 100,
                         guidance_scale: float = 7.5,
                         seed: int = 0):
    """Outdated: leaving here as refernece
    """
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.manual_seed(seed) 
    batch_size = 1

    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

    # 2 configure guidance for conditioning the Unet
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )

    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # 3 generate noise
    latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
    )
    latents = latents.to(torch_device)

    # 4 initialize scheduler 
    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.init_noise_sigma
    
    for t in tqdm(scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    latents = 1 / 0.18215 * latents

    with torch.no_grad():
        image = vae.decode(latents).sample



    # normalize back to 0-1
    image_norm = (image / 2 + 0.5).clamp(0, 1)
    np_image = image_norm.detach().cpu().permute(0, 2, 3, 1).numpy()
    np_image = (np_image * 255).round().astype("uint8")

    fig = plt.figure(figsize=(height/100, width/100))
    ax = fig.add_subplot()
    ax.imshow(np_image[0])
    ax.axis('off')
    ax.set_title(f'prompt: {prompt[0]}')
