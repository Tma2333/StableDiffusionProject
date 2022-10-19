import typing as T

import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import   PNDMScheduler, DDIMScheduler, LMSDiscreteScheduler

from .custom_base_pipeline import CustomBasePipeline


class Text2ImagePipeline(CustomBasePipeline):
    def __init__(self, vae: AutoencoderKL, 
                       text_encoder: CLIPTextModel, 
                       tokenizer: CLIPTokenizer, 
                       unet: UNet2DConditionModel, 
                       scheduler: T.Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],) -> None:
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler)


    @torch.no_grad()
    def __call__(self, prompt: T.Union[str, T.List[str]],
                       height: int = 512,
                       width: int = 512,
                       num_inference_steps: int = 50,
                       guidance_scale: float = 7.5,
                       negative_prompt: T.Optional[T.Union[str, T.List[str]]] = None,
                       num_images_per_prompt: T.Optional[int] = 1,
                       eta: float = 0.0,
                       generator: T.Optional[torch.Generator] = None,
                       latents: T.Optional[torch.FloatTensor] = None,
                       output_type: str = 'numpy'):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
        if output_type not in ['numpy', 'pil', 'full_steps']:
            raise ValueError(f'output type: {output_type} is not supported')
        
        do_classifier_free_guidance = guidance_scale > 1.0
        # convert text to embeding
        text_embeddings, text_input = self.text_to_embeding(prompt, num_images_per_prompt)
        text_input_ids = text_input.input_ids
        max_length = text_input_ids.shape[-1]

        # configure negative prompt
        if do_classifier_free_guidance:
            negative_prompt = self.check_negative_prompt(prompt, negative_prompt)
            uncond_embeddings, _ = self.text_to_embeding(negative_prompt, num_images_per_prompt, max_length)

            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        # Some schedulers like PNDM have timesteps as arrays
        # It's more optimized to move all timesteps to correct device beforehand
        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        #init_latent
        latents = self.initialize_lanten_input(text_embeddings.dtype, 
                                               latents=latents,
                                               height=height, 
                                               width=width,
                                               num_images_per_prompt=num_images_per_prompt, 
                                               generator=generator)
        
        latents_steps = []
        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            latents = self.diffusion_step(latents=latents,
                                          text_embeddings=text_embeddings,
                                          t = t,
                                          do_classifier_free_guidance=do_classifier_free_guidance,
                                          guidance_scale = guidance_scale, 
                                          eta = eta)
            latents_steps.append(latents)
        
        # outputs:
        last_image_tensor = self.vae_decode(latents_steps[-1])
        if output_type == 'pil':
            return self.RGB_tensor_to_numpy(last_image_tensor, True)
        elif output_type == 'numpy':
            return self.RGB_tensor_to_numpy(last_image_tensor, False)
        elif output_type == 'full_steps':
            image_list = []
            for i, _ in enumerate(self.progress_bar(timesteps_tensor)):
                image_tensor = last_image_tensor = self.vae_decode(latents_steps[i])
                image_list.append(self.RGB_tensor_to_numpy(image_tensor, False))
            return np.concatenate(image_list, axis=0)


    @torch.no_grad()
    def initialize_lanten_input(self, latents_dtype: T.Any,
                                      latents: torch.FloatTensor = None,
                                      height: int = 512,
                                      width: int = 512,
                                      num_images_per_prompt: int = 1, 
                                      generator: T.Optional[torch.Generator] = None,) -> torch.Tensor:
        latents_shape = (self.batch_size * num_images_per_prompt, self.unet.in_channels, height // 8, width // 8)
        if latents is None:
                latents = torch.randn(latents_shape, generator=generator, device=self.device, dtype=latents_dtype)
        else:
            if latents.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
            latents = latents.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        return latents