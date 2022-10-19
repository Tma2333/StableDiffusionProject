import inspect
import typing as T

import torch
from diffusers import DiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import   PNDMScheduler, DDIMScheduler, LMSDiscreteScheduler
from PIL import Image

class CustomBasePipeline (DiffusionPipeline):
    def __init__(self, vae: AutoencoderKL,
                       text_encoder: CLIPTextModel,
                       tokenizer: CLIPTokenizer,
                       unet: UNet2DConditionModel,
                       scheduler: T.Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],) -> None:
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
    
    
    def enable_attention_slicing(self, slice_size: T.Optional[T.Union[str, int]] = "auto"):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                a number is provided, uses as many slices as `attention_head_dim // slice_size`. In this case,
                `attention_head_dim` must be a multiple of `slice_size`.
        """
        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)


    def disable_attention_slicing(self):
        r"""
        Disable sliced attention computation. If `enable_attention_slicing` was previously invoked, this method will go
        back to computing attention in one step.
        """
        # set slice_size = `None` to disable `set_attention_slice`
        self.enable_attention_slicing(None)


    @torch.no_grad()
    def __call__(self):
        raise NotImplementedError("Forward call has not implemented")


    @torch.no_grad()
    def text_to_embeding(self, prompt: T.Union[str, T.List[str]],
                               num_images_per_prompt: T.Optional[int] = 1,
                               max_text_length: int = None) -> T.Tuple[torch.Tensor]:
        # Input check
        if isinstance(prompt, str):
            self.batch_size = 1
        elif isinstance(prompt, list):
            self.batch_size = len(prompt)

        if max_text_length is None:
            max_length = max_text_length
            truncation = True
        else:
            max_length = self.tokenizer.model_max_length
            truncation = False
        
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation = truncation,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        if max_text_length is not None and text_input_ids.shape[-1] > max_length :
            removed_text = self.tokenizer.batch_decode(text_input_ids[:, max_length :])
            print(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, : max_length]
    
        text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]
        text_embeddings = text_embeddings.repeat_interleave(num_images_per_prompt, dim=0)
        
        return text_embeddings, text_inputs


    def check_negative_prompt (self, prompt: T.Union[str, T.List[str]],
                                     negative_prompt: T.Union[str, T.List[str]],
                                ) -> T.Optional[T.Union[str, T.List[str]]]:
        if negative_prompt is None:
            negative_prompt = [""] * self.batch_size
        elif type(prompt) is not type(negative_prompt):
            raise TypeError(
                "`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                " {type(prompt)}."
            )
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        elif self.batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {self.batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            negative_prompt = negative_prompt
        
        return negative_prompt


    @torch.no_grad()
    def initialize_lanten_input(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError(f'You need to implment how latent variable is implemented')

    
    @torch.no_grad()
    def diffusion_step(self, latents: torch.Tensor,
                             text_embeddings: torch.Tensor,
                             t: torch.Tensor,
                             do_classifier_free_guidance: bool, 
                             guidance_scale: float = 7.5,
                             eta: float = 0.0,) -> torch.Tensor:
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        return latents


    @torch.no_grad()
    def vae_decode(self, latents: torch.Tensor,) -> torch.Tensor:
        latents = 1 / 0.18215 * latents
        return self.vae.decode(latents).sample


    def RGB_tensor_to_numpy (self, RGB_tensor: torch.Tensor, return_pil: bool = False) -> T.List:
        RGB_tensor = (RGB_tensor / 2 + 0.5).clamp(0, 1)
        numpy_images = RGB_tensor.cpu().permute(0, 2, 3, 1).numpy()

        if numpy_images.ndim == 3:
            numpy_images = numpy_images[None, ...]
        if return_pil:
            numpy_images = (numpy_images * 255).round().astype("uint8")
            out_images = [Image.fromarray(image) for image in numpy_images]
            return out_images
        else:
            return numpy_images