from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import torch
from torch import autocast
from hooks.deactivation_context import DeactivateHooksContext
from PIL import Image
from hooks.block_activations import RescaleLinearActivations
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DiffusionPipeline


def load_sd_components(model_path):
    if model_path == 'v1-4':
        model_path = 'CompVis/stable-diffusion-v1-4'
    elif model_path == 'v1-5':
        model_path = 'runwayml/stable-diffusion-v1-5'
    vae = AutoencoderKL.from_pretrained(model_path,
                                        subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(
        model_path,
        subfolder="unet")
    
    scheduler = DDIMScheduler.from_pretrained(model_path, subfolder='scheduler')
    return vae, unet, scheduler


def load_text_components(model_path):
    if model_path == 'v1-4' or model_path =='v1-5':
        model_path = 'openai/clip-vit-large-patch14'
        tokenizer = CLIPTokenizer.from_pretrained(model_path)
        text_encoder = CLIPTextModel.from_pretrained(model_path)
    else:
        scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
        pipe = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler, torch_dtype=torch.float32)
        tokenizer = pipe.tokenizer
        text_encoder = pipe.text_encoder
    return tokenizer, text_encoder


@torch.no_grad
def generate_images(prompts, tokenizer, text_encoder, vae, unet, scheduler, blocked_indices=None, scaling_factor=0.0, num_inference_steps=50, early_stopping=None, seed=1, guidance_scale=7, samples_per_prompt=1, hooks=None, inactive_hook_steps=None):
    with torch.no_grad():
        height = 512
        width = 512
        generator = torch.manual_seed(seed)
        if samples_per_prompt > 1:
            prompts = [prompt for prompt in prompts for _ in range(samples_per_prompt)]
        
        text_input = tokenizer(prompts,
                                padding="max_length",
                                max_length=tokenizer.model_max_length,
                                truncation=True,
                                return_tensors="pt")
        text_embeddings = text_encoder(
            text_input.input_ids.to(text_encoder.device))[0]

        max_length = text_input.input_ids.shape[-1]
        if guidance_scale != 0:
            uncond_input = tokenizer([""] * len(prompts),
                                        padding="max_length",
                                        max_length=max_length,
                                        return_tensors="pt")
            uncond_embeddings = text_encoder(
                uncond_input.input_ids.to(text_encoder.device))[0]
            text_embeddings = torch.cat([text_embeddings, uncond_embeddings])

        latents = torch.randn(
            (len(prompts), unet.config.in_channels, height // 8, width // 8),
            generator=generator,
        )
        latents = latents.to(text_embeddings.device)
        scheduler.set_timesteps(num_inference_steps)
        
        if blocked_indices:
            block_handles = []
            block_hooks = []
            for down_block in range(3):
                for attention in range(2):
                    indices = blocked_indices[down_block * 2 + attention]
                    block_hook = RescaleLinearActivations(indices=indices, factor=scaling_factor)
                    block_handle = unet.down_blocks[down_block].attentions[attention].transformer_blocks[0].attn2.to_v.register_forward_hook(block_hook)
                    block_handles.append(block_handle)
                    block_hooks.append(block_hook)
            block_hook = RescaleLinearActivations(indices=blocked_indices[-1], factor=scaling_factor)
            block_handle = unet.mid_block.attentions[0].transformer_blocks[0].attn2.to_v.register_forward_hook(block_hook)
            block_handles.append(block_handle)
            block_hooks.append(block_hook)           
            print(f'Number of blocked value neurons: {sum([len(block_hook.indices) for block_hook in block_hooks])}')

        with autocast("cuda", dtype=torch.float16):                
            for i, t in enumerate(scheduler.timesteps):
                
                if early_stopping is not None and i+1 > early_stopping:
                    break
                if guidance_scale == 0:
                    latent_model_input = latents
                else:
                    latent_model_input = torch.cat([latents] * 2)
                    
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                with torch.no_grad():
                    if inactive_hook_steps and i in inactive_hook_steps:
                        if hooks is not None:
                            with DeactivateHooksContext(hooks):
                                noise_pred = unet(
                                    latent_model_input,
                                    t,
                                    encoder_hidden_states=text_embeddings, return_dict=False)[0]
                    else:
                        noise_pred = unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=text_embeddings, return_dict=False)[0]
                if guidance_scale != 0:
                    noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond)
                    
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
                torch.cuda.empty_cache()
                
            if blocked_indices:
                for handle in block_handles:
                    handle.remove()

            latents = 1 / 0.18215 * latents
            image = vae.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (image * 255).round().astype("uint8")
            pil_images = [Image.fromarray(image) for image in images]
            return pil_images
