import torch
from hooks.collect_activations import CollectActivationsLinearNoMean
from hooks.block_activations import RescaleLinearActivations
import copy
from torchmetrics.functional import total_variation, multiscale_structural_similarity_index_measure


@torch.no_grad()
def prepare_diffusion_inputs(prompts, tokenizer, text_encoder, unet, guidance_scale, samples_per_prompt, seed):
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

    latents = torch.randn(
        (len(prompts), unet.config.in_channels, height // 8, width // 8),
        generator=generator,
    )
    
    if guidance_scale != 0:
        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer([""] * len(prompts),
                                    padding="max_length",
                                    max_length=max_length,
                                    return_tensors="pt")
        uncond_embeddings = text_encoder(
            uncond_input.input_ids.to(text_encoder.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latents = latents.to(text_embeddings.device)
    return latents, text_embeddings


# run the denoising process to collect the activations with a hook (has to be added beforehand)
@torch.no_grad()
def collect_activations(prompts, tokenizer, text_encoder, unet, scheduler, num_inference_steps=50, early_stopping=None, seed=1, samples_per_prompt=1):
    latents, text_embeddings = prepare_diffusion_inputs(prompts, tokenizer, text_encoder, unet, guidance_scale=0, samples_per_prompt=samples_per_prompt, seed=seed)
    scheduler.set_timesteps(num_inference_steps)

    # inject hooks into value layers
    v_handles = []
    v_hooks = []
    for down_block in range(3):
        for attention in range(2):
            v_hook = CollectActivationsLinearNoMean()
            v_handle = unet.down_blocks[down_block].attentions[attention].transformer_blocks[0].attn2.to_v.register_forward_hook(v_hook)
            v_handles.append(v_handle)
            v_hooks.append(v_hook)
    v_hook = CollectActivationsLinearNoMean()
    v_handle = unet.mid_block.attentions[0].transformer_blocks[0].attn2.to_v.register_forward_hook(v_hook)
    v_handles.append(v_handle)
    v_hooks.append(v_hook)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        for i, t in enumerate(scheduler.timesteps):
            latent_model_input = latents                
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            noise_pred = unet(
                latent_model_input.cuda(),
                t,
                encoder_hidden_states=text_embeddings, return_dict=False)[0]
                                        
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            torch.cuda.empty_cache()
                        
            if early_stopping is not None and i < num_inference_steps - 1:
                break
    
    activations = []
    for hook, handle in zip(v_hooks, v_handles):
        activations.append(hook.activations()[0].abs().mean(dim=0))
        handle.remove()
        
    return activations


@torch.no_grad()
def initial_neuron_selection(prompt, tokenizer, text_encoder, unet, scheduler, layer_depth, theta, k, seed=1, version='v1-5'):
    # load statistics from unmemorized LAION prompts
    if version == 'v1-4':
        mean_list, std_list = torch.load('statistics/statistics_additional_laion_prompts_v1_4.pt')
    else:
        mean_list, std_list = torch.load(version)

    # variables to count number of (deactivated) neurons
    deactivated_neurons = 0
    total_neurons = 0

    # compute and collect activations based on OOD detection and top-k absolute activations
    activations_list = collect_activations([prompt], tokenizer, text_encoder, unet, scheduler, num_inference_steps=50, samples_per_prompt=1, early_stopping=1, seed=seed)

    blocking_indices = [[] for i in range(7)]
    for layer_id in range(layer_depth):
        activations = activations_list[layer_id]
        diff = (activations.cpu() - mean_list[layer_id]).abs() / std_list[layer_id]
        indices = (diff > theta).nonzero().flatten().tolist()
        
        topk_indices = activations.abs().topk(k=min(k, len(mean_list[layer_id]))).indices
        indices += [e.item() for e in topk_indices]
        
        total_neurons += activations.shape[0]
        deactivated_neurons += len(indices)
        blocking_indices[layer_id] = indices
            
    return blocking_indices


def calculate_max_pairwise_ssim(noise_diffs):    
    pairwise_combination_indices = torch.combinations(torch.arange(len(noise_diffs)), r=2)

    input_1 = noise_diffs[pairwise_combination_indices[:,0]]
    input_2 = noise_diffs[pairwise_combination_indices[:,1]]
    ssim = multiscale_structural_similarity_index_measure(input_1, input_2, reduction='none', kernel_size=11, betas=(0.33, 0.33, 0.33))

    max_ssims = []
    for index in range(len(noise_diffs)):
        max_ssims.append(ssim[(pairwise_combination_indices == index).max(-1).values].max())
    max_ssims = torch.stack(max_ssims)

    return max_ssims


# denoising process to collect noise diffs between the predicted noise and the noise latents from the previous step.
@torch.no_grad()
def compute_noise_diff(prompts, tokenizer, text_encoder, unet, scheduler, blocked_indices, guidance_scale, seed, samples_per_prompt, scaling_factor, num_inference_steps=50, early_stopping=1, seed_indices_to_return=None):
    latents, text_embeddings = prepare_diffusion_inputs(prompts, tokenizer, text_encoder, unet, guidance_scale=guidance_scale, samples_per_prompt=samples_per_prompt, seed=seed)
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


    with torch.autocast(device_type="cuda", dtype=torch.float16):
        for i, t in enumerate(scheduler.timesteps):
            if guidance_scale == 0:
                latent_model_input = latents
            else:
                latent_model_input = torch.cat([latents] * 2)

            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            text_embeddings.requires_grad = False
            latent_model_input.requires_grad = False

            noise_pred = unet(
                latent_model_input.cuda(),
                t,
                encoder_hidden_states=text_embeddings, return_dict=False)[0]
            
            if guidance_scale != 0:
                    noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond)
                    
            if i >= early_stopping or i == num_inference_steps - 1:
                if blocked_indices:
                    for handle in block_handles:
                        handle.remove()
                                                
                noise_diff = noise_pred - latents
                min_values = noise_diff.amin(dim=[2,3])
                max_values = noise_diff.amax(dim=[2,3])
                noise_diff_scaled = (noise_diff - min_values.unsqueeze(-1).unsqueeze(-1)) / (max_values - min_values).unsqueeze(-1).unsqueeze(-1)

                if seed_indices_to_return is not None:
                    return noise_diff_scaled[seed_indices_to_return]
                
                return noise_diff_scaled



@torch.no_grad()
def neuron_refinement(prompt, tokenizer, text_encoder, unet, scheduler, input_indices, scaling_factor, metric='ssim', threshold=None, samples_per_prompt=8, guidance_scale=0, seed=1, seeds_to_look_at=None, rel_threshold=None):
    noise_diff_vanilla = compute_noise_diff([prompt], tokenizer, text_encoder, unet, scheduler, seed=seed, blocked_indices=None, scaling_factor=1, samples_per_prompt=samples_per_prompt, guidance_scale=guidance_scale, num_inference_steps=50, seed_indices_to_return=seeds_to_look_at)
    blocking_indices = copy.deepcopy(input_indices)
    active_layers = set(i for i in range(len(blocking_indices)))

    if rel_threshold is not None:
        print('Using relative threshold')
        threshold = rel_threshold

    # 1.) remove all layers with no blocked neurons or neurons without any impact
    total_neurons = 0
    neurons_removed = 0
    noise_diff_all_blocked = compute_noise_diff([prompt], tokenizer, text_encoder, unet, scheduler, blocked_indices=blocking_indices, scaling_factor=scaling_factor, seed=seed, samples_per_prompt=samples_per_prompt, guidance_scale=guidance_scale, num_inference_steps=50, seed_indices_to_return=seeds_to_look_at)
    diff_all_blocked = multiscale_structural_similarity_index_measure(noise_diff_vanilla, noise_diff_all_blocked, reduction='none', kernel_size=11, betas=(0.33, 0.33, 0.33))
    for layer_idx, layer_blocked_indices in reversed(list(enumerate(blocking_indices))):
        # unblock all neurons from a specific layer
        if len(layer_blocked_indices) == 0:
            active_layers.remove(layer_idx)
        else:
            # get all list elements except the current layer
            total_neurons += len(layer_blocked_indices)
            curr_indices = copy.deepcopy(blocking_indices)
            curr_indices[layer_idx] = []
            noise_diff_blocked = compute_noise_diff([prompt], tokenizer, text_encoder, unet, scheduler, blocked_indices=curr_indices, scaling_factor=scaling_factor, seed=seed, samples_per_prompt=samples_per_prompt, guidance_scale=guidance_scale, num_inference_steps=50, seed_indices_to_return=seeds_to_look_at)
            if metric == 'ssim':
                diff = multiscale_structural_similarity_index_measure(noise_diff_vanilla, noise_diff_blocked, reduction='none', kernel_size=11, betas=(0.33, 0.33, 0.33))
            
            comparison_value = diff.max()
            if rel_threshold is not None:
                comparison_value = ((diff - diff_all_blocked) / (diff_all_blocked.abs() + 1e-9)).abs().max()
            
            if comparison_value < threshold:
                neurons_removed += len(layer_blocked_indices)
                blocking_indices[layer_idx] = []
                active_layers.remove(layer_idx)
                
    print('Removed the following layers:', set(range(7)) - active_layers, f'with {neurons_removed} neurons.')
    print('Remaining layers:', active_layers, f'with {total_neurons - neurons_removed} neurons.')

    # 2.) check individual neurons in remaining layers
    neurons_removed = 0
    for layer_idx, blocked_indices in reversed(list(enumerate(blocking_indices))):
        if len(blocked_indices) == 0:
            continue
        blocking_indices_copy = copy.deepcopy(blocking_indices)
        for neuron in blocking_indices_copy[layer_idx]:
            curr_indices = copy.deepcopy(blocking_indices)
            curr_indices[layer_idx].remove(neuron)
            noise_diff_blocked = compute_noise_diff([prompt], tokenizer, text_encoder, unet, scheduler, blocked_indices=curr_indices, scaling_factor=scaling_factor, seed=seed, samples_per_prompt=samples_per_prompt, guidance_scale=guidance_scale, num_inference_steps=50, seed_indices_to_return=seeds_to_look_at)
            if metric=='ssim':
                diff = multiscale_structural_similarity_index_measure(noise_diff_vanilla, noise_diff_blocked, reduction='none', kernel_size=11, betas=(0.33, 0.33, 0.33))
            elif metric == 'tv':
                diff = total_variation(noise_diff_blocked, reduction='none')

            comparison_value = diff.max()
            if rel_threshold is not None:
                comparison_value = ((diff - diff_all_blocked) / (diff_all_blocked.abs() + 1e-9)).abs().max()

            if comparison_value < threshold:
                neurons_removed += 1
                blocking_indices[layer_idx].remove(neuron)

    print(f'Removed {neurons_removed} neurons.')
    print(f'Remaining neurons: {blocking_indices}')
    return blocking_indices