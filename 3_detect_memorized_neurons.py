import argparse
import os
from utils.stable_diffusion import load_sd_components, load_text_components, generate_images
import pandas as pd
from utils.activation_detection import compute_noise_diff, initial_neuron_selection, neuron_refinement, calculate_max_pairwise_ssim
from torchmetrics.functional import structural_similarity_index_measure, multiscale_structural_similarity_index_measure
from rtpt import RTPT
from tqdm import tqdm
import sys
import csv
import numpy as np

def compute_memorization_statistics(args):

    # load Stable Diffusion components
    vae, unet, scheduler = load_sd_components(args.version)
    tokenizer, text_encoder = load_text_components(args.version)

    torch_device = "cuda"
    vae.to(torch_device)
    text_encoder.to(torch_device)
    unet.to(torch_device)

    # load prompts
    prompts = pd.read_csv(args.dataset, sep=';')
    
    
    if args.continue_run:
        print('Continuing the previous run...')
        output_file = pd.read_csv(args.output, sep=';')
        prompts = prompts.loc[len(output_file):]
        # open output file in append mode
        file = open(args.output, 'a', newline ='')
        writer = csv.DictWriter(file, fieldnames = output_file.columns, delimiter=';')
    else:
        # create output csv file
        header = ['Caption', 'URL', 'Initial Neurons', 'Initial SSIM', 'Refined Neurons', 'Refined SSIM']
        file = open(args.output, 'w', newline ='')
        writer = csv.DictWriter(file, fieldnames = header, delimiter=';')
        writer.writeheader()
    
    # start RTPT
    rtpt = RTPT(args.name, 'Memorization Statistics', len(prompts))
    rtpt.start()
    
    for i, row in tqdm(prompts.iterrows(), total=len(prompts)):
        prompt = row['Caption']
        output_dict = {'Caption': prompt, 'URL': row['URL']}
        
        # find the initial selection of blocked neurons
        noise_diff_unblocked = compute_noise_diff([prompt], tokenizer, text_encoder, unet, scheduler, seed=args.seed, blocked_indices=None, scaling_factor=args.scaling_factor, samples_per_prompt=args.samples_per_prompt, guidance_scale=args.guidance_scale, num_inference_steps=args.num_inference_steps)

        max_ssims_per_noise_diff = calculate_max_pairwise_ssim(noise_diff_unblocked)
        sample_indices_to_look_at = max_ssims_per_noise_diff > args.pairwise_ssim_threshold
        noise_diff_unblocked = noise_diff_unblocked[sample_indices_to_look_at]

        # if there are SSIM values that are above the threshold skip this sample as it seems to be not memorized
        if sample_indices_to_look_at.sum() == 0:
            print(f'\nSkipping Prompt {i} as it does not seem to be memorized based on the pairwise SSIM threshold.')
            # write an empty line to the csv
            output_dict['Initial Neurons'] = [[]] * 7
            output_dict['Initial SSIM'] = 0
            output_dict['Refined Neurons'] = [[]] * 7
            output_dict['Refined SSIM'] = 0
            # write results to csv file
            writer.writerow(output_dict)
            file.flush()
            continue

        ssim = 1.0
        theta = args.initial_theta
        layer_depth = 7
        k=args.initial_k

        if args.no_top_k:
            k = 0

        if args.no_theta:
            theta = np.inf

        refinement_ssim_threshold = args.ssim_threshold_refinement
        while ssim > args.ssim_threshold_initial_selection:
            blocked_indices = initial_neuron_selection(prompt, tokenizer, text_encoder, unet, scheduler, layer_depth=layer_depth, theta=theta, k=k, seed=args.seed, version=args.version)
            noise_diff_blocked = compute_noise_diff([prompt], tokenizer, text_encoder, unet, scheduler, seed=args.seed, blocked_indices=blocked_indices, scaling_factor=args.scaling_factor, samples_per_prompt=args.samples_per_prompt, guidance_scale=args.guidance_scale, num_inference_steps=args.num_inference_steps, seed_indices_to_return=sample_indices_to_look_at)
            ssim = multiscale_structural_similarity_index_measure(noise_diff_unblocked, noise_diff_blocked, reduction='none', kernel_size=11, betas=(0.33, 0.33, 0.33)).max()
                
            if ssim > args.ssim_threshold_initial_selection:
                if theta > args.min_theta and not args.no_theta:
                    theta = theta - args.theta_reduction_per_step
                if not args.no_top_k:
                    k += 1
            else:
                print(f'Initial selection of blocked neurons found with theta={theta} and k={k}. SSIM={ssim}')

            if theta == 1 or k >= 1280:
                refinement_ssim_threshold = ssim
                break
                
        output_dict['Initial Neurons'] = blocked_indices
        output_dict['Initial SSIM'] = ssim.cpu().item()
        
        # refine the selection of blocked prompt
        refined_blocking_indices = neuron_refinement(prompt, tokenizer, text_encoder, unet, scheduler, input_indices=blocked_indices, scaling_factor=args.scaling_factor, threshold=refinement_ssim_threshold, rel_threshold=args.rel_threshold_refinement, samples_per_prompt=args.samples_per_prompt, guidance_scale=args.guidance_scale, seed=args.seed, seeds_to_look_at=sample_indices_to_look_at)
        output_dict['Refined Neurons'] = refined_blocking_indices
        
        # compute SSIM with refined neurons
        noise_diff_blocked = compute_noise_diff([prompt], tokenizer, text_encoder, unet, scheduler, seed=args.seed, blocked_indices=refined_blocking_indices, scaling_factor=args.scaling_factor, samples_per_prompt=args.samples_per_prompt, guidance_scale=args.guidance_scale, num_inference_steps=args.num_inference_steps, seed_indices_to_return=sample_indices_to_look_at)
        ssim = multiscale_structural_similarity_index_measure(noise_diff_unblocked, noise_diff_blocked, reduction='none', kernel_size=11, betas=(0.33, 0.33, 0.33)).max()
        output_dict['Refined SSIM'] = ssim.cpu().item()
        
        # write results to csv file
        writer.writerow(output_dict)
        file.flush()

        rtpt.step()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', default='v1-4', type=str, help='Stable diffusion version (default: v1-4)')
    parser.add_argument('-d', '--dataset', default='prompts/memorized_laion_prompts.csv', type=str, help='Dataset of memorized prompts (default: prompts/memorized_laion_prompts.csv)')
    parser.add_argument('-o', '--output', default='results/memorization_statistics.csv', type=str, help='Output file for memorization statistics (default: results/memorization_statistics.csv)')
    parser.add_argument('-s', '--seed', default=1, type=int, help='Random seed (default: 1)')
    parser.add_argument('--samples_per_prompt', default=10, type=int, help='Number of samples generated per prompt (default: 10)')
    parser.add_argument('--num_inference_steps', default=50, type=int, help='Number of inference steps used to generate the images. Even though only the first step is used, this has an effect on the noise prediction of the first step. (default: 50)')
    parser.add_argument('-sf', '--scaling_factor', default=0.0, type=float, help='Scaling factor for neuron activations (default: 0.0)')
    parser.add_argument('--pairwise_ssim_threshold', default=0.428, type=float, help='Threshold for the pairwise SSIM for choosing at which initial samples to look at (default: 0.428)')
    parser.add_argument('--ssim_threshold_initial_selection', default=0.428, type=float, help='SSIM threshold for the initial neuron selection (default: 0.428)')
    parser.add_argument('--ssim_threshold_refinement', default=0.428, type=float, help='SSIM threshold for the neuron refinement (default: 0.428)')
    parser.add_argument('--rel_threshold_refinement', default=None, type=float, help='Relative threshold for the neuron refinement (default: None)')
    parser.add_argument('--guidance_scale', default=0, type=float, help='The guidance scale for the image generation during and after the neuron selection (default: 0)')
    parser.add_argument('--theta_reduction_per_step', default=0.25, type=float, help='The reduction of theta value per step during the initial neuron selection(default: 0.25)')
    parser.add_argument('--min_theta', default=1, type=float, help='The minimum theta value for the initial neuron selection (default: 1)')
    parser.add_argument('--no_theta', action='store_true', help='Do not use the theta value for the initial selection')
    parser.add_argument('--initial_theta', default=5, type=float, help='The initial theta value for the initial neuron selection (default: 5)')
    parser.add_argument('--no_top_k', action='store_true', help='Do not use the top k neurons for the initial selection')
    parser.add_argument('--initial_k', default=0, type=int, help='The initial k value for the initial neuron selection (default: 0)')
    parser.add_argument('-n', '--name', default='XX',  type=str,help='RTPT user name (Default: XX)')
    parser.add_argument('--continue_run', action='store_true', help='Continue the previous run')
    
    args = parser.parse_args()
    
    if not os.path.exists('results'):
        os.makedirs('results')

    args.output = args.output.replace('.csv', f'_{args.version.replace("-", "_")}.csv')
    if args.continue_run and not os.path.exists(args.output):
        raise ValueError('The output file does not exist. Please provide an existing file to continue the run.')

    if not args.continue_run:
        if os.path.exists(args.output):
            args.output = args.output.replace('.csv', '_1.csv')

        with open(args.output.replace('.csv', '.txt'), 'w') as f:
            for key, value in args.__dict__.items():
                f.write(f'{key}: {value}\n')
            f.write(f'Command: {" ".join(sys.argv)}')
    else:
        # check that the parameters match
        with open(args.output.replace('.csv', '.txt'), "r") as f:
            lines = f.readlines()
            old_args = {}
            for line in lines:
                key, value = line.split(': ')
                old_args[key] = value.replace('\n', '')
            
            # compare the dictionary of the old run with the current args
            for key, value in args.__dict__.items():
                if key == 'continue_run':
                    continue
                if key not in old_args:
                    raise ValueError(f'Parameter {key} not found in the previous run.')
                if str(value) != old_args[key]:
                    raise ValueError(f'Parameter {key} does not match the previous run. Old: {old_args[key]}, New: {value}')
        
    compute_memorization_statistics(args)