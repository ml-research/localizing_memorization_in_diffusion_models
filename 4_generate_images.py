import os
import random

import torch
from PIL import Image
from torch import autocast
from tqdm.auto import tqdm
from utils.stable_diffusion import generate_images
from utils.stable_diffusion import load_sd_components, load_text_components
import argparse
from utils.datasets import load_prompts
from rtpt import RTPT
import pandas as pd
import json
import re
import sys
from random import sample 

def str_to_list(s):
    pattern = re.compile(r'\[.*?\]')
    sublists = pattern.findall(s)
    return [list(map(int, re.findall(r'\d+', sublist))) for sublist in sublists]

def get_num_neurons_per_layer(unet):
    num_neurons = []
    for layer_idx in range(7):
        if layer_idx < 6:
            num_neurons.append(unet.down_blocks[int(layer_idx / 2)].attentions[layer_idx % 2].transformer_blocks[0].attn2.to_v.out_features )
        else:
            num_neurons.append(unet.mid_block.attentions[0].transformer_blocks[0].attn2.to_v.out_features )
    return num_neurons

@torch.no_grad()
def main():
    args = create_parser()

    vae, unet, scheduler = load_sd_components(args.version)
    tokenizer, text_encoder = load_text_components(args.version)
    
    torch_device = "cuda"
    vae.to(torch_device)
    text_encoder.to(torch_device)
    unet.to(torch_device)

    os.makedirs(args.output_path, exist_ok=False)
    
    with open(os.path.join(args.output_path, "config.json"), "w") as outfile:
        args_to_save = vars(args)
        args_to_save['command'] = " ".join(sys.argv)
        json.dump(args_to_save, outfile)
    
    # only one flag allowed
    assert not (args.initial_neurons and args.refined_neurons and args.original_images and args.block_random_neurons is not None)
    # only one flag allowed
    assert not (args.block_top_k_neurons_per_layer is not None and args.block_top_k_neurons is not None and args.block_top_k_neuron_subgroups is not None and args.block_random_neurons is not None)
    # assert that either the initial neurons or the refined neurons are chosen when blocking the top k neurons
    assert (
        ((args.block_top_k_neurons_per_layer is not None or args.block_top_k_neurons is not None or args.block_top_k_neuron_subgroups is not None) and args.initial_neurons) or 
        ((args.block_top_k_neurons_per_layer is not None or args.block_top_k_neurons is not None or args.block_top_k_neuron_subgroups is not None) and args.refined_neurons) or 
        (args.initial_neurons or args.refined_neurons) or
        args.original_images or
        args.block_random_neurons is not None
    ), "Either the initial neurons or the refined neurons must be chosen when blocking the top k neurons"
    
    # load csv file
    df = pd.read_csv(args.result_file, sep=';')
    
    # filter for vm or tm prompts
    if args.memorization_type is not None:
        df_original_prompts = pd.read_csv('prompts/memorized_laion_prompts.csv', sep=';')
        assert len(df) == len(df_original_prompts)
        df = df[df_original_prompts['type'] == args.memorization_type.upper()]
        if len(df) == 0:
            print(f"No prompts found for the type {args.memorization_type}. Use one of [VM, TM]")
            return
        else:
            print(f'Only taking neurons of {args.memorization_type.upper()} prompts, {len(df)} results remaining')

    if args.block_top_k_neurons_per_layer is not None or args.block_top_k_neurons is not None:
        # get the top k neurons for each layer or over all layers
        neuron_counts = {i: {} for i in range(len(str_to_list(df['Refined Neurons'][0])))}
        for row in df['Refined Neurons' if args.refined_neurons else 'Initial Neurons']:
            for layer_idx, layer in enumerate(str_to_list(row)):
                for neuron in layer:
                    if neuron in neuron_counts[layer_idx]:
                        neuron_counts[layer_idx][neuron] += 1
                    else:
                        neuron_counts[layer_idx][neuron] = 1
        if args.block_top_k_neurons_per_layer is not None:
            neuron_counts = {layer_idx: sorted(neuron_counts[layer_idx].items(), key=lambda x: x[1], reverse=True) for layer_idx in neuron_counts.keys()}
        elif args.block_top_k_neurons is not None:
            neuron_counts_overall = {}
            for layer_idx in neuron_counts.keys():
                for neuron, count in neuron_counts[layer_idx].items():
                    neuron_counts_overall[(layer_idx, neuron)] = count
            neuron_counts_overall = sorted(neuron_counts_overall.items(), key=lambda x: x[1], reverse=True)
            
            neuron_counts = {i: {} for i in range(len(str_to_list(df['Refined Neurons'][0])))}
            for i in range(args.block_top_k_neurons):
                current_layer_idx = neuron_counts_overall[i][0][0]
                current_neuron = neuron_counts_overall[i][0][1]
                count = neuron_counts_overall[i][1]

                neuron_counts[current_layer_idx][current_neuron] = count
            neuron_counts = {layer_idx: sorted(neuron_counts[layer_idx].items(), key=lambda x: x[1], reverse=True) for layer_idx in neuron_counts.keys()}

        # get the top k neurons for each layer
        blocked_indices = [[] for _ in range(7)]
        for layer_idx in neuron_counts.keys():
            blocked_indices[layer_idx] = [neuron[0] for neuron in neuron_counts[layer_idx][:args.block_top_k_neurons_per_layer]]
    elif args.block_top_k_neuron_subgroups is not None or args.block_random_neuron_subgroups is not None:
        num_groups = args.block_top_k_neuron_subgroups if args.block_top_k_neuron_subgroups is not None else args.block_random_neuron_subgroups

        # get the most frequent neuron subgroups
        neuron_subgroup_counts = {}
        for row in df['Refined Neurons' if args.refined_neurons else 'Initial Neurons']:
            # sort the neurons in each layer to prevent permutations
            neuron_list = str_to_list(row)
            neuron_list = [sorted(neuron) for neuron in neuron_list]
            neuron_list_str = str(neuron_list)

            if neuron_list_str in neuron_subgroup_counts:
                neuron_subgroup_counts[neuron_list_str] += 1
            else:
                neuron_subgroup_counts[neuron_list_str] = 1
        neuron_subgroup_counts = sorted(neuron_subgroup_counts.items(), key=lambda x: x[1], reverse=True)

        # get the neurons for each layer of the top k subgroups
        blocked_indices = [[] for _ in range(7)]
        for subgroup_str in neuron_subgroup_counts[:num_groups]:
            for layer_idx in range(len(str_to_list(subgroup_str[0]))):
                subgroup_list = str_to_list(subgroup_str[0])
                for neuron in subgroup_list[layer_idx]:
                    if neuron not in blocked_indices[layer_idx]:
                        blocked_indices[layer_idx].append(neuron)
    elif args.block_random_neurons is not None:
        neuron_counts = {i: {} for i in range(len(str_to_list(df['Refined Neurons'].iloc[0])))}
        for row in df['Refined Neurons' if args.refined_neurons else 'Initial Neurons']:
            for layer_idx, layer in enumerate(str_to_list(row)):
                for neuron in layer:
                    if neuron in neuron_counts[layer_idx]:
                        neuron_counts[layer_idx][neuron] += 1
                    else:
                        neuron_counts[layer_idx][neuron] = 1
        neuron_counts_overall = {}
        for layer_idx in neuron_counts.keys():
            for neuron, count in neuron_counts[layer_idx].items():
                neuron_counts_overall[(layer_idx, neuron)] = count
        neuron_counts_overall = sorted(neuron_counts_overall.items(), key=lambda x: x[1], reverse=True)
        print(f'Found {len(neuron_counts_overall)} neurons')
        
        neuron_counts = {i: {} for i in range(len(str_to_list(df['Refined Neurons'][0])))}
        for i in range(args.block_random_neurons):
            current_layer_idx = neuron_counts_overall[i][0][0]
            current_neuron = neuron_counts_overall[i][0][1]
            count = neuron_counts_overall[i][1]

            neuron_counts[current_layer_idx][current_neuron] = count
        neuron_counts = {layer_idx: sorted(neuron_counts[layer_idx].items(), key=lambda x: x[1], reverse=True) for layer_idx in neuron_counts.keys()}

        # get the top k neurons for each layer
        blocked_indices = [[] for _ in range(7)]
        for layer_idx in neuron_counts.keys():
            blocked_indices[layer_idx] = [neuron[0] for neuron in neuron_counts[layer_idx][:args.block_top_k_neurons_per_layer]]


        # find top 1000 neurons
        neuron_counts = {i: {} for i in range(len(str_to_list(df['Refined Neurons'][0])))}
        for i in range(len(neuron_counts_overall)):
            current_layer_idx = neuron_counts_overall[i][0][0]
            current_neuron = neuron_counts_overall[i][0][1]
            count = neuron_counts_overall[i][1]

            neuron_counts[current_layer_idx][current_neuron] = count
        neuron_counts = {layer_idx: sorted(neuron_counts[layer_idx].items(), key=lambda x: x[1], reverse=True) for layer_idx in neuron_counts.keys()}

        blocked_indices_1000 = [[] for _ in range(7)]
        for layer_idx in neuron_counts.keys():
            blocked_indices_1000[layer_idx] = [neuron[0] for neuron in neuron_counts[layer_idx][:args.block_top_k_neurons_per_layer]]

        print('Most common neurons:', blocked_indices)
        
        # print num elements
        print('Number of neurons:', sum([len(layer) for layer in blocked_indices]))

        # replace neurons with random values
        for layer in range(len(blocked_indices)):
            num_neurons = get_num_neurons_per_layer(unet)[layer]
            candidates = list(range(num_neurons))
            candidates = [ elem for elem in candidates if elem not in blocked_indices_1000[layer] ]
            blocked_indices[layer] = random.sample(range(num_neurons), len(blocked_indices[layer]))
        
        print('Random replacement:', blocked_indices)

    if args.block_random_neuron_subgroups:
        assert args.refined_neurons or args.initial_neurons , "The refined/initial neurons must be chosen when blocking random neuron subgroups"
        # choose for each of the layer the same number of neurons but randomly
        num_neurons_per_layer = get_num_neurons_per_layer(unet)
        # sample random neurons for each layer
        blocked_indices_new = [[] for _ in range(7)]
        for layer_idx in range(len(num_neurons_per_layer)):
            random.seed(args.seed)
            blocked_indices_new[layer_idx] = random.sample(range(num_neurons_per_layer[layer_idx]), len(blocked_indices[layer_idx]))

        blocked_indices = blocked_indices_new


    # if we want to use the unmemorized prompts, load them here
    if args.unmemorized_prompts is not None:
        df = pd.read_csv(args.unmemorized_prompts, sep=';')

    rtpt = RTPT(args.user, 'image generation', len(df) // args.batch_size)
    rtpt.start()
    for i in tqdm(range(len(df) // args.batch_size), total=len(df) // args.batch_size):
        rows = df.iloc[i*args.batch_size:(i+1)*args.batch_size]
        prompts = rows['Caption'].to_list()

        if args.block_top_k_neurons_per_layer is not None or args.block_top_k_neurons is not None or args.block_top_k_neuron_subgroups is not None or args.block_random_neurons or args.block_random_neuron_subgroups:   
            pass
        elif args.initial_neurons:
            blocked_indices = str_to_list(rows.iloc[0]['Initial Neurons'])
        elif args.refined_neurons:
            blocked_indices = str_to_list(rows.iloc[0]['Refined Neurons'])
        elif args.original_images:
            blocked_indices = None
                    
        images = generate_images(prompts, tokenizer, text_encoder, vae, unet, scheduler, num_inference_steps=args.num_steps, blocked_indices=blocked_indices, scaling_factor=args.scaling_factor, guidance_scale=args.guidance_scale, samples_per_prompt=args.num_samples, seed=args.seed)

        for j in range(len(images)):
            images[j].save(f"{args.output_path}/img_{i*args.batch_size + j // args.num_samples:04d}_{j%args.num_samples:02d}.jpg")
        rtpt.step()

def create_parser():
    parser = argparse.ArgumentParser(description='Generating images')
    parser.add_argument(
        '-f',
        '--result_file',
        default='results/memorization_statistics_v1_4.csv',
        type=str,
        dest="result_file",
        help='path to file with image descriptions (default: results/memorization_statistics_v1_4.csv)')
    parser.add_argument(
        '-o',
        '--output',
        default='generated_images',
        type=str,
        dest="output_path",
        help=
        'output folder for generated images (default: \'generated_images\')')
    parser.add_argument('-s',
                        '--seed',
                        default=2,
                        type=int,
                        dest="seed",
                        help='seed for generated images (default: 2')
    parser.add_argument(
        '-n',
        '--num_samples',
        default=10,
        type=int,
        dest="num_samples",
        help='number of generated samples for each prompt (default: 10)')
    parser.add_argument('--steps',
                        default=50,
                        type=int,
                        dest="num_steps",
                        help='number of denoising steps (default: 50)')
    parser.add_argument('-g',
                        '--guidance_scale',
                        default=7,
                        type=float,
                        dest="guidance_scale",
                        help='guidance scale (default: 7)')
    parser.add_argument('-u',
                        '--user',
                        default='XX',
                        type=str,
                        dest="user",
                        help='name initials for RTPT (default: "XX")')
    parser.add_argument('-v',
                        '--version',
                        default='v1-4',
                        type=str,
                        dest="version",
                        help='Stable Diffusion version (default: "v1-4")')
    parser.add_argument('-b', '--batch_size', default=1, type=int, help='Number of prompts per batch')
    parser.add_argument('--original_images', action='store_true', default=False, help='Generate the original images')
    parser.add_argument('--initial_neurons', action='store_true', default=False, help='Block initial neurons')
    parser.add_argument('--refined_neurons', action='store_true', default=False, help='Block refined neurons')
    parser.add_argument('--block_top_k_neurons_per_layer', default=None, type=int, help='Blocks the top k found neurons for each layer for all the memorized sampels')
    parser.add_argument('--block_top_k_neurons', default=None, type=int, help='Blocks the top k found neurons over all layers for all the memorized sampels')
    parser.add_argument('--block_top_k_neuron_subgroups', default=None, type=int, help='Blocks the top k found neuron subgroups for all the memorized sampels')
    parser.add_argument('--block_random_neuron_subgroups', default=None, type=int, help='Blocks random neurons based on what the subgroups for the memorized samples were found.')
    parser.add_argument('--block_random_neurons', default=None, type=int, help='Blocks random neurons throughout all layers')
    parser.add_argument('--unmemorized_prompts', default=None, type=str, help='Path to the unmemorized prompt files. If set, the unmemorized prompts will be used instead of the memorized prompts. Only usable when blocking the top k or random neurons')
    parser.add_argument('--memorization_type', default=None, type=str, help='Decide if the neurons of the verbatim or template prompts should be used')
    parser.add_argument('--scaling_factor', default=0, type=float, help='Scaling factor for the blocking of neurons')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()