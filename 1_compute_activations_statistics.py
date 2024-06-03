from utils.stable_diffusion import load_sd_components, load_text_components
from rtpt import RTPT
import torch
from tqdm.auto import tqdm
import pandas as pd
import argparse


def main():
    args = create_parser()

    # load prompts
    prompts = pd.read_csv(args.prompts, sep=';')

    RTPT(args.user, 'Compute_Activations', len(prompts)).start()

    # Load SD components
    vae, unet, scheduler = load_sd_components(args.version)
    tokenizer, text_encoder = load_text_components(args.version)

    torch_device = "cuda"
    vae.to(torch_device)
    text_encoder.to(torch_device)
    unet.to(torch_device)

    value_layers = torch.nn.ModuleList()
    for down_block in range(3):
        for attention in range(2):
            value_layers.append(unet.down_blocks[down_block].attentions[attention].transformer_blocks[0].attn2.to_v)
    value_layers.append(unet.mid_block.attentions[0].transformer_blocks[0].attn2.to_v)

    list_of_activations = [[] for _ in range(7)]

    for prompt in tqdm(prompts.iterrows(), total=len(prompts)):
        prompt = prompt[1]['Caption']
        
        with torch.no_grad():
            text_input = tokenizer([prompt],
                                    padding="max_length",
                                    max_length=tokenizer.model_max_length,
                                    truncation=True,
                                    return_tensors="pt")
            text_embeddings = text_encoder(
                text_input.input_ids.to(text_encoder.device))[0]
            for idx, layer in enumerate(value_layers):
                activations = layer.forward(text_embeddings)[0]
                activations = activations.abs().mean(dim=0).unsqueeze(0).cpu()
                list_of_activations[idx].append(activations)
                
    if args.no_statistics:
        result = []
        for idx in range(len(value_layers)):
            result.append(torch.cat(list_of_activations[idx], dim=0))
        torch.save(result, args.output.replace('.pt', f'_no_statistics_calculated_{args.version.replace("-", "_")}.pt'))
        return

    mean_list = []
    std_list = []
    for idx in range(len(value_layers)):
        mean = torch.cat(list_of_activations[idx], dim=0).mean(dim=0)
        std = torch.cat(list_of_activations[idx], dim=0).std(dim=0)    
        mean_list.append(mean)
        std_list.append(std)
        
    torch.save((mean_list, std_list), args.output.replace('.pt', '') + '_' + args.version.replace("-", "_") + '.pt')


def create_parser():
    parser = argparse.ArgumentParser(description='Calculating activation statistics')
    
    parser.add_argument('--prompts', default='prompts/additional_laion_prompts.csv', type=str, help='The file from which the prompts are loaded to calculate the statistics (default: \'prompts/additional_laion_prompts.csv\').')
    parser.add_argument('--output', default='statistics/statistics_additional_laion_prompts.pt', type=str, help='The file to which the activation statistics are written (default: \'statistics/statistics_additional_laion_prompts.pt\').')
    parser.add_argument('-v',
                        '--version',
                        default='v1-4',
                        type=str,
                        dest="version",
                        help='Stable Diffusion version (default: "v1-4")')
    parser.add_argument('-u',
                    '--user',
                    default='XX',
                    type=str,
                    dest="user",
                    help='name initials for RTPT (default: "XX")')
    parser.add_argument(
        '--no_statistics',
        default=False,
        action='store_true',
        help='Do not calculate the statistics and instead just return the activations.'
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()