import argparse
from rtpt import RTPT
import os
import clip
import torch
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import pandas as pd

# Download SSCD model first using 'wget https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt'

@torch.no_grad()
def compute_similarity_scores(args):
    folder = args.folder
    ref_folder = args.reference
    
    torch.set_num_threads(4)

    files = sorted(os.listdir(folder))
    files_reference = sorted(os.listdir(ref_folder))
        
    assert len(files) == len(files_reference), 'Number of images in the folder and reference folder should be the same'
    
    if 'prompts' in args:
        df = pd.read_csv(args.prompts, sep=';')
    else:
        print('No prompt file provided.')
    
    if args.method == 'clip':
        model, preprocess = clip.load("ViT-B/32", device='cuda')
    elif args.method == 'sscd':
        model = torch.jit.load("sscd_disc_mixup.torchscript.pt").cuda()
        preprocess = transforms.Compose([
            transforms.Resize([320, 320]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        raise ValueError('Invalid method. Select one of [clip, sscd]')

    rtpt = RTPT(args.name, 'Sim score', len(files))
    rtpt.start()
    
    sim_scores = []
    sim_scores_vm = []
    sim_scores_tm = []

    for id in tqdm(range(len(files) // args.num_samples)):
        # load images
        imgs = []
        references = []
        for sample_num in range(args.num_samples):
            img = Image.open(os.path.join(folder, f'img_{id:04d}_{sample_num:02d}.jpg'))
            img = preprocess(img).unsqueeze(0).to('cuda')
            reference = Image.open(os.path.join(ref_folder, f'img_{id:04d}_{sample_num:02d}.jpg'))
            reference = preprocess(reference).unsqueeze(0).to('cuda')
            imgs.append(img)
            references.append(reference)
        
        imgs = torch.cat(imgs, dim=0)
        references = torch.cat(references, dim=0)
    
        # compute embeddings
        if args.method == 'clip':
            embeddings = model.encode_image(torch.cat([imgs, references], dim=0))
        elif args.method == 'sscd':
            embeddings = model(torch.cat([imgs, references], dim=0))
        
        # compute similarity score
        similarity_score = cosine_similarity(embeddings[:args.num_samples], embeddings[args.num_samples:]).cpu()
        sim_scores.append(similarity_score.median())
        
        if 'prompts' in args:
            if 'type' in df.iloc[id]:
                if df.iloc[id]['type'] == 'VM':
                    sim_scores_vm.append(similarity_score.median())
                elif df.iloc[id]['type'] == 'TM':
                    sim_scores_tm.append(similarity_score.median())
                else:
                    print(f'Invalid memorization type {df.iloc[id]["type"]}')  
            else:
                print('No memorization type provided')  

            
                
        # log the similarity score
        rtpt.step()
    sim_scores = torch.stack(sim_scores)
    median = sim_scores.median().item()
    deviation = (sim_scores - median).abs().median().item()
    print(f'Median similarity score: {median:.4f}±{deviation:.2f}')
    
    # compute statistics over VM samples
    if len(sim_scores_vm) > 0:
        sim_scores_vm = torch.stack(sim_scores_vm)
        median_vm = sim_scores_vm.median().item()
        deviation_vm = (sim_scores_vm - median_vm).abs().median().item()
        print(f'Median similarity score (VM) for {len(sim_scores_vm)} samples: {median_vm:.4f}±{deviation_vm:.2f}')
    
    # compute statistics over TM samples
    if len(sim_scores_tm) > 0:
        sim_scores_tm = torch.stack(sim_scores_tm)
        median_tm = sim_scores_tm.median().item()
        deviation_tm = (sim_scores_tm - median_tm).abs().median().item()
        print(f'Median similarity score (TM) for {len(sim_scores_tm)} samples: {median_tm:.4f}±{deviation_tm:.2f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str, help='Folder 1 containing the images')
    parser.add_argument('-r', '--reference', type=str, help='Folder 2 containing the reference images')
    parser.add_argument('-m', '--method', default='sscd', type=str, help='Method to compute the similarity score. Select one of [clip, sscd]. (Default: sscd)')
    parser.add_argument('-n', '--name', default='XX',  type=str,help='RTPT user name (Default: XX)')
    parser.add_argument('--num_samples', default=10, type=int, help='Number of samples per prompt to compute the similarity score (Default: 10)')
    parser.add_argument('-p', '--prompts', type=str, help='csv file containing the prompts')

    args = parser.parse_args()
    
    compute_similarity_scores(args)