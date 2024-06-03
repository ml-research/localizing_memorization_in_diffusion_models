import argparse
from rtpt import RTPT
import os
import clip
import torch
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from PIL import Image
import pandas as pd

@torch.no_grad()
def compute_similarity_scores(args):
    image_folder = args.folder
    image_files = sorted(os.listdir(image_folder))
    
    torch.set_num_threads(4)
    
    # load csv file
    df = pd.read_csv(args.prompts, sep=';')
    
    assert len(image_files) // args.num_samples == len(df) , 'Number of images in the folder and number of prompts should be the same'
    
    model, preprocess = clip.load("ViT-B/32", device='cuda')

    rtpt = RTPT(args.name, 'Alignment score', len(df))
    rtpt.start()
    
    alignment_scores = []
    alignment_scores_vm = []
    alignment_scores_tm = []

    
    for id, row in tqdm(enumerate(df.iterrows())):
        # load images
        imgs = []
        for sample_num in range(args.num_samples):
            img = Image.open(os.path.join(image_folder, f'img_{id:04d}_{sample_num:02d}.jpg'))
            img = preprocess(img).unsqueeze(0).to('cuda')
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)
        
        # compute embeddings
        image_features = model.encode_image(imgs)
        text = clip.tokenize([row[1]['Caption']]).to('cuda')
        text_features = model.encode_text(text)
        
        # compute similarity score
        similarity_score = cosine_similarity(image_features, text_features).cpu()
        alignment_scores.append(similarity_score.median())
        
        if 'type' in row[1]:
            if row[1]['type'] == 'VM':
                alignment_scores_vm.append(similarity_score.median())
            elif row[1]['type'] == 'TM':
                alignment_scores_tm.append(similarity_score.median())
            else:
                print(f'Invalid memorization type {row[1]["type"]}')  
        else:
            print('No memorization type provided')  
                    
        rtpt.step()
        
    # compute statistics over the whole set
    alignment_scores = torch.stack(alignment_scores)
    median = alignment_scores.median().item()
    deviation = (alignment_scores - median).abs().median().item()
    print(f'Median similarity score (Overall): {median:.4f}±{deviation:.2f}')
    
    # compute statistics over VM samples
    if len(alignment_scores_vm) > 0:
        alignment_scores_vm = torch.stack(alignment_scores_vm)
        median_vm = alignment_scores_vm.median().item()
        deviation_vm = (alignment_scores_vm - median_vm).abs().median().item()
        print(f'Median similarity score (VM) for {len(alignment_scores_vm)} samples: {median_vm:.4f}±{deviation_vm:.2f}')
    
    # compute statistics over TM samples
    if len(alignment_scores_tm) > 0:
        alignment_scores_tm = torch.stack(alignment_scores_tm)
        median_tm = alignment_scores_tm.median().item()
        deviation_tm = (alignment_scores_tm - median_tm).abs().median().item()
        print(f'Median similarity score (TM) for {len(alignment_scores_tm)} samples: {median_tm:.4f}±{deviation_tm:.2f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str, help='Folder containing the images')
    parser.add_argument('-p', '--prompts', type=str, help='csv file containing the prompts')
    parser.add_argument('-n', '--name', default='XX',  type=str,help='RTPT user name (Default: XX)')
    parser.add_argument('--num_samples', default=10, type=int, help='Number of samples per prompt to compute the alignment score (Default: 10)')
    
    args = parser.parse_args()
    
    compute_similarity_scores(args)