from torchvision import transforms
import torch
import os
from PIL import Image
import pickle
from tqdm import tqdm
from rtpt import RTPT
import argparse
import pandas as pd
import re
from torchmetrics.functional.pairwise import pairwise_cosine_similarity

# Download model first via wget https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt

@torch.no_grad()
def compute_sscd_scores(args):
    rtpt = RTPT(args.name, 'SSCD', len(os.listdir(args.reference)))
    rtpt.start()

    model = torch.jit.load("sscd_disc_mixup.torchscript.pt").cuda().eval()

    torch.set_num_threads(4)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
    )
    skew_320 = transforms.Compose([
        transforms.Resize([320, 320]),
        transforms.ToTensor(),
        normalize,
    ])

    result_dict = {}
    
    sim_scores_max = []
    sim_scores_vm_max = []
    sim_scores_tm_max = []
    sim_scores_median = []
    sim_scores_vm_median = []
    sim_scores_tm_median = []

    if 'prompts' in args:
        df = pd.read_csv(args.prompts, sep=';')
    else:
        print('No prompt file provided.')

    files = sorted([f for f in os.listdir(args.folder) if re.search(r'.(png|jpg)', f)])
    reference_files = sorted([f for f in os.listdir(args.reference) if re.search(r'.(png|jpg)', f)])

    assert len(files) // args.num_samples == len(reference_files) // args.num_ref_samples, 'Number of images in the folder and reference folder should be the same'

    for idx in tqdm(range(len(reference_files) // args.num_ref_samples)):
        try:
            real_images = []
            for i in range(args.num_ref_samples):
                img = f'img_{idx:04d}_{i:02d}.jpg' if reference_files[idx + args.num_ref_samples].endswith('.jpg') else list(filter(lambda x: x.startswith(f'{idx:04d}'), reference_files))[0]
                image_real = Image.open(os.path.join(args.reference, img)).convert('RGB')
                image_real = skew_320(image_real).unsqueeze(0).cuda()
                real_images.append(image_real)
            real_images = torch.cat(real_images, dim=0).cuda()
        except:
            print('Image corrupted: ', img)
            continue
        
        images_generated = []
        for i in range(args.num_samples):
            image_generated = Image.open(os.path.join(args.folder, f'img_{idx:04d}_{i:02d}.jpg')).convert('RGB')
            image_generated = skew_320(image_generated).unsqueeze(0).cuda()
            images_generated.append(image_generated)
        images_generated = torch.cat(images_generated, dim=0).cuda()
        
        with torch.no_grad():
            features_real = model(real_images)
            features_generated = model(images_generated)
            cosine_similarity = pairwise_cosine_similarity(features_real, features_generated).median(0).values.detach().cpu()
            euclidean_distance = torch.cdist(features_real, features_generated, p=2).median(0).values.detach().cpu()
        result_dict[idx] = {'img': img, 'cosine_similarity': cosine_similarity, 'euclidean_distance': euclidean_distance}

        sim_scores_max.append(cosine_similarity.max())
        sim_scores_median.append(cosine_similarity.median())

        if 'prompts' in args:
            if 'type' in df.iloc[idx]:
                if df.iloc[idx]['type'] == 'VM':
                    sim_scores_vm_max.append(cosine_similarity.max())
                    sim_scores_vm_median.append(cosine_similarity.median())
                elif df.iloc[idx]['type'] == 'TM':
                    sim_scores_tm_max.append(cosine_similarity.max())
                    sim_scores_tm_median.append(cosine_similarity.median())
                else:
                    print(f'Invalid memorization type {df.iloc[idx]["type"]}')  
            else:
                print('No memorization type provided')  
        rtpt.step()
        
    sim_scores_max = torch.stack(sim_scores_max)
    median = sim_scores_max.median().item()
    deviation = (sim_scores_max - median).abs().median().item()
    result_dict['max'] = {'median': median, 'deviation': deviation}
    print(f'Median similarity score (Max): {median:.4f}±{deviation:.2f}')
    
    sim_scores_median = torch.stack(sim_scores_median)
    median = sim_scores_median.median().item()
    deviation = (sim_scores_median - median).abs().median().item()
    result_dict['median'] = {'median': median, 'deviation': deviation}
    print(f'Median similarity score (Median): {median:.4f}±{deviation:.2f}')

    # compute statistics over VM samples
    if len(sim_scores_vm_max) > 0:
        sim_scores_vm_max = torch.stack(sim_scores_vm_max)
        median_vm = sim_scores_vm_max.median().item()
        deviation_vm = (sim_scores_vm_max - median_vm).abs().median().item()
        result_dict['max_vm'] = {'median': median_vm, 'deviation': deviation_vm}
        print(f'\nMedian similarity score (Max, VM) for {len(sim_scores_vm_max)} samples: {median_vm:.4f}±{deviation_vm:.2f}')
        
        sim_scores_vm_median = torch.stack(sim_scores_vm_median)
        median_vm = sim_scores_vm_median.median().item()
        deviation_vm = (sim_scores_vm_median - median_vm).abs().median().item()
        result_dict['median_vm'] = {'median': median_vm, 'deviation': deviation_vm}
        print(f'Median similarity score (Median, VM) for {len(sim_scores_vm_median)} samples: {median_vm:.4f}±{deviation_vm:.2f}')
    
    # compute statistics over TM samples
    if len(sim_scores_tm_max) > 0:
        sim_scores_tm_max = torch.stack(sim_scores_tm_max)
        median_tm = sim_scores_tm_max.median().item()
        deviation_tm = (sim_scores_tm_max - median_tm).abs().median().item()
        result_dict['max_tm'] = {'median': median_tm, 'deviation': deviation_tm}
        print(f'\nMedian similarity score (Max, TM) for {len(sim_scores_tm_max)} samples: {median_tm:.4f}±{deviation_tm:.2f}')

        sim_scores_tm_median = torch.stack(sim_scores_tm_median)
        median_tm = sim_scores_tm_median.median().item()
        deviation_tm = (sim_scores_tm_median - median_tm).abs().median().item()
        result_dict['median_tm'] = {'median': median_tm, 'deviation': deviation_tm}
        print(f'Median similarity score (Median, TM) for {len(sim_scores_tm_median)} samples: {median_tm:.4f}±{deviation_tm:.2f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str, help='Folder containing the generated images')
    parser.add_argument('-r', '--reference', type=str, help='Folder containing the real images')
    parser.add_argument('-n', '--name', default='XX',  type=str,help='RTPT user name (Default: XX)')
    parser.add_argument('--num_samples', default=10, type=int, help='Number of samples per prompt to compute the similarity score (Default: 10)')
    parser.add_argument('--num_ref_samples', default=1, type=int, help='Number of reference samples per prompt to compute the similarity score (Default: 1)')
    parser.add_argument('-p', '--prompts', type=str, help='csv file containing the prompts')

    args = parser.parse_args()
    
    compute_sscd_scores(args)