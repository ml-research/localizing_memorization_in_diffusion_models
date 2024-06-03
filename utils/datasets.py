import json
import requests 

def load_prompts(file_path):
    with open(file_path, 'r') as f:
        samples = []
        if '.json' in file_path:
            for line in f:
                samples.append(json.loads(line)['caption'].strip())
        elif '.txt' in file_path:
            for line in f:
                samples.append(line.strip())
        else:
            raise ValueError('Invalid file type')
    return samples


def load_images(file_path, output_folder):
    with open(file_path, 'r') as f:
        urls = []
        if '.json' in file_path:
            for line in f:
                urls.append((json.loads(line)['index'], json.loads(line)['url'].strip()))
        for i, (index, url) in enumerate(urls):
            try:
                data = requests.get(url, verify=False).content 
                f = open(f'{output_folder}/{i:04d}_{index}.png','wb')
                f.write(data) 
                f.close() 
            except:
                print(f'Error downloading {url}')