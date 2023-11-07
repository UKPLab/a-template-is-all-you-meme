import json
import glob
import os
import shutil
from tqdm import tqdm

workstation = 'workstation_scrapes/meme_examples/*_'
cluster = 'memes/meme_examples/*_'
template_info = []
with open('template_examples/jsons/template_info.json', 'r') as f:
        for line in f:
            template_info.append(dict(json.loads(line)))
count = 0
for dictionary1 in tqdm(template_info):
    for key in dictionary1.keys():
        examples, seen_bytes = [], []

        #"out_paths": ["template_examples/templates/SansinSmash/SansinSmash.jpg"
        write_path = dictionary1[key]['out_paths'][0].split('/')[:-1]
        write_title = write_path[-1]
        write_path = '/'.join(write_path)
        write_path = f'{write_path}/examples'
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        station_title_path = f'{workstation}{write_title}/*'
        #print(glob.glob(station_title_path))
        cluster_title_path = f'{cluster}{write_title}/*'
        for example in glob.glob(station_title_path):
            examples.append(example)
        for example in glob.glob(cluster_title_path):
            examples.append(example)
    
        for idx, example in enumerate(examples):
            src = example
            dst = example.split('/')[-1].split('_')[-1]
            dst = f'{write_path}/{idx}_{dst}'
            if len(dst.split('/')) !=5:
                 print(example)
                 print(dst)
                 print()
                       
           
            shutil.copyfile(src, dst)
            count+=1
            if count%1000 == 0:
                print(count)
                print('copied files')

print('jobs done!')