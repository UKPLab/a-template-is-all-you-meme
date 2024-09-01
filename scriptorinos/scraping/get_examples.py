import requests
from bs4 import BeautifulSoup
import json
import time
import os
import re
from tqdm import tqdm
from utils import request_get_snapshot
import glob
examples = 'meme_examples'
jsons = 'jsons'
if not os.path.exists(examples):
    os.makedirs(examples)
if not os.path.exists(jsons):
    os.makedirs(jsons)

def process_images():
    img_data = requests.get(image_snapshot['url']).content
    pic_dir ='{}{}_{}'.format(examples, hit_count, pic_title)
        
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)
    
    example_title = '{}_{}.jpg'.format(example_count, pic_title)

    out_json = {'url': html_file,
                'title': title,
                'image': image_snapshot['url'],
                'about': about,
                'pic_directory': pic_dir,
                'example_title': example_title}
    print(out_json)
    with open('{}/{}'.format(jsons, 'example_pictures.json'), 'a') as f:
        f.write(json.dumps(out_json)+'\n')
    with open('{}/{}'.format(pic_dir, example_title), 'wb') as handler:
        handler.write(img_data)
    print()
    time.sleep(1)

def process_misses(if_statement=False):
    if if_statement:
        out_json = {'filename': html_file, 'title': title,'count': miss_count, 'exception': 'failed if'}
        with open('{}/{}'.format(jsons, 'miss_examples.json'), 'a') as f:
            f.write(json.dumps(out_json)+'\n')
    else:
        out_json = {'filename': html_file, 'title': title,'count': miss_count, 'exception': 'exception'}
        with open('{}/{}'.format(jsons, 'miss_examples.json'), 'a') as f:
            f.write(json.dumps(out_json)+'\n')
        

print('sleep check')
through = 0


for html_file in tqdm(glob.glob('snapshots/*/*_tablecontent.html')):
    time.sleep(1)
    hit_files = []
    miss_files = []
    seen_files = [] 
    try:
        with open('{}/{}'.format(jsons, 'example_pictures.json'), 'r') as f:
            for line in f:
                #print(line)
                results = dict(json.loads(line))
                file_name = results['url']
                hit_files.append(file_name)
                seen_files.append(file_name)        
        
        with open('{}/{}'.format(jsons, 'miss_examples.json'), 'r') as f:
            for line in f:
                #print(line)
                results = dict(json.loads(line))
                file_name = results['filename']
                miss_files.append(file_name)
                seen_files.append(file_name)
       
    except Exception as e:
         print('first Exception')
         print(e)
    hits = set(hit_files)
    print()
    hit_count = len(hits)
    misses = set(miss_files)
    miss_count = len(misses)
    
    seen_files = set(seen_files)
    if html_file in seen_files:
        continue
    
    with open(html_file, 'rb') as f:
        soup = BeautifulSoup(f, 'html.parser')
    
    try:
        title = soup.find("meta", property="og:title")["content"]
    except Exception as e:
        process_misses()
        continue
   
    pic_title = re.sub(r'\W+', '', title)
    about = soup.find("meta", property="og:description")["content"]
    section = soup.find_all("section", class_='bodycopy')
    for html_stuff in section:
        try:
            images = html_stuff.findAll('img')
            example_count = 0
            for img in images:
                example_count+=1
                link = img['data-src']
                print('link')
                link = link.split('/')
                image_url = "/".join(link[5:])
                print(image_url)
                image_snapshot = request_get_snapshot(image_url)
                if image_snapshot:
                    
                    print('snapshot')
                    
                    try:
                        process_images()
                    except:
                        print('sleep time 2')
                        time.sleep(60*10)
                        try:
                            process_images()
                        except Exception as e:
                            print(e)
                            #miss_count+=1
                            process_misses()                           
                            continue
                else:
                    #miss_count+=1
                    process_misses(if_statement=True)
                    continue 

        except Exception as e:
            print('last exception')
            print(e)
            #miss_count+=1
            process_misses()
            continue

print('jobs done!')

        
    