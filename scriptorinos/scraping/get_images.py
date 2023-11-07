import requests
from bs4 import BeautifulSoup
import json
import time
import os
import re
from tqdm import tqdm
import glob

pictures = 'pictures'
jsons = 'jsons'
if not os.path.exists(pictures):
    os.makedirs(pictures)

def request_get_snapshot(url):
    try:
        response = requests.get(f'http://archive.org/wayback/available?url={url}')
    except:
        time.sleep(60*10)
        response = requests.get(f'http://archive.org/wayback/available?url={url}')
    print(response)
    try:
        data = response.json()
    except:
        return None
    
    if 'closest' in data['archived_snapshots']:
        snapshot = data['archived_snapshots']['closest']
        assert snapshot['available'] is True, f'Snapshot not available: {url}'
        assert snapshot['url'] is not None, f'Snapshot URL is None: {url}'
        return snapshot
    return None

hit_count = 0
miss_count = 0
through = 0
for html_file in tqdm(glob.glob('snapshots/*/*_tablecontent.html')):
    with open(html_file, 'rb') as f:
        soup = BeautifulSoup(f, 'html.parser')
    #image = soup.find("meta", property="og:image")["content"]
    #image_snapshot = request_get_snapshot(image)
    #if image_snapshot:
    hit_count+=1
    try:
        image = soup.find("meta", property="og:image")["content"]
    except:
        miss_count+=1
        out_json = {'filename': html_file, 'count': miss_count}
        with open('{}/{}'.format(jsons, 'miss_images.json'), 'a') as f:
            f.write(json.dumps(out_json)+'\n')
        continue

    title = soup.find("meta", property="og:title")["content"]
    about = soup.find("meta", property="og:description")["content"]
    try:
        img_data = requests.get(image).content
    except:
        time.sleep(60*10)
        try:
            img_data = requests.get(image).content
        except:
            miss_count+=1
            out_json = {'filename': html_file, 'count': miss_count}
            with open('{}/{}'.format(jsons, 'miss_images.json'), 'a') as f:
                f.write(json.dumps(out_json)+'\n')
            continue

        
    pic_title = re.sub(r'\W+', '', title)
    pic_title ='pictures/{}_{}.jpg'.format(hit_count, pic_title) 

    out_json = {'url': image,
                'title': title,
                'image': image,
                'about': about,
                'pic_title': pic_title}
    print(out_json)
    with open('{}/{}'.format(jsons, 'meme_pictures.json'), 'a') as f:
        f.write(json.dumps(out_json)+'\n')
    with open(pic_title, 'wb') as handler:
        handler.write(img_data)
    print()
    time.sleep(1)
    #else:
    #    miss_count+=1
    #    out_json = {'filename': html_file, 'count': miss_count}
    #    with open('{}/{}'.format(jsons, 'miss_images.json'), 'a') as f:
    #        f.write(json.dumps(out_json)+'\n')
print('jobs done!')