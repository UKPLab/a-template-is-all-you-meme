import requests
from bs4 import BeautifulSoup
import json
import time
import os
import re
from tqdm import tqdm

snapshots = 'snapshots/'
jsons = 'jsons'
if not os.path.exists(jsons):
    os.makedirs(jsons)

def request_get_snapshot(url):
 
    try:
        response = requests.get(f'http://archive.org/wayback/available?url={url}')
    except:
        time.sleep(60*10)
        response = requests.get(f'http://archive.org/wayback/available?url={url}')
    data = response.json()
    if 'closest' in data['archived_snapshots']:
        snapshot = data['archived_snapshots']['closest']
        assert snapshot['available'] is True, f'Snapshot not available: {url}'
        assert snapshot['url'] is not None, f'Snapshot URL is None: {url}'
        return snapshot
    return None

with open('html_data/3749_basecontent.html', 'rb') as f:# last table found. replace with your own
    base_soup = BeautifulSoup(f, 'html.parser')

table = base_soup.find('table', class_='entry_list')
urls = set(table.findAll('a', href=True))
meme_urls = set()

seen_urls = set()
counts = []
if os.path.isfile('{}/meme_tables.json'.format(jsons)):
    with open('{}/meme_tables.json'.format(jsons), 'r') as f:
        for line in f:
            print(line)
            line = dict(json.loads(line))
            this_url = line['url']
            counts.append(line['count'])
            if this_url == 'this is where the fun begins':
                continue
            seen_urls.add(this_url)
            
else:
    with open('{}/meme_tables.json'.format(jsons), 'a') as f:
        start_json = {'url': 'this is where the fun begins', 'count': 0}
        f.write(json.dumps(start_json)+'\n')




for url in urls:
    url = url['href']
    if not url.startswith('/memes/'):
        continue
    else:
         if 'https://knowyourmeme.com/'+url not in seen_urls:
            meme_urls.add('https://knowyourmeme.com/'+url)

meme_url_count = len(counts)
if meme_url_count > 0:
    meme_url_count-=1

if os.path.isfile('{}/{}'.format(jsons, 'miss.json')):
    miss_count = 0
    with open('{}/{}'.format(jsons, 'miss.json'), 'r') as f:
        for line in f:
            miss_count+=1
else:
    miss_count = 0

for meme_url in tqdm(meme_urls):          
    snapshot = request_get_snapshot(meme_url)
    if snapshot:
        timestamp = snapshot['timestamp']
        outdir = snapshots + timestamp 
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        print(meme_url)
        print(snapshot['url'])
        meme_url_count+=1
        try:
            response = requests.get(snapshot['url'])
        except:
            time.sleep(60*10)
            response = requests.get(snapshot['url'])

        with open('{}/{}_tablecontent.html'.format(outdir, meme_url_count), 'wb+') as f:
            f.write(response.content)
        
        out_json = {'url': snapshot['url'],
                    'count': meme_url_count}
        print(out_json)
        with open('{}/{}'.format(jsons, 'meme_tables.json'), 'a') as f:
            f.write(json.dumps(out_json)+'\n')
        print()
        
        time.sleep(1)
    else:
        miss_count+=1
        out_json = {'url': meme_url, 'count': miss_count}
        with open('{}/{}'.format(jsons, 'miss.json'), 'a') as f:
            f.write(json.dumps(out_json)+'\n')
