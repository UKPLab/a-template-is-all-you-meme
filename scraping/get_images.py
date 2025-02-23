import requests
from bs4 import BeautifulSoup
import json
import time
import os
import re
from tqdm import tqdm
import glob

# Directories
pictures = 'pictures'
jsons = 'jsons'
os.makedirs(pictures, exist_ok=True)
os.makedirs(jsons, exist_ok=True)

# Function to request the Wayback Machine snapshot
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

# Counters
hit_count = 0
miss_count = 0

# Iterate over HTML files in 'selenium_snapshots' directory
for html_file in tqdm(glob.glob('selenium_snapshots/*/*_tablecontent.html')):  # Adjust the file path as needed
    with open(html_file, 'rb') as f:
        soup = BeautifulSoup(f, 'html.parser')

    hit_count += 1

    # Attempt to extract the 'og:image' meta tag content (image URL)
    try:
        image = soup.find("meta", property="og:image")["content"]
    except:
        miss_count += 1
        out_json = {'filename': html_file, 'count': miss_count}
        with open(f'{jsons}/miss_images.json', 'a') as f:
            f.write(json.dumps(out_json) + '\n')
        continue

    # Extract other metadata fields safely
    title = soup.find("meta", property="og:title")["content"] if soup.find("meta", property="og:title") else "No title"
    about = soup.find("meta", property="og:description")["content"] if soup.find("meta", property="og:description") else "No description"
    origin = soup.find("meta", property="og:site_name")["content"] if soup.find("meta", property="og:site_name") else "No origin"
    
    # Collect additional text data
    other_meta = {}
    for meta in soup.find_all("meta"):
        if "name" in meta.attrs:
            name = meta.attrs["name"]
            content = meta.attrs.get("content", "")
            if content:
                other_meta[name] = content

    # Handle image download with retry mechanism
    try:
        img_data = requests.get(image).content
    except:
        time.sleep(60*10)
        try:
            img_data = requests.get(image).content
        except:
            miss_count += 1
            out_json = {'filename': html_file, 'count': miss_count}
            with open(f'{jsons}/miss_images.json', 'a') as f:
                f.write(json.dumps(out_json) + '\n')
            continue

    # Clean and format the image title (remove non-alphanumeric characters)
    pic_title = re.sub(r'\W+', '', title)
    pic_title = f'pictures/{hit_count}_{pic_title}.jpg'

    # Prepare the output JSON
    out_json = {
        'url': image,
        'title': title,
        'about': about,
        'origin': origin,
        'pic_title': pic_title,
        'other_meta': other_meta  # Save other meta tags as additional information
    }

    # Print the metadata for debugging
    print(out_json)

    # Save the meme picture metadata
    with open(f'{jsons}/meme_pictures.json', 'a') as f:
        f.write(json.dumps(out_json) + '\n')

    # Save the image to the pictures folder
    with open(pic_title, 'wb') as handler:
        handler.write(img_data)

    # Add a delay to prevent overloading the server
    time.sleep(1)

print('Jobs done!')
