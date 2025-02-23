import requests
from bs4 import BeautifulSoup
import json
import time
import os
import re
from tqdm import tqdm
from utils import request_get_snapshot
import glob

# Define constants for paths
examples = 'meme_examples'
jsons = 'jsons'
if not os.path.exists(examples):
    os.makedirs(examples)
if not os.path.exists(jsons):
    os.makedirs(jsons)

# Define a function for processing images
def process_images(hit_count, pic_title, example_count, image_snapshot, html_file, title, about):
    """Download image and store metadata."""
    img_data = requests.get(image_snapshot['url']).content
    pic_dir = os.path.join(examples, f"{hit_count}_{pic_title}")
    
    # Create directory if it doesn't exist
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)
    
    example_title = f"{example_count}_{pic_title}.jpg"

    # Save image metadata to JSON
    out_json = {
        'url': html_file,
        'title': title,
        'image': image_snapshot['url'],
        'about': about,
        'pic_directory': pic_dir,
        'example_title': example_title
    }
    save_json(out_json, 'example_pictures.json')

    # Save image to disk
    with open(os.path.join(pic_dir, example_title), 'wb') as handler:
        handler.write(img_data)

    time.sleep(1)

def process_misses(html_file, title, miss_count, if_statement=False):
    """Handle cases where an image download or snapshot fails."""
    exception_type = 'failed if' if if_statement else 'exception'
    out_json = {
        'filename': html_file,
        'title': title,
        'count': miss_count,
        'exception': exception_type
    }
    save_json(out_json, 'miss_examples.json')

def save_json(data, filename):
    """Helper function to write JSON data to file."""
    with open(os.path.join(jsons, filename), 'a') as f:
        f.write(json.dumps(data) + '\n')

def process_files(html_file, hit_files, miss_files, seen_files, hit_count, miss_count):
    """Process individual HTML files and extract metadata."""
    if html_file in seen_files:
        return hit_count, miss_count

    with open(html_file, 'rb') as f:
        soup = BeautifulSoup(f, 'html.parser')

    try:
        title = soup.find("meta", property="og:title")["content"]
    except Exception as e:
        process_misses(html_file, "", miss_count)
        return hit_count, miss_count

    pic_title = re.sub(r'\W+', '', title)
    about = soup.find("meta", property="og:description")["content"]
    section = soup.find_all("section", class_='bodycopy')

    example_count = 0
    for html_stuff in section:
        try:
            images = html_stuff.findAll('img')
            for img in images:
                example_count += 1
                link = img['data-src']
                print('link:', link)
                link_parts = link.split('/')
                image_url = "/".join(link_parts[5:])
                print('image_url:', image_url)
                
                # Request snapshot for image URL
                image_snapshot = request_get_snapshot(image_url)
                
                if image_snapshot:
                    try:
                        process_images(hit_count, pic_title, example_count, image_snapshot, html_file, title, about)
                    except Exception as e:
                        print(f"Error during image processing: {e}")
                        time.sleep(60*10)
                        try:
                            process_images(hit_count, pic_title, example_count, image_snapshot, html_file, title, about)
                        except Exception as e:
                            print(f"Error after retry: {e}")
                            process_misses(html_file, title, miss_count)
                            continue
                else:
                    process_misses(html_file, title, miss_count, if_statement=True)
                    continue
        except Exception as e:
            print(f"Error processing section for {html_file}: {e}")
            process_misses(html_file, title, miss_count)

    return hit_count + 1, miss_count

def main():
    hit_files = []
    miss_files = []
    seen_files = []

    try:
        # Load previous hits and misses from the JSON files
        with open(os.path.join(jsons, 'example_pictures.json'), 'r') as f:
            for line in f:
                results = dict(json.loads(line))
                file_name = results['url']
                hit_files.append(file_name)
                seen_files.append(file_name)
        
        with open(os.path.join(jsons, 'miss_examples.json'), 'r') as f:
            for line in f:
                results = dict(json.loads(line))
                file_name = results['filename']
                miss_files.append(file_name)
                seen_files.append(file_name)
    except Exception as e:
        print(f"Error reading JSON files: {e}")

    hits = set(hit_files)
    misses = set(miss_files)
    hit_count = len(hits)
    miss_count = len(misses)

    for html_file in tqdm(glob.glob('selenium_snapshots/*/*_tablecontent.html')):
        hit_count, miss_count = process_files(html_file, hit_files, miss_files, seen_files, hit_count, miss_count)

    print('Jobs done!')

if __name__ == "__main__":
    main()
