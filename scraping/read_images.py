import json
import cv2
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
meme_size = 64
double_check = 'double_check'
if not os.path.exists(double_check):
    os.makedirs(double_check)

found_meme_path = 'jsons/meme_pictures.json'
problems = []
info = []

# Initialize the image template array
templates = np.zeros(shape=(len(info), meme_size, meme_size, 3))

# Load meme data and filter duplicates
seen_urls = set()
keep_dicts = []

with open(found_meme_path, 'r') as f:
    for line in f:
        results = dict(json.loads(line))
        url = results['url'].split('/')[5:]
        if url not in seen_urls:
            seen_urls.add(url)
            keep_dicts.append(results)

def process_image(image_path, is_gif=False):
    """Process image or gif and return resized image."""
    try:
        im = cv2.imread(image_path)
        if im is None:
            raise Exception(f"Could not read image: {image_path}")

        if is_gif:
            gif = cv2.VideoCapture(image_path)
            frames = []
            ret, im = gif.read()
            while ret:
                frames.append(im)
                ret, im = gif.read()
            im = resize(frames[0], (meme_size, meme_size))  # Resize first frame
        else:
            im = cv2.resize(im, (meme_size, meme_size), interpolation=cv2.INTER_AREA)

        return im
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        return None

# Process all meme images
for idx, dictionary in tqdm(enumerate(keep_dicts), total=len(keep_dicts), desc="Processing memes"):
    image_path = dictionary['pic_title']

    # Try processing the image
    im = process_image(image_path)

    if im is None:
        problems.append(dictionary)
        continue

    # Save image if it's a valid processed image
    pic_title = re.sub(r'\W+', '', dictionary['title'])
    pic_title = f"{double_check}/{idx}_{pic_title}.jpg"

    # Resize image before saving
    try:
        im = resize(im, (meme_size, meme_size))
        plt.imsave(pic_title, im)
    except Exception as e:
        logging.error(f"Error resizing or saving image {pic_title}: {e}")
        problems.append(dictionary)
        continue

    # Store the processed image in the templates array
    templates[idx] = im
    info.append((im, dictionary))

# Flatten templates for further processing
templates_flatten = templates.reshape(templates.shape[0], -1)

logging.info('Job done!')
