import json
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from skimage.transform import resize
double_check = 'double_check'
if not os.path.exists(double_check):
    os.makedirs(double_check)

seen_urls = []
keep_dicts = []
found_meme_path = 'jsons/meme_pictures.json'
with open(found_meme_path, 'r') as f:
    for line in f:
        results = dict(json.loads(line))
        url = results['url']
        url = url.split('/')
        url = "/".join(url[5:])
        if url not in seen_urls:
            seen_urls.append(url)
            keep_dicts.append(results)

info = []
problems = []
meme_size = 64
templates = np.zeros(shape=(len(keep_dicts), meme_size, meme_size, 3))
print(templates.shape)
for idx, dictionary in enumerate(keep_dicts):
    im = cv2.imread(dictionary['pic_title'])
    if im is not None:            
        try:
            im = cv2.resize(im, (meme_size, meme_size), interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(e)        
            print('found a gif!')
            gif = cv2.VideoCapture(dictionary['pic_title'])
            ret, im = gif.read()  # ret=True if it finds a frame else False.
            frames = [im]
            while ret:
                ret, im = gif.read()
                frames.append(im)
            print('this many frames')
            print(len(frames))
            print(dictionary)
            frames = [np.array(frame) for frame in frames]
            if len(frames) > 1:
                im = frames[1]
                im = resize(im, (meme_size, meme_size))
            else:
                try:
                    im = resize(frames[0], (meme_size, meme_size))
                except Exception as e:
                    print(e)
                    print('what')
                    print(dictionary)
                    problems.append(dictionary)
    else:
        print('no image!')
        try:
            im = plt.imread(dictionary['pic_title'])
            print(im.shape)
            im = im[:, :, 0:3]
            im = resize(im, (meme_size, meme_size))
            outdir = dictionary['pic_title'].split('/')[-1]
            outdir = double_check + '/' + outdir
            plt.imsave(outdir, im)


        except Exception as e:
            print(e)
            print(dictionary)
            problems.append(dictionary)
            continue
    keep_tup = (im, dictionary)
    info.append(keep_tup)
    templates[idx] = im       

print(templates.shape)
templates_flatten = templates.reshape(templates.shape[0],-1)
print(templates_flatten.shape)

print('jobs done!')
