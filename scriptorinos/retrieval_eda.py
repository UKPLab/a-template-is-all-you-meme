import pickle
import argparse
import os
import shutil
import random
from random import randrange
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from memetils import load_dataset, str2bool
random.seed(42)
parser = argparse.ArgumentParser(description='retrieval eda')
parser.add_argument('--template_path', action='store', type=str, dest='template_path', default='data/template_examples/jsons/template_info.json')
parser.add_argument('--dataset', action="store", type=str, dest='dataset', default='figmemes')
parser.add_argument('--data_root', action="store", type=str, dest='data_root', default='data/annotations')
parser.add_argument('--num_neigh', action="store", type=int, dest='num_neigh', default=1)
parser.add_argument('--split', action="store", type=str, dest='split', default='standard')
parser.add_argument('--all_feature_type', action="store", type=str, dest='all_feature_type', default='')
parser.add_argument('--include_examples', action="store", type=str2bool, dest='examples', default=False)
parser.add_argument('--feature_extraction', action="store", type=str, dest='feature', default='pixel')
parser.add_argument('--meme_size', action="store", type=int, dest='meme_size', default=64)
parser.add_argument('--task', action="store", type=int, dest='task', default=1)
parser.add_argument('--combine', action="store", type=str, dest='combine', default='None')

args = parser.parse_args()

dataset = load_dataset(args)
label_lst, ds = dataset
ds_dict = {'figmemes': [7],
           'memotion3': [4],
           'memex': [2],
           'multioff': [2],
           'mami': [4]}
k = ds_dict[args.dataset][args.task-1]

feat = args.feature
combine = args.combine
feature_dir = f'embeddings/{feat}/{combine}/'
print('feature_dir ', feature_dir)
with open('{}info_dicts.pickle'.format(feature_dir), 'rb') as handle:
    info_dicts = pickle.load(handle)

if not args.examples:
    #with open('{}info_dicts.pickle'.format(feature_dir), 'rb') as handle:
    #    info_dicts = pickle.load(handle)
    template_embeddings = np.load('{}template_embeddings.npy'.format(feature_dir))
else:
    with open('{}ex_info_dicts.pickle'.format(feature_dir), 'rb') as handle:
        ex_info_dicts = pickle.load(handle)
    with open('{}idx_lst.pickle'.format(feature_dir), 'rb') as handle:
        idx_lst = pickle.load(handle)
    template_embeddings = np.load('{}ex_template_embeddings.npy'.format(feature_dir))

ds_dir = f'{feature_dir}{args.dataset}/'
print('ds dir', ds_dir)
train_embeddings = np.load('{}train_embeddings.npy'.format(ds_dir))
test_embeddings = np.load('{}test_embeddings.npy'.format(ds_dir))
dataset_embeddings = np.concatenate((train_embeddings, test_embeddings), axis=0)

try:
    memes = ds['train']['img_path'] + ds['validation']['img_path']
    
except:
    memes = ds['train']['img_path']

if not args.examples:
    outdir_ds = 'retrieval_images/{}/{}/{}/template/'.format(feat, combine, args.dataset)
else:
    outdir_ds = 'retrieval_images/{}/{}/{}/example/'.format(feat, combine, args.dataset)
if not os.path.exists(outdir_ds):
    os.makedirs(outdir_ds)

knn = NearestNeighbors(n_neighbors=1)
knn.fit(template_embeddings)
dists, indices = knn.kneighbors(dataset_embeddings, return_distance=True)
meme_idxs = [i for i in range(len(memes))]
dists, indices, meme_idxs = zip(*sorted(zip(dists, indices, meme_idxs)))
print(k)
print()
indices = indices[:500]
meme_idxs = meme_idxs[:500]
random_indices = set()
while len(random_indices) < k:
    random_indices.add(randrange(len(indices)))


for r_idx in random_indices:
    template_idx = indices[r_idx][0]
    meme_idx = meme_idxs[r_idx]
    print(template_idx)
    if args.examples:
        t_name = idx_lst[template_idx]
        for info in info_dicts:
       
            for template in info.keys():
                template = info[template]
                template_name = template['original_info'][0]['title']
                if t_name == template_name:
                    pic_file = template["out_paths"][0].split('/')[-1]
                    pic_file = f'{template_idx}_{pic_file}'
                    pic_path = 'data/' +  template["out_paths"][0]
                    shutil.copy2(pic_path, outdir_ds+pic_file) # complete target filename given
  
    else:
        info = info_dicts[template_idx]
        for template in info.keys():
            template = info[template]
            pic_file = template["out_paths"][0].split('/')[-1]
            pic_file = f'{template_idx}_{pic_file}'
            pic_path = 'data/' +  template["out_paths"][0]
            shutil.copy2(pic_path, outdir_ds+pic_file) # complete target filename given
    
    try:
        pic_file = memes[meme_idx].split('/')[-1]
        pic_file = f'{args.dataset}_{template_idx}_{pic_file}'
        pic_path = memes[meme_idx]
        shutil.copy2(pic_path, outdir_ds+pic_file)
    except:
        pic_file = memes[meme_idx].split('/')[-1]
        pic_file = f'{args.dataset}_{template_idx}_{pic_file}'
        pic_path = 'data/'+ memes[meme_idx]
        shutil.copy2(pic_path, outdir_ds+pic_file)


print('jobs done')