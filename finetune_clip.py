import argparse
import clip
import gc
import json
import os
import PIL
import pickle
import torch
import numpy as np
import torch.optim as optim
from itertools import zip_longest
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
from memetils import (load_dataset, 
                      str2bool, 
                      seed_everything, 
                      downsample_train_val, 
                      downsample_tsplit, 
                      gimme_f1s, 
                      write_checkpoint,
                      process,
                      torch_dataset,
                      logits_to_preds)
from clip_model import CLIPClf
from tsplit import TSplit

parser = argparse.ArgumentParser(description='tsplit')
parser.add_argument('--dataset', action="store", type=str, dest='dataset', default='multioff')
parser.add_argument('--data_root', action="store", type=str, dest='data_root', default='data/MultiOFF_Dataset')
parser.add_argument('--split', action="store", type=str, dest='split', default='standard')
parser.add_argument('--all_feature_type', action="store", type=str, dest='all_feature_type', default='')
parser.add_argument('--feature_extraction', action="store", type=str, dest='feature', default='ViT-L/14@336px')
parser.add_argument('--task', action="store", type=int, dest='task', default=1)
parser.add_argument('--reorganize', action="store", type=str, dest='reorganize', default='original')
parser.add_argument('--quantile_value', action="store", type=float, dest='qv', default=0.25)#percentile in the paper
parser.add_argument('--batch_size', action="store", type=int, dest='batch', default=16)
parser.add_argument('--epochs', action="store", type=int, dest='epoch', default=20)
parser.add_argument('--seed', action="store", type=int, dest='seed', default=0)
parser.add_argument('--need_to_read', action="store", type=str2bool, dest='need_to_read', default='True')#if you write the embeddings to disk, it will speed things up considerably.
parser.add_argument('--sample_train', action="store", type=str2bool, dest='sample_train', default='False')#Downsample TSplit/CLIP Baseline (Table 3)
parser.add_argument('--random_downsample_tsplit', action="store", type=str2bool, dest='random_downsample_tsplit', default='False')#randomly downsample after TSplitting entire dataset
parser.add_argument('--sample_tsplit', action="store", type=str2bool, dest='sample_tsplit', default='False')#TSplit downsampling on entire dataset
parser.add_argument('--overfit', action="store", type=str2bool, dest='overfit', default='False')#do inference only with model trained for arg.epochs number of epochs

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9} GB")
else:
    print("No GPU available")

args = parser.parse_args()
seed_everything(args.seed)

if args.reorganize in ['original', 'baseline']:
    label_names, dataset = load_dataset(args)
    if 'validation' not in dataset:
        split_ds = dataset['train'].train_test_split(test_size=0.2, seed=args.seed)
        train_ds = split_ds['train']
        val_ds = split_ds['test']
    else:
        train_ds = dataset['train']
        val_ds = dataset['validation']
    test_ds = dataset['test']
    if args.reorganize in ['baseline']:
        print('resampling!!')
        train_ds, val_ds = downsample_train_val(train_ds, val_ds, args)    
        print(f'resample training split size: {len(train_ds)}')
        print()
        print(f'resample validation split size: {len(val_ds)}')
        print()
        test_size = len(test_ds)
        print(f'test split size: {test_size}')
        print()
else:
    label_names, dataset = load_dataset(args)
    model, preprocess = clip.load(args.feature, download_root='models/clip_models/')
    model.cuda()
    t_split = TSplit(args, dataset, model, preprocess)
    dataset, train_templates, test_templates = t_split.resplit()
    print('tsplitting, right?')
    train_ds = dataset['train']
    val_ds = dataset['validation']
    test_ds = dataset['test']
    if args.random_downsample_tsplit:
        train_ds, val_ds = downsample_tsplit(train_ds, val_ds, args)
        print('sizes of randomly downsampling tsplit:')
        r_train_size = len(train_ds)
        r_val_size = len(val_ds)
        r_test_size = len(test_ds)
        print(f'train: {r_train_size}')
        print()
        print(f'val: {r_val_size}')
        print()
        print(f'test: {r_test_size}')

train_size = len(train_ds)
val_size = len(val_ds)
test_size = len(test_ds)
print(f'training split size: {train_size}')
print()
print(f'validation split size: {val_size}')
print()
print(f'test split size: {test_size}')
print()
  

device = torch.device('cuda:0')
#special settings for fine-tuning clip (https://github.com/openai/CLIP/issues/40)
model, preprocess = clip.load(args.feature, download_root='models/clip_models/', jit=False)
model.float().to(device)

clf = CLIPClf(model, len(label_names))
clf = clf.to(device)


criterion = BCEWithLogitsLoss()
optimizer = optim.AdamW(clf.parameters(), lr=1e-5)

train_memes, train_ocr = process(train_ds['img_path'], train_ds['ocr_text'], preprocess)
train_memes = torch_dataset(train_memes, train_ds['labels'], args)
train_ocr = torch_dataset(train_ocr, train_ds['labels'], args)

val_memes, val_ocr = process(val_ds['img_path'], val_ds['ocr_text'], preprocess)
val_memes = torch_dataset(val_memes, val_ds['labels'], args)
val_ocr = torch_dataset(val_ocr, val_ds['labels'], args)
val_dict = dict()

for epoch in range(args.epoch):
    seed_everything(args.seed)
    val_preds = np.zeros(shape=(len(val_ds), len(label_names)))
    train_preds = np.zeros(shape=(len(train_ds), len(label_names)))
    train_cost = 0.0
    for batch_memes, batch_ocrs in tqdm(zip_longest(train_memes, train_ocr, fillvalue=None)):
        if batch_memes is None or batch_ocrs is None:
            continue  # Skip incomplete batches
        batch_memes, batch_y = batch_memes
        batch_ocrs, _ = batch_ocrs
        output = clf(batch_memes.to(device), batch_ocrs.to(device))
        loss = criterion(output, batch_y.to(device))
        train_cost += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Free up GPU memory
        del output, loss, batch_memes, batch_ocrs, batch_y
        torch.cuda.empty_cache()

    print('-----------------------')
    print(train_cost/len(train_memes))
    

    save_folder = write_checkpoint(clf, epoch, args)
    print(f'model written to {save_folder}')

    print('running training prediction')
    train_sanity_f1 = gimme_f1s(train_ds['labels'], train_ds['labels'], args)
    train_preds = logits_to_preds(clf, train_preds, train_memes, train_ocr)
    train_f1 = gimme_f1s(train_ds['labels'], train_preds, args)
    print()

    print('running validation prediction')
    val_sanity_f1 = gimme_f1s(val_ds['labels'], val_ds['labels'], args)
    val_preds = logits_to_preds(clf, val_preds, val_memes, val_ocr)
    val_f1 = gimme_f1s(val_ds['labels'], val_preds, args)

    val_dict[save_folder] = val_f1
    
    print(f'end of epoch {epoch}')
    gc.collect()
    torch.cuda.empty_cache()

print('show me configurations')

for key, value in val_dict.items():
    print(key)
    print(value)
    print()


if args.overfit:
    winner = save_folder
    winning_clf = clf
else:
    winner = max(val_dict, key=val_dict.get)
    winning_clf = CLIPClf(model, len(label_names))
    winning_clf.load_state_dict(torch.load(f'{winner}model.pt'))
    winning_clf.to(device)


test_memes, test_ocr = process(test_ds['img_path'], test_ds['ocr_text'], preprocess)
test_memes = torch_dataset(test_memes, test_ds['labels'], args)
test_ocr = torch_dataset(test_ocr, test_ds['labels'], args)

print('running testing prediction')
test_preds = np.zeros(shape=(len(test_ds), len(label_names)))
test_sanity_f1 = gimme_f1s(test_ds['labels'], test_ds['labels'], args)
test_preds = logits_to_preds(winning_clf, test_preds, test_memes, test_ocr)
test_f1 = gimme_f1s(test_ds['labels'], test_preds, args)


out_json = {'train_sanity_f1':train_sanity_f1, 
            'val_sanity_f1': val_sanity_f1,
            'test_sanity_f1':test_sanity_f1,
            'train_f1': train_f1,
            'val_f1': val_f1,
            'test_f1': test_f1}
                    
out_path = f'clip_results/{args.overfit}/{args.sample_train}/{args.random_downsample_tsplit}/{args.sample_tsplit}/{args.dataset}/{args.reorganize}/{args.feature}/{args.task}/{args.seed}/'
print(f'here is out path: {out_path}')
            
if not os.path.exists(out_path):
    os.makedirs(out_path)
          
with open(f'{out_path}results.json', 'w', encoding='utf-8') as f:
    json.dump(out_json, f, ensure_ascii=False, indent=4)

print(f'winning f1 is {test_f1}')
print(f'winning config is {winner}')

print('deleting checkpoints')

for folder in val_dict.keys():
    if folder == winner:
        print(f'keeping {winner}')
    else:
        if os.path.exists(f'{folder}model.pt'):
            os.remove(f'{folder}model.pt')

print(f'winning f1 is {test_f1}')
print(f'winning config is {winner}')

print('jobs done')
