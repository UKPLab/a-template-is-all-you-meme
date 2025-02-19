import json
import gc
import glob
import os
import pickle
import PIL
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from datasets import Dataset, DatasetDict
from PIL import ImageFile
from sklearn.neighbors import NearestNeighbors
from memetils import seed_everything
ImageFile.LOAD_TRUNCATED_IMAGES = True

class TSplit(object):
    def __init__(self, 
                 args,
                 dataset,                
                 model=None,
                 preprocess=None):
        self.args = args
        self.dataset = dataset
        self.model = model
        self.preprocess = preprocess
        if not self.args.need_to_read:
            self.get_template_example_embeddings()
            self.template_idx_lst() 
            self.get_template_embeddings                  
        else:          
            feature_dir = f'embeddings/{self.args.feature}/None/'
            with open('{}ex_info_dicts.pickle'.format(feature_dir), 'rb') as handle:
                self.info = pickle.load(handle)
            with open('{}idx_lst.pickle'.format(feature_dir), 'rb') as handle:
                self.idx_lst = pickle.load(handle)
            self.template_embeddings = np.load('{}ex_template_embeddings.npy'.format(feature_dir))
            print(self.template_embeddings.shape)
            self.just_template_embeddings = np.load('{}template_embeddings.npy'.format(feature_dir))
            print(self.just_template_embeddings.shape)

        
        self.set_template_thresholds()
        if not self.args.sample_train:
            self.get_meme_embeddings()
        else:
            self.resample_training()
        self.eda()
        self.meme_thresholds()
    
    def set_template_thresholds(self):
        thresholds = dict()

        for template_name, template_embed in tqdm(zip(self.idx_lst, self.template_embeddings)):
            if template_name not in thresholds:
                thresholds[template_name] = [template_embed]
            else:
                ref = thresholds[template_name][0]
                dist = np.linalg.norm(ref-template_embed)
                thresholds[template_name].append(dist)
        
        self.threshold_dict = dict()
        acceptables = []
        for template_name, dist_lst in thresholds.items():
            if len(dist_lst) == 1:
                self.threshold_dict[template_name] = None
                continue
            else:
                dist_lst = dist_lst[1:]
                if self.args.reorganize == 'max':
                    accept = max(dist_lst)
                elif self.args.reorganize == 'mean':
                    accept = np.mean(dist_lst)
                elif self.args.reorganize == 'median':
                    accept = np.median(dist_lst)
                elif self.args.reorganize == 'quantile':
                    accept = np.quantile(dist_lst, self.args.qv)
                
                self.threshold_dict[template_name] = accept
                acceptables.append(accept)
        
        if self.args.reorganize == 'max':
            acceptable = max(acceptables)
        elif self.args.reorganize == 'mean':
            acceptable = np.mean(acceptables)
        elif self.args.reorganize == 'median':
            acceptable = np.median(acceptables)
        elif self.args.reorganize == 'quantile':
            acceptable = np.quantile(acceptables, self.args.qv)
        
        for template_name, dist in self.threshold_dict.items():
            if not dist:
                self.threshold_dict[template_name] = acceptable
    
    def torch_dataset(self, X):
        dataset = TensorDataset(X)
        return DataLoader(dataset, batch_size=self.args.batch)  
    
    def template_idx_lst(self):
        self.idx_lst = []
        for template_info in self.info:
            for template in template_info.keys():
                template = template_info[template]
                template_name = template['original_info'][0]['title']
            self.idx_lst.append(template_name)
    
    def ds_to_embeddings(self, ds):
        dank_memes = []
        for dank_meme in tqdm(ds['img_path']):
            try:        
                dank_meme = self.preprocess(PIL.Image.open(dank_meme))
            except FileNotFoundError:
                dank_meme = 'data/' + dank_meme
                dank_meme = self.preprocess(PIL.Image.open(dank_meme))

            dank_memes.append(dank_meme)
        
        embeddings = self.clip_features(dank_memes)        
        return embeddings
    
    def resample_training(self):
        self.memes, self.labels, self.ocr = [], [], []
        
        self.train_embeddings = self.ds_to_embeddings(self.dataset['train'])
        self.train_size = len(self.dataset['train'])
        self.memes += self.dataset['train']['img_path']
        self.labels += self.dataset['train']['labels']
        self.ocr += self.dataset['train']['ocr_text']
        self.all_meme_embeddings = self.train_embeddings

        if 'validation' in self.dataset:
            self.val_embeddings = self.ds_to_embeddings(self.dataset['validation'])
            self.val_size = len(self.dataset['validation'])
            self.memes += self.dataset['validation']['img_path']
            self.labels += self.dataset['validation']['labels']
            self.ocr += self.dataset['validation']['ocr_text']
            self.all_meme_embeddings = np.vstack((self.all_meme_embeddings, self.val_embeddings))
        else:
            self.val_size = 0
       
        self.test_size = len(self.dataset['test'])
        self.dataset_size = self.train_size + self.val_size + self.test_size
    
    def get_meme_embeddings(self):
        self.memes, self.labels, self.ocr = [], [], []
        
        self.train_embeddings = self.ds_to_embeddings(self.dataset['train'])
        self.train_size = len(self.dataset['train'])
        self.memes += self.dataset['train']['img_path']
        self.labels += self.dataset['train']['labels']
        self.ocr += self.dataset['train']['ocr_text']
        self.all_meme_embeddings = self.train_embeddings

        if 'validation' in self.dataset:
            self.val_embeddings = self.ds_to_embeddings(self.dataset['validation'])
            self.val_size = len(self.dataset['validation'])
            self.memes += self.dataset['validation']['img_path']
            self.labels += self.dataset['validation']['labels']
            self.ocr += self.dataset['validation']['ocr_text']
            self.all_meme_embeddings = np.vstack((self.all_meme_embeddings, self.val_embeddings))
        else:
            self.val_size = 0

        self.test_embeddings = self.ds_to_embeddings(self.dataset['test'])
        self.test_size = len(self.dataset['test'])
        self.memes += self.dataset['test']['img_path']
        self.labels += self.dataset['test']['labels']
        self.ocr += self.dataset['test']['ocr_text']
        
        self.all_meme_embeddings = np.vstack((self.all_meme_embeddings, self.test_embeddings))
        self.dataset_size = self.train_size + self.val_size + self.test_size
    
    def get_template_embeddings(self):
        self.info, template_images = [], []

        with open(self.args.path, 'r') as f:
            for line in tqdm(f):
                template_info = dict(json.loads(line))
                self.info.append(template_info)
                for template in template_info.keys():
                    template = template_info[template]
                    im = self.preprocess(PIL.Image.open('data/'+template["out_paths"][0]))
            
                template_images.append(im)
        
        self.just_template_embeddings = self.clip_features(template_images)

    def get_template_example_embeddings(self):
        miss_count = 0
        self.info, self.idx_lst, temps_and_examples = [], [], []
        with open(self.args.path, 'r') as f:
            for line in tqdm(f):
                template_info = dict(json.loads(line))
                self.info.append(template_info)
                for template_name in template_info.keys():
                    template = template_info[template_name]
                    template_im = self.preprocess(PIL.Image.open('data/'+template["out_paths"][0]))
                    
                    temps_and_examples.append(template_im)
                    self.idx_lst.append(template_name) 

                    example_path = 'data/' + template['out_paths'][0]
                    example_path = '/'.join(example_path.split("/")[:-1])
                    example_path += '/examples/*'
                    
                    for example in glob.glob(example_path):
                        try:
                            example = self.preprocess(PIL.Image.open(example))
                        except:
                            print('miss: {}'.format(example))
                            miss_count+=1
                            continue

                        temps_and_examples.append(example)
                        self.idx_lst.append(template_name)
        print('total misses : {}'.format(miss_count))
        self.template_embeddings = self.clip_features(temps_and_examples)

    def clip_features(self, image_lst):
        tensor = torch.tensor(np.stack(image_lst)).cuda()
        dataset = self.torch_dataset(tensor)
        if len(dataset) == 1:
            with torch.no_grad():
                for x in dataset:
                    embeddings = np.array(self.model.encode_image(x[0]).float().cpu())#.cpu()
            gc.collect()
            torch.cuda.empty_cache()
            return embeddings
        else:
            embeddings = np.zeros(shape=(len(image_lst), self.model.ln_final.normalized_shape[0]))
            stop = 0
            for idx, x in tqdm(enumerate(dataset)):
                with torch.no_grad():
                    image_features = np.array(self.model.encode_image(x[0]).float().cpu())#.cpu()
                
                rows = image_features.shape[0]
                if idx != len(dataset)-1:
                    start = (idx * rows)
                    stop = (idx+1) * rows    
            
                else:
                    start = stop
                    stop = stop + rows

                embeddings[start:stop, :] = image_features
            gc.collect()
            torch.cuda.empty_cache()
            return embeddings
    def eda(self):
        self.knn = NearestNeighbors(n_neighbors = 1)
        self.knn.fit(self.just_template_embeddings)
        eda_dists = []
        self.meme_dists = []
        indices = self.knn.kneighbors(self.all_meme_embeddings, return_distance=False)
        none_count = 0
        for count, (meme, idx) in tqdm(enumerate(zip(self.all_meme_embeddings, indices))):
            idx = idx[0]
            template = self.just_template_embeddings[idx]
            dist = np.linalg.norm(meme-template)
            self.meme_dists.append(dist)
            eda_dists.append(dist)
        outdir = f'base_eda/{self.args.dataset}/'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        with open(f'{outdir}eda_dists.pkl', 'wb') as f:
            pickle.dump(eda_dists, f)
    
    def meme_thresholds(self):
        self.detected_templates = [None] * self.dataset_size
        self.knn = NearestNeighbors(n_neighbors = 1)
        self.knn.fit(self.template_embeddings)
        eda_dists = []
        self.meme_dists = []
        indices = self.knn.kneighbors(self.all_meme_embeddings, return_distance=False)
        none_count = 0
        for count, (meme, idx) in tqdm(enumerate(zip(self.all_meme_embeddings, indices))):
            idx = idx[0]
            template = self.template_embeddings[idx]
            template_name = self.idx_lst[idx]
            dist = np.linalg.norm(meme-template)
            if dist <= self.threshold_dict[template_name]:
                self.detected_templates[count] = template_name
            else:
                none_count+=1
                self.detected_templates[count] = f'none_{none_count}'
            
            self.meme_dists.append(dist)
            eda_dists.append(dist)
        outdir = f'eda/{self.args.dataset}/'
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        with open(f'{outdir}eda_dists.pkl', 'wb') as f:
            pickle.dump(eda_dists, f)
    
    def templateness_index(self):
        knn = NearestNeighbors(n_neighbors = 1)
        knn.fit(self.just_template_embeddings)
        meme_dists = []
        indices = knn.kneighbors(self.all_meme_embeddings, return_distance=False)
        print(self.all_meme_embeddings.shape)
        print(len(indices))
        for count, (meme, idx) in tqdm(enumerate(zip(self.all_meme_embeddings, indices))):
            idx = idx[0]
            template = self.just_template_embeddings[idx]
           
            dist = np.linalg.norm(meme-template)
            
            meme_dists.append(dist)
        
        index = sum(meme_dists)/self.dataset_size

        return index
    
    def resplit(self):
        test_resplit = {"img_path": [], "ocr_text": [], "labels": []}
        train_resplit = {"img_path": [], "ocr_text": [], "labels": []}
        distinct_templates = sorted(list(set(self.detected_templates)))
        test_split_num = int((self.test_size/self.dataset_size) * len(distinct_templates))
        seed_everything(self.args.seed)
        random.shuffle(distinct_templates)
        if self.args.sample_tsplit:
            discard_resplit = {"img_path": [], "ocr_text": [], "labels": []}
            encoders = ['ViT-L/14@336px', 'ViT-B/32', 'ViT-B/16']
            assert self.args.feature in encoders
            if self.args.feature == encoders[0]:
                downsample_dict = {'multioff': {'train': 381, 'val': 96},
                                   'memotion3': {'train': 4674, 'val':1169},
                                   'figmemes': {'train':2333, 'val':260},
                                   'mami': {'train':7353, 'val':1839}}
            elif self.args.feature == encoders[1]:
                downsample_dict = {'multioff': {'train':354, 'val': 90},
                                   'memotion3': {'train':4723, 'val':1181},
                                   'figmemes': {'train':2327, 'val':260},
                                   'mami': {'train':7295, 'val':1824}}
            elif self.args.feature == encoders[2]:
                downsample_dict = {'multioff': {'train':367, 'val':93},
                                  'memotion3': {'train':4930, 'val':1233},
                                  'figmemes': {'train': 2293, 'val':256},
                                  'mami': {'train':7213, 'val':1804}}
            
            train_samp_size = downsample_dict[self.args.dataset]['train']
            val_samp_size = downsample_dict[self.args.dataset]['val']
            train_ratio = train_samp_size/self.train_size
            assert 0 < train_ratio < 1
            if self.val_size > 0:
                val_ratio = val_samp_size/self.val_size
                assert 0 < val_ratio < 1
                samp_val_size = val_ratio * self.val_size
                samp_val_ratio = samp_val_size/self.dataset_size
            else:
                val_ratio = None
            
                
        
        test_templates = distinct_templates[-test_split_num:]
        train_templates = distinct_templates[:-test_split_num]
        if self.args.sample_tsplit:
            cutoff = int(len(train_templates) * train_ratio)
            discard_templates = train_templates[cutoff:]
            train_templates = train_templates[:cutoff]            

        for detected_template, img_path, label, ocr in tqdm(zip(self.detected_templates, 
                                                                self.memes,
                                                                self.labels,
                                                                self.ocr)):
            if detected_template in train_templates:
                train_resplit['img_path'].append(img_path)
                train_resplit['labels'].append(label)
                train_resplit['ocr_text'].append(ocr)
            
            elif detected_template in test_templates:
                test_resplit['img_path'].append(img_path)
                test_resplit['labels'].append(label)
                test_resplit['ocr_text'].append(ocr)
            
            if self.args.sample_tsplit:
                if detected_template in discard_templates:
                    discard_resplit['img_path'].append(img_path)
                    discard_resplit['labels'].append(label)
                    discard_resplit['ocr_text'].append(ocr)

        
        test_ds = Dataset.from_pandas(pd.DataFrame(test_resplit))
        split_ds = Dataset.from_pandas(pd.DataFrame(train_resplit))
        if self.args.sample_tsplit:
            discard_ds = Dataset.from_pandas(pd.DataFrame(discard_resplit))
        if self.val_size > 0 and not self.args.sample_tsplit:
            split_ds = split_ds.train_test_split(test_size=(self.val_size/self.dataset_size), seed=self.args.seed)
        
        elif self.val_size > 0 and self.args.sample_tsplit:
            split_ds = split_ds.train_test_split(test_size=samp_val_ratio, seed=self.args.seed)

        elif self.val_size == 0:
            split_ds = split_ds.train_test_split(test_size=0.2, seed=self.args.seed)
        train_ds = split_ds['train']
        val_ds = split_ds['test']
        if not self.args.sample_train:
            if self.args.sample_tsplit:
                print('SAMPLE tsplit')
                print(f'size of training data: {len(train_ds)}')
                print(f'size of validation data: {len(val_ds)}')
                print(f'size of test data: {len(test_ds)}')
                print(f'size of discard data: {len(discard_ds)}')
                print('********************************************')
                resplit_dataset = {'train': train_ds, 'validation': val_ds, 'test': test_ds, 'discard': discard_ds}
            else:
                resplit_dataset = {'train': train_ds, 'validation': val_ds, 'test': test_ds}
        else:
            resplit_dataset = {'train': train_ds, 'validation': val_ds, 'test': self.dataset['test']}
            print('RESAMPLE')
            print(f'size of training data: {len(train_ds)}')
            print(f'size of validation data: {len(val_ds)}')
            print(f'size of dummy test data: {len(test_ds)}')
            print('********************************************')
        return DatasetDict(resplit_dataset), train_templates, test_templates
            

