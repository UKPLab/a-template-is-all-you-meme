import ast
import clip
import json
import glob
import pickle
import PIL
import torch
import gc
import numpy as np
from collections import Counter
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from PIL import ImageFile
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
ImageFile.LOAD_TRUNCATED_IMAGES = True

class TemplateLabelCounter(object):
    def __init__(self, 
                 args,
                 dataset,                
                 model=None,
                 preprocess=None,
                 need_to_read=False):
        self.args = args
        self.label_lst, self.dataset = dataset
        self.model = model
        self.preprocess = preprocess
        self.need_to_read = need_to_read

        if not self.need_to_read:
            if not self.args.examples:
                self.get_template_embeddings()
                if self.args.combine not in ['None']:
                    about_embeddings = self.get_template_about()
                    self.template_embeddings = (self.template_embeddings, about_embeddings)
                    self.template_embeddings = self.combine_features(self.template_embeddings)
            else:
                self.get_template_example_embeddings()
                if self.args.combine not in ['None']:
                    about_embeddings = self.get_example_about()
                    self.template_embeddings = (self.template_embeddings, about_embeddings)
                    self.template_embeddings = self.combine_features(self.template_embeddings)
            if self.args.combine not in ['None']:
                self.get_combined_embeddings()   
            else:
                self.get_meme_embeddings()
        else:
            feat = self.args.feature
            if self.args.just_text:
                combine = 'just_text'
                feature_dir = f'embeddings/{feat}/{combine}/'
                print('feature_dir ', feature_dir)  
                with open('embeddings/info_dicts.pickle', 'rb') as handle:
                    self.info = pickle.load(handle)
                self.template_embeddings = np.load('{}about_embeddings.npy'.format(feature_dir))
            
            elif not self.args.just_text and self.args.combine in ['None', 'fancy', 'fusion', 'concatenate']:   
                combine = self.args.combine
                feature_dir = f'embeddings/{feat}/{combine}/'
                print('feature_dir ', feature_dir)
                if not self.args.examples:
                    with open('{}info_dicts.pickle'.format(feature_dir), 'rb') as handle:
                        self.info = pickle.load(handle)
                    self.template_embeddings = np.load('{}template_embeddings.npy'.format(feature_dir))
                else:
                    with open('{}ex_info_dicts.pickle'.format(feature_dir), 'rb') as handle:
                        self.info = pickle.load(handle)
                    with open('{}idx_lst.pickle'.format(feature_dir), 'rb') as handle:
                        self.idx_lst = pickle.load(handle)
                    self.template_embeddings = np.load('{}ex_template_embeddings.npy'.format(feature_dir))
            
            elif not self.args.just_text and self.args.combine in ['latefusion']:
                about_dir = f'embeddings/{feat}/just_text/'
                self.about_embeddings = np.load('{}about_embeddings.npy'.format(about_dir))

                template_dir = f'embeddings/{feat}/None/'
                self.template_embeddings = np.load('{}template_embeddings.npy'.format(template_dir))
                with open('embeddings/info_dicts.pickle', 'rb') as handle:
                    self.info = pickle.load(handle)
                
            if args.just_text:
                ds_dir = f'{feature_dir}{self.args.dataset}/'
                self.train_embeddings = np.load('{}train_ocr_embeddings.npy'.format(ds_dir))
                self.test_embeddings = np.load('{}test_ocr_embeddings.npy'.format(ds_dir))
            elif not self.args.just_text and self.args.combine in ['None', 'fancy', 'fusion', 'concatenate']:
                ds_dir = f'{feature_dir}{self.args.dataset}/' 
                self.train_embeddings = np.load('{}train_embeddings.npy'.format(ds_dir))
                self.test_embeddings = np.load('{}test_embeddings.npy'.format(ds_dir))
            elif not self.args.just_text and self.args.combine in ['latefusion']:
                ocr_dir = f'{about_dir}{self.args.dataset}/'
                self.train_text_embeddings = np.load('{}train_ocr_embeddings.npy'.format(ocr_dir))
                self.test_text_embeddings = np.load('{}test_ocr_embeddings.npy'.format(ocr_dir))
                meme_dir = f'{template_dir}{self.args.dataset}/'
                self.train_meme_embeddings = np.load('{}train_embeddings.npy'.format(meme_dir))
                self.test_meme_embeddings = np.load('{}test_embeddings.npy'.format(meme_dir))
        if not self.args.examples:
            self.template_idx_lst()

        
    def run(self):
        if self.args.vote_type in ['label']:
            if self.args.combine in ['latefusion']:
                self.train_late_fusion()
                self.test_late_fusion()
            else:
                self.label_train()
                self.label_test()
        else:
            self.template_train()
            self.template_test()     

    def label_fix(self, lst):
        return [int(i) for i in lst]
    
    def template_idx_lst(self):
        self.idx_lst = []
        for template_info in self.info:
            for template in template_info.keys():
                template = template_info[template]
                template_name = template['original_info'][0]['title']
            self.idx_lst.append(template_name)

    
    def get_majority(self, lst):
        lst = [self.label_fix(lab) for lab in lst]
        count = Counter([str(lab) for lab in lst])
        maj_item = ast.literal_eval(count.most_common()[0][0])
        if sum(maj_item) > 0:
            return maj_item
        else:
            try:
                return ast.literal_eval(count.most_common()[1][0])
            except:
                return maj_item
            
    def template_vote(self, indices):
        voting_results = []
        for idx_row in indices:
            candidates = [self.idx_lst[idx] for idx in idx_row]
            winner = Counter(candidates).most_common()
            if len(winner) < len(idx_row):
                voting_results.append(winner[0][0])
            else:
                voting_results.append('idk')
        return voting_results
    
    def label_vote(self, indices):
        voting_results = []
        for idx_row in indices:
            candidates = [self.idx_lst[idx] for idx in idx_row]
            candidate_labs = []
            for candidate in candidates:
                try:
                    lab = self.maj_dict[candidate]
                except KeyError:
                    lab = self.maj
                candidate_labs.append(lab)
    
            count = Counter([str(lab) for lab in candidate_labs])
            maj_item = count.most_common()
            if len(maj_item) < len(idx_row):
                voting_results.append(ast.literal_eval(maj_item[0][0]))
            else:
                voting_results.append(self.maj)
        return voting_results


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
        
        self.template_embeddings = self.clip_features(template_images)

    
    def clean_up(self):
        torch.cuda.empty_cache()
        gc.collect()
    
    def torch_dataset(self, X):
        dataset = TensorDataset(X)
        return DataLoader(dataset, batch_size=64)        
    
    def clip_features(self, image_lst):
        self.clean_up()
        tensor = torch.tensor(np.stack(image_lst)).cuda()
        dataset = self.torch_dataset(tensor)
        if len(dataset) == 1:
            with torch.no_grad():
                for x in dataset:
                    embeddings = np.array(self.model.encode_image(x[0]).float().cpu())#.cpu()
            self.clean_up()
            return embeddings
        else:
            embeddings = np.zeros(shape=(len(image_lst), self.model.ln_final.normalized_shape[0]))
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
                self.clean_up()
            return embeddings
    
    def get_template_about(self):
        about = []
        for template_info in tqdm(self.info):
            for template in template_info.keys():
                template = template_info[template]
                template_about = template['original_info'][0]['about']
                about.append(template_about)
    
        return self.clip_text(about)

    def clip_text(self, text_lst):
        self.clean_up()
        embeddings = np.zeros(shape=(len(text_lst), self.model.ln_final.normalized_shape[0]))
        for idx, text in tqdm(enumerate(text_lst)):
            text = clip.tokenize([text], truncate=True).cuda()
            text = self.model.encode_text(text).cpu().detach().numpy()
            embeddings[idx] = text
        self.clean_up()
        return embeddings
    
    def combine_features(self, embeddings):
        pic, text = embeddings
        
        if self.args.combine in ['fusion']:
            print('fusing')
            output = np.multiply(pic, text)
        
        elif self.args.combine in ['concatenate']:
            print('concatenating')
            output = np.concatenate((pic, text), axis=1)
        
        elif self.args.combine in ['fancy']:
            print('fancy')
            pic = normalize(pic, axis=1, norm='l2')
            text = normalize(text, axis=1, norm='l2')
            output = np.mean([pic, text], axis=0)
        
        return output
    
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

    def get_example_about(self):
        about = [None]*len(self.idx_lst)
        for template_info in tqdm(self.info):
            for template in template_info.keys():
                template_access = template_info[template]
                template_about = template_access['original_info'][0]['about']
                for idx, template_name in enumerate(self.idx_lst):
                    if template_name == template:
                        about[idx] = template_about
        assert None not in about
        return self.clip_text(about)
    
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
    
    def get_meme_embeddings(self):
        self.train_embeddings = self.ds_to_embeddings(self.dataset['train'])
        try:
            self.val_embeddings = self.ds_to_embeddings(self.dataset['validation'])
        except KeyError:
            self.val_embeddings = None
        self.test_embeddings = self.ds_to_embeddings(self.dataset['test'])
    
    def get_combined_embeddings(self):
        self.train_embeddings = self.ds_to_embeddings(self.dataset['train'])
        train_ocr = self.clip_text(self.dataset['train']['ocr_text'])
        self.train_embeddings = (self.train_embeddings, train_ocr)
        self.train_embeddings = self.combine_features(self.train_embeddings)

        try:
            self.val_embeddings = self.ds_to_embeddings(self.dataset['validation'])
            val_ocr = self.clip_text(self.dataset['validation']['ocr_text'])
            self.val_embeddings = (self.val_embeddings, val_ocr)
            self.val_embeddings = self.combine_features(self.val_embeddings)
        except KeyError:
            self.val_embeddings = None
        
        self.test_embeddings = self.ds_to_embeddings(self.dataset['test'])
        test_ocr = self.clip_text(self.dataset['test']['ocr_text'])
        self.test_embeddings = (self.test_embeddings, test_ocr)
        self.test_embeddings = self.combine_features(self.test_embeddings)
    
    def train_late_fusion(self):
        y_train = self.dataset['train']['labels']
        if self.need_to_read:
            try:
                y_train += self.dataset['validation']['labels']
            except:
                print('No validation set available.')
        else:
            pass        
        
        self.text_knn = NearestNeighbors(n_neighbors = self.args.num_neigh)
        self.text_knn.fit(self.about_embeddings)
        _, text_train_indices = self.text_knn.kneighbors(self.train_text_embeddings, return_distance=True)
        
        self.template_knn = NearestNeighbors(n_neighbors = self.args.num_neigh)
        self.template_knn.fit(self.template_embeddings)
        _, meme_train_indices = self.template_knn.kneighbors(self.train_meme_embeddings, return_distance=True)

        template_dict = dict()
        text_dict = dict()
        self.maj = self.get_majority(y_train)    
        for place, (template_row, text_idx_row) in tqdm(enumerate(zip(meme_train_indices, text_train_indices))):
            label = y_train[place]
            for template_idx, text_idx in zip(template_row, text_idx_row):

                template_name = self.idx_lst[template_idx]
                text_name = self.idx_lst[text_idx]
        
                if template_name in template_dict:
                    template_dict[template_name].append(label)
                else:
                    template_dict[template_name] = [label]
                
                if text_name in text_dict:
                    text_dict[text_name].append(label)
                else:
                    text_dict[text_name] = [label]

        
        self.maj_dict = dict()
        for template_title in self.idx_lst:
            if template_title in text_dict:
                text_labels = text_dict[template_title]
            else:
                text_labels = [self.maj]
            
            if template_title in template_dict:
                template_labels = template_dict[template_title]
            else:
                template_labels = [self.maj]
            title_labels = []
         
            for lab in text_labels:
                title_labels.append(lab)
          
            for lab in template_labels:
                title_labels.append(lab)

        
            self.maj_dict[template_title] = self.get_majority(title_labels)
             
        print("TRAINING")
        self.gimme_f1s(y_train, y_train)
        train_indices = np.concatenate((text_train_indices, meme_train_indices), axis=1)
        y_pred_train = self.label_vote(train_indices)
     
        print()
        print("TRAINING PREDICTION")
        self.gimme_f1s(y_train, y_pred_train)          
        print("-------------------------------------")
        print()
    
    def test_late_fusion(self):
        y_test = self.dataset['test']['labels']
        print("TESTING")
        self.gimme_f1s(y_test, y_test)
        print()
        _, text_test_indices = self.text_knn.kneighbors(self.test_text_embeddings, return_distance=True)
        _, meme_test_indices = self.template_knn.kneighbors(self.test_meme_embeddings, return_distance=True)
        test_indices = np.concatenate((text_test_indices, meme_test_indices), axis=1) 
        y_pred_test = self.label_vote(test_indices)
        print("TESTING PREDICTION")
        self.gimme_f1s(y_test, y_pred_test)
        
    def label_train(self):
        y_train = self.dataset['train']['labels']
        if self.need_to_read:
            try:
                y_train += self.dataset['validation']['labels']
            except:
                print('No validation set available.')
        else:
            try:                
                self.val_embeddings.shape
                self.train_embeddings = np.concatenate((self.train_embeddings, self.val_embeddings), axis=0)
                y_train += self.dataset['validation']['labels']
            except:
                print('No validation set available.')
        print(self.train_embeddings.shape)

        print(self.template_embeddings.shape)
        self.knn = NearestNeighbors(n_neighbors = self.args.num_neigh)
        self.knn.fit(self.template_embeddings)
        _, train_indices = self.knn.kneighbors(self.train_embeddings, return_distance=True)

        template_dict = dict()
        for place, idx_row in tqdm(enumerate(train_indices)):
            for idx in idx_row:
                template_name = self.idx_lst[idx]
                label = y_train[place]
                if template_name in template_dict:
                    template_dict[template_name].append(label)
                else:
                    template_dict[template_name] = [label]

        self.maj_dict = dict()
        for template_name, labels in template_dict.items():
            self.maj_dict[template_name] = self.get_majority(labels)
        
        self.maj = self.get_majority(y_train)
        
        print("TRAINING")
        self.gimme_f1s(y_train, y_train)
    
        y_pred_train = self.label_vote(train_indices)
     
        print()
        print("TRAINING PREDICTION")
        self.gimme_f1s(y_train, y_pred_train)          
        print("-------------------------------------")
        print()
    
    def label_test(self):
        y_test = self.dataset['test']['labels']
        print("TESTING")
        self.gimme_f1s(y_test, y_test)
        print()
        _, test_indices = self.knn.kneighbors(self.test_embeddings, return_distance=True)
        y_pred_test = self.label_vote(test_indices)
        print("TESTING PREDICTION")
        self.gimme_f1s(y_test, y_pred_test)
    
    def _template_train(self):
        y_train = self.dataset['train']['labels']
        if self.need_to_read:
            try:
                y_train += self.dataset['validation']['labels']
            except:
                print('No validation set available.')
        else:
            try:                
                self.val_embeddings.shape
                self.train_embeddings = np.concatenate((self.train_embeddings, self.val_embeddings), axis=0)
                y_train += self.dataset['validation']['labels']
            except:
                print('No validation set available.')
        print(self.train_embeddings.shape)

        print(self.template_embeddings.shape)
        self.knn = NearestNeighbors(n_neighbors = self.args.num_neigh)
        self.knn.fit(self.template_embeddings)
        _, train_indices = self.knn.kneighbors(self.train_embeddings, return_distance=True)

        template_dict = dict()
        if self.args.num_neigh > 1:
            train_votes = self.template_vote(train_indices)
            for template_name, label in tqdm(zip(train_votes, y_train)):
                if template_name in template_dict:
                    template_dict[template_name].append(label)
                else:
                    template_dict[template_name] = [label]
        else:
            train_indices = train_indices[:, 0] 
            for place, idx in tqdm(enumerate(train_indices)):
                template_name = self.idx_lst[idx]
                label = y_train[place]
                if template_name in template_dict:
                    template_dict[template_name].append(label)
                else:
                    template_dict[template_name] = [label]

        self.maj_dict = dict()
        for template_name, labels in template_dict.items():
            self.maj_dict[template_name] = self.get_majority(labels)

        self.maj = self.get_majority(y_train)

        print("TRAINING")
        self.gimme_f1s(y_train, y_train)

    
        if self.args.num_neigh > 1:
            y_pred_train = self.votes_to_pred(train_indices)
        else:
            y_pred_train = self.maj_to_pred(train_indices)
        print()
        print("TRAINING PREDICTION")
        self.gimme_f1s(y_train, y_pred_train)          
        print("-------------------------------------")
        print()
    
    def _template_test(self):
        y_test = self.dataset['test']['labels']
        print("TESTING")
        self.gimme_f1s(y_test, y_test)
        print()
        _, test_indices = self.knn.kneighbors(self.test_embeddings, return_distance=True)

        
        if self.args.num_neigh > 1:
            y_pred_test = self.votes_to_pred(test_indices)
        else:
            test_indices = test_indices[:, 0]
            y_pred_test = self.maj_to_pred(test_indices)
        print("TESTING PREDICTION")
        self.gimme_f1s(y_test, y_pred_test)
    
    def template_train(self):
        y_train = self.dataset['train']['labels']
        if self.need_to_read:
            try:
                y_train += self.dataset['validation']['labels']
            except:
                print('No validation set available.')
        else:
            try:                
                self.val_embeddings.shape
                self.train_embeddings = np.concatenate((self.train_embeddings, self.val_embeddings), axis=0)
                y_train += self.dataset['validation']['labels']
            except:
                print('No validation set available.')
        print(self.train_embeddings.shape)

        print(self.template_embeddings.shape)
        self.knn = NearestNeighbors(n_neighbors = self.args.num_neigh)
        self.knn.fit(self.template_embeddings)
        _, train_indices = self.knn.kneighbors(self.train_embeddings, return_distance=True)

        template_dict = dict()
        if self.args.num_neigh > 1:
            train_votes = self.template_vote(train_indices)
            for template_name, label in tqdm(zip(train_votes, y_train)):
                if template_name in template_dict:
                    template_dict[template_name].append(label)
                else:
                    template_dict[template_name] = [label]
        else:
            train_indices = train_indices[:, 0] 
            for place, idx in tqdm(enumerate(train_indices)):
                template_name = self.idx_lst[idx]
                label = y_train[place]
                if template_name in template_dict:
                    template_dict[template_name].append(label)
                else:
                    template_dict[template_name] = [label]

        self.maj_dict = dict()
        for template_name, labels in template_dict.items():
            self.maj_dict[template_name] = self.get_majority(labels)

        self.maj = self.get_majority(y_train)

        print("TRAINING")
        self.gimme_f1s(y_train, y_train)

    
        if self.args.num_neigh > 1:
            y_pred_train = self.votes_to_pred(train_indices)
        else:
            y_pred_train = self.maj_to_pred(train_indices)
        print()
        print("TRAINING PREDICTION")
        self.gimme_f1s(y_train, y_pred_train)          
        print("-------------------------------------")
        print()
    
    def template_test(self):
        y_test = self.dataset['test']['labels']
        print("TESTING")
        self.gimme_f1s(y_test, y_test)
        print()
        _, test_indices = self.knn.kneighbors(self.test_embeddings, return_distance=True)

        
        if self.args.num_neigh > 1:
            y_pred_test = self.votes_to_pred(test_indices)
        else:
            test_indices = test_indices[:, 0]
            y_pred_test = self.maj_to_pred(test_indices)
        print("TESTING PREDICTION")
        self.gimme_f1s(y_test, y_pred_test)
        

    def maj_to_pred(self, indices):
        y_pred = []
        just_maj = 0    
        for idx in tqdm(indices):  
            template_name = self.idx_lst[idx]
            try:
                prediction = self.maj_dict[template_name]
        
            except KeyError:
                just_maj+=1
                prediction = self.maj  
        
            y_pred.append(prediction)
        print(f'just maj count: {just_maj}')
        return y_pred
    
    def votes_to_pred(self, indices):
        idk_count = 0
        pred = []
        votes = self.template_vote(indices)
        for template in votes:
            if template == 'idk':
                pred.append(self.maj)
                idk_count+=1
            else:
                try:
                    pred.append(self.maj_dict[template])
                except KeyError:
                    pred.append(self.maj)
        print(f'idks: {idk_count}')
        return pred
    
    def gimme_f1s(self, y_true, y_pred):
        print('zero')
        print(classification_report(y_true=y_true, y_pred=y_pred, zero_division=0))
        print('one')
        print(classification_report(y_true=y_true, y_pred=y_pred, zero_division=1))
        print()
        print('zero')
        f1s = ['micro', 'macro', 'weighted', 'samples']
        for score in f1s:
            f1 = f1_score(y_true=y_true, y_pred=y_pred, zero_division=0, average=score)*100
            print(score)
            print(f1)
            print()
        print('one')
        for score in f1s:
            f1 = f1_score(y_true=y_true, y_pred=y_pred, zero_division=1, average=score)*100
            print(score)
            print(f1)
            print()