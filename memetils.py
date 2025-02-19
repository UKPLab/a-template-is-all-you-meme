import csv
import clip
import os
import ast
import datasets
import gc
import h5py
import torch
import random
import pandas as pd
import numpy as np
import PIL
from PIL import ImageFile
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader, TensorDataset
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device('cuda:0')

def process(memes, ocr, preprocess):
    dank_memes, dank_ocrs = [], []
    for dank_meme, dank_ocr in tqdm(zip(memes, ocr)):
        try:        
            dank_meme = preprocess(PIL.Image.open(dank_meme))
        except FileNotFoundError:
            dank_meme = 'data/' + dank_meme
            dank_meme = preprocess(PIL.Image.open(dank_meme))           

        dank_memes.append(dank_meme)
        
        dank_ocr = dank_ocr.replace("|||", "|")
        dank_ocrs.append(dank_ocr)  
    return torch.tensor(np.stack(dank_memes)).float(), clip.tokenize(dank_ocrs, truncate=True)

def torch_dataset(X, y, args):
    dataset = TensorDataset(X, torch.Tensor(y).float())
    loader = DataLoader(dataset, batch_size=args.batch)
    return loader

def logits_to_preds(clf, matrix, split_memes, split_ocrs):
    clf.eval()
    stop = 0
    for idx, (batch_memes, batch_ocrs) in tqdm(enumerate(zip(split_memes, split_ocrs))):
        batch_memes, batch_y = batch_memes
        batch_ocrs, _ = batch_ocrs
        output = clf(batch_memes.to(device), batch_ocrs.to(device))
        output = torch.sigmoid(output).cpu()
        output = output >= 0.5
        output = np.array(output)
        rows = output.shape[0]

        if idx != len(split_memes)-1:
            start = (idx * rows)
            stop = (idx+1) * rows    
    
        else:
            start = stop
            stop = stop + rows

        matrix[start:stop, :] = output
    return matrix

def write_checkpoint(model, epoch, args):
    folder = f'clip_checkpoints/{args.overfit}/{args.sample_train}/{args.random_downsample_tsplit}/{args.sample_tsplit}/{args.dataset}/{args.reorganize}/{args.feature}/{args.task}/{args.seed}/{epoch}/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    torch.save(model.state_dict(), f'{folder}/model.pt')

    return folder

def gimme_f1s(y_true, y_pred, args):
    if args.dataset in ['figmemes', 'multioff']:
        score = 'macro'
    elif args.dataset in ['memotion3']:
        score = 'weighted'
    elif args.dataset in ['mami']:
        if args.task == 1:
            score = 'macro'
        elif args.task == 2:
            score = 'weighted'

    print(classification_report(y_true=y_true, y_pred=y_pred))
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average=score)*100
    print(f1)
    return f1

def ds_to_embeddings(ds, model, preprocess):
    dank_memes = []
    for dank_meme in tqdm(ds['img_path']):
        try:        
            dank_meme = preprocess(PIL.Image.open(dank_meme))
        except FileNotFoundError:
            dank_meme = 'data/' + dank_meme
            dank_meme = preprocess(PIL.Image.open(dank_meme))

        dank_memes.append(dank_meme)
    
    embeddings = clip_features(dank_memes, model)
    
    return embeddings

def get_meme_embeddings(dataset, model, preprocess):

    train_embeddings = ds_to_embeddings(dataset['train'], model, preprocess)
    try:
        val_embeddings = ds_to_embeddings(dataset['validation'], model, preprocess)
    except KeyError:
        val_embeddings = None
    test_embeddings = ds_to_embeddings(dataset['test'], model, preprocess)
    return train_embeddings, val_embeddings, test_embeddings 


def get_combined_embeddings(args, dataset, model, preprocess):
    train_embeddings = ds_to_embeddings(dataset['train'], model, preprocess)
    train_ocr = clip_text(dataset['train']['ocr_text'], model)
    train_embeddings = (train_embeddings, train_ocr)
    train_embeddings = combine_features(args, train_embeddings)

    try:
        val_embeddings = ds_to_embeddings(dataset['validation'], model, preprocess)
        val_ocr = clip_text(dataset['validation']['ocr_text'], model)
        val_embeddings = (val_embeddings, val_ocr)
        val_embeddings = combine_features(args, val_embeddings)
    except KeyError:
        val_embeddings = None
    
    test_embeddings = ds_to_embeddings(dataset['test'], model, preprocess)
    test_ocr = clip_text(dataset['test']['ocr_text'], model)
    test_embeddings = (test_embeddings, test_ocr)
    test_embeddings = combine_features(args, test_embeddings)
    
    return train_embeddings, val_embeddings, test_embeddings 

def clip_features(image_lst, model):
    clean_up()
    tensor = torch.tensor(np.stack(image_lst)).cuda()
    dataset = torch_dataset(tensor)
    if len(dataset) == 1:
        with torch.no_grad():
            for x in dataset:
                embeddings = np.array(model.encode_image(x[0]).float().cpu())#.cpu()
        clean_up()
        return embeddings
    else:
        embeddings = np.zeros(shape=(len(image_lst), model.ln_final.normalized_shape[0]))
        stop=0
        for idx, x in tqdm(enumerate(dataset)):
            with torch.no_grad():
                image_features = np.array(model.encode_image(x[0]).float().cpu())#.cpu()
            
            rows = image_features.shape[0]
            if idx != len(dataset)-1:
                start = (idx * rows)
                stop = (idx+1) * rows    
        
            else:
                start = stop
                stop = stop + rows

            embeddings[start:stop, :] = image_features
            clean_up()
        return embeddings

def seed_everything(seed: int):
    random.seed(seed)  # Python's built-in random module
    os.environ['PYTHONHASHSEED'] = str(seed)  # Ensures hash-based operations are deterministic
    np.random.seed(seed)  # NumPy's random generator
    torch.manual_seed(seed)  # PyTorch's CPU RNG
    torch.cuda.manual_seed(seed)  # PyTorch's CUDA RNG (single GPU)
    torch.cuda.manual_seed_all(seed)  # CUDA RNG for multi-GPU
    torch.backends.cudnn.deterministic = True  # Forces cuDNN to be deterministic
    torch.backends.cudnn.benchmark = False  # Disables cuDNN auto-tuner for deterministic behavior

    # For torch's new Generator API (PyTorch 1.8+)
    torch.use_deterministic_algorithms(True, warn_only=True)
    # Optional: If using `transformers` library (Hugging Face)
    try:
        import transformers
        transformers.set_seed(seed)
    except ImportError:
        pass
    # Optional: If using Dataloader workers in PyTorch
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # Ensures deterministic CuBLAS behavior (PyTorch 1.10+)
    print(f"Random seed set to {seed}")


def clean_up():
    torch.cuda.empty_cache()
    gc.collect()

def clip_text(text_lst, model):
    clean_up()
    embeddings = np.zeros(shape=(len(text_lst), model.ln_final.normalized_shape[0]))
    for idx, text in tqdm(enumerate(text_lst)):
        text = clip.tokenize([text], truncate=True).cuda()
        text = model.encode_text(text).cpu().detach().numpy()
        embeddings[idx] = text
    
    return embeddings

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

def combine_features(args, embeddings):
    pic, text = embeddings
    
    if args.combine in ['fusion']:
        print('fusing')
        output = np.multiply(pic, text)
    
    elif args.combine in ['concatenate']:
        print('concatenating')
        output = np.concatenate((pic, text), axis=1)
    
    elif args.combine in ['fancy']:
        print('fancy')
        pic = normalize(pic, axis=1, norm='l2')
        text = normalize(text, axis=1, norm='l2')
        output = np.mean([pic, text], axis=0)
    
    return output

def memotion3(task):
    dataset = dict()
    add_path = 'data/Memotion 3/'
    if task == 1:
        LABEL_LIST = ['very positive', 'positive', 'neutral', 'negative', 'very negative']
        overall_dict = {'very_positive': [1, 0, 0, 0, 0], 
                        'negative': [0, 1, 0, 0, 0], 
                        'very_negative': [0, 0, 1, 0, 0], 
                        'neutral': [0, 0, 0, 1, 0], 
                        'positive': [0, 0, 0, 0, 1]}
    elif task == 2:
        LABEL_LIST = ['humourous', 'sarcastic', 'offensive', 'motivational']
        humor_dict = {'hilarious': 1, 'not_funny': 0, 'very_funny': 1, 'funny': 1}
        sarc_dict = {'not_sarcastic': 0, 'very_twisted': 1, 'general': 1, 'twisted_meaning': 1}
        offensive_dict = {'very_offensive': 1, 'hateful_offensive': 1, 'slight': 1, 'not_offensive':0}
        motive_dict = {'not_motivational': 0, 'motivational': 1}
        
    splits = ['train', 'test']
    for split in splits:
        img_path = []
        labels = []
        if split == 'train':
            df = pd.read_csv(f'{add_path}memotion3/train.csv')
            add_image = f'{add_path}trainImages/trainImages/'
        elif split == 'test':
            df = pd.read_csv(f'{add_path}memotion3-val/val.csv', delimiter='\t')
            add_image = f'{add_path}valImages/'
        df.rename(columns={'ocr': 'ocr_text'}, inplace=True)       
        if task == 1:
            for idx, label in enumerate(df.overall.to_list()):
                img = f'{add_image}{idx}.jpg'
                img_path.append(img)
                label = overall_dict[label]
                labels.append(label)
        elif task == 2:
            humor = df.humour.to_list()
            sarc = df.sarcastic.to_list()
            offen = df.offensive.to_list()
            motive = df.motivational.to_list()
            for idx, (h, s, o, m) in enumerate(zip(humor, sarc, offen, motive)):
                img = f'{add_image}{idx}.jpg'
                img_path.append(img)
                h = humor_dict[h]
                s = sarc_dict[s]
                o = offensive_dict[o]
                m = motive_dict[m]
                label = [h, s, o, m]
                labels.append(label)

        df['img_path'] = img_path
        df['labels'] = labels
        df = datasets.Dataset.from_pandas(df)
        dataset[split] = df

    dataset = datasets.DatasetDict(dataset)

    return LABEL_LIST, dataset

def memex():
    LABEL_LIST = ['baseless', 'valid']
    add_path = 'data/MCC_MemesContextCorpus/archive/'
    #add_img_path = 'data/MCC_MemesContextCorpus/archive/MemeExplanationData/images/'
    add_img_path = 'MCC_MemesContextCorpus/archive/MemeExplanationData/images/'
    dataset = dict()
    for split in ['trainv2', 'testv2']:
        df = pd.read_csv(f'{add_path}{split}.csv')
        
        df['sentences'] = df['sentences'].apply(ast.literal_eval)
        df['labels'] = df['labels'].apply(ast.literal_eval)        
        
        sentences, labels, text_label =  [], [], []
        #image,labels,link,evidence,ocr_text
        img_path = []
        links, evidences, ocr_texts = [], [], []
        for idx, (sentence_list, label_list) in enumerate(zip(df.sentences.to_list(), df.labels.to_list())):
            img = df.image.to_list()[idx]
            img = f'{add_img_path}{img}'
            
            link = df.link.to_list()[idx]; evidence = df.evidence.to_list()[idx]; ocr_text = df.ocr_text.to_list()[idx]

            for sentence, label in zip(sentence_list, label_list):
                if label == 0:
                    keep_label = [0, 1]
                elif label == 1:
                    keep_label = [1, 0]
                sentences.append(sentence)
                labels.append(keep_label)
                text_label.append(LABEL_LIST[label])
                img_path.append(img)
                links.append(link); evidences.append(evidence); ocr_texts.append(ocr_text)
        
        df = {'sentences': sentences,
              'labels': labels,
              'text_label': text_label,
              'img_path': img_path,
              'link': links,
              'evidence': evidences,
              'ocr_text': ocr_texts}
        df = datasets.Dataset.from_pandas(pd.DataFrame(df))
        
        if split in ['trainv2']:
            dataset['train'] = df
        elif split in ['testv2']:
            dataset['test'] = df

    dataset = datasets.DatasetDict(dataset)
    return LABEL_LIST, dataset

def load_dataset(args):
    if args.dataset == "mami":
        return mami(args)
    elif args.dataset == "multioff":
        return multioff(args)
    elif args.dataset == 'memex':
        return memex()
    elif args.dataset == 'memotion3':
        return memotion3(args.task)
    elif args.dataset == 'figmemes':
        return figmemes(args)

def mami(args):
    task = args.task
    path =  args.data_root
    train_df = pd.read_csv(f'{path}training.csv', delimiter='\t')
    test_df = pd.read_csv(f'{path}test.csv', delimiter='\t')
    dataset = dict()
    if task == 1:
        LABEL_LIST = ['neutral', 'misogynous']
        train_df = train_df[['file_name', 'misogynous', 'Text Transcription']]
        test_df = test_df[['file_name', 'misogynous', 'Text Transcription']]
    elif task == 2:
        LABEL_LIST = ['shaming', 'stereotype', 'objectification', 'violence']
        train_df = train_df[['file_name', 'shaming', 'stereotype', 'objectification', 'violence', 'Text Transcription']]
        test_df = test_df[['file_name', 'shaming', 'stereotype', 'objectification', 'violence', 'Text Transcription']]
    
    for split in ['training', 'test']:
        if split in ['training']:
            add_path = 'training/'
            df = train_df
        elif split in ['test']:
            add_path = 'test/'
            df = test_df
        img_path = [f'{path}{add_path}{img}' for img in df.file_name.tolist()]
        df['img_path'] = img_path
        
        df.rename(columns={'Text Transcription': 'ocr_text'}, inplace=True)
        
        if task == 1:
            labels = []
            for lab in df.misogynous.tolist():
                if lab == 0:
                    labels.append([0, 1])
                elif lab == 1:
                    labels.append([1, 0])
            df['labels'] = labels
        elif task == 2:
            df['labels'] = df[LABEL_LIST].values.tolist()
        
        df = df[['img_path', 'labels', 'ocr_text']] 
        df = datasets.Dataset.from_pandas(df)
        
        if split in ['training']:
            dataset['train'] = df
        elif split in ['test']:
            dataset['test'] = df
        
    dataset = datasets.DatasetDict(dataset)
    return LABEL_LIST, dataset


def figmemes(args):
    LABEL_LIST = ["allusion", "exaggeration", "irony", "anthrop", "metaphor", "contrast"]
    STYLE_LIST = ["arts", "real", "infograph", "mixed"]
    folder = os.path.join(args.data_root)
   
    all_features = {}
    with open(os.path.join(folder, "figmemes_annotations.tsv"), "r", encoding="utf-8") as f:
        for row in tqdm(csv.DictReader(f, delimiter="\t"), desc="Annotations"):
            all_features[row["img_name"]] = {
                "img_id": row["img_name"],
                "img_path": os.path.join(folder, "images", row["img_name"]),
                "year": row["year"],
                "labels": [float(row[label]) for label in LABEL_LIST],
                "style": [style for style in STYLE_LIST if row[style]=="1"][0]
            }
    with open(os.path.join(folder, "figmemes_ocrs.tsv"), "r", encoding="utf-8") as f:
        for row in tqdm(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE), desc="Annotations"):
            all_features[row["img_name"]]["ocr_text"] = row["text"]
    with open(os.path.join(folder, f"{args.split}_split.tsv"), "r", encoding="utf-8") as f:
        for row in tqdm(csv.DictReader(f, delimiter="\t"), desc="Annotations"):
            all_features[row["img_name"]]["split"] = row["split"]
            if "year" in row:
                all_features[row["img_name"]]["year"] = row["year"]
            if "cluster" in row:
                all_features[row["img_name"]]["cluster"] = row["cluster"]
   
    split_features = {"train": [], "validation": [], "test": []}
    for feature in all_features.values():
        split = feature.pop("split")
        split_features[split].append(feature)
    for split, split_feature in split_features.items():
        split_features[split] = {k: [f[k] for f in split_feature] for k in split_feature[0].keys()}
    dataset = datasets.DatasetDict({split: datasets.Dataset.from_dict(split_features[split]) for split in split_features})
       
    return LABEL_LIST, dataset

def multioff(args):
    LABEL_LIST = ["Non-offensiv", "offensive"]
    folder = os.path.join(args.data_root)
    all_features = {}
    converter = {"Training_meme_dataset":"train", "Validation_meme_dataset":"validation","Testing_meme_dataset":"test"}
    for split in ["Training_meme_dataset", "Validation_meme_dataset","Testing_meme_dataset"]:
        with open(os.path.join(folder,"Split_Dataset", f"{split}.csv"), "r", encoding="utf-8-sig") as f:
            for row in tqdm(csv.DictReader(f, delimiter=","), desc="Annotations"):
                all_features[row["image_name"]] = {
                    "img_id": row["image_name"],
                    "img_path": os.path.join(folder, "Labelled_Images", row["image_name"]),
                    "ocr_text": row["sentence"],
                    "labels": [1.0 if label in row["label"] else 0.0 for label in LABEL_LIST],
                }
                all_features[row["image_name"]]["split"] = converter[split]

    split_features = {"train": [], "validation": [], "test": []}
    for feature in all_features.values():
        split = feature.pop("split")
        split_features[split].append(feature)
    for split, split_feature in split_features.items():
        split_features[split] = {k: [f[k] for f in split_feature] for k in split_feature[0].keys()}
    dataset = datasets.DatasetDict({split: datasets.Dataset.from_dict(split_features[split]) for split in split_features})
       

    return LABEL_LIST, dataset

def downsample_train_val(train, val, args):
    encoders = ['ViT-L/14@336px', 'ViT-B/32', 'ViT-B/16']
    assert args.feature in encoders
    if args.feature == encoders[0]:
        downsample_dict = {'multioff': {'train': 64, 'val': 53},
                           'memotion3': {'train':926, 'val':231},
                           'figmemes': {'train':751, 'val':255},
                           'mami': {'train':647, 'val':161}}
    elif args.feature == encoders[1]:
        downsample_dict = {'multioff': {'train': 91, 'val': 59},
                           'memotion3': {'train':877, 'val':219},
                           'figmemes': {'train':757, 'val':255},
                           'mami': {'train':705, 'val':176}}
    elif args.feature == encoders[2]:
        downsample_dict = {'multioff': {'train': 78, 'val': 56},
                           'memotion3': {'train':670, 'val':167},
                           'figmemes': {'train':791, 'val':259},
                           'mami': {'train':787, 'val':196}}

    train = train.to_pandas()
    val = val.to_pandas()
    train_samp_size = downsample_dict[args.dataset]['train']
    val_samp_size = downsample_dict[args.dataset]['val'] 
    samp_train = train.sample(train_samp_size, random_state=args.seed)
    samp_val = val.sample(val_samp_size, random_state=args.seed)
    train = train.drop(samp_train.index)
    val = val.drop(samp_val.index)

    return datasets.Dataset.from_pandas(train), datasets.Dataset.from_pandas(val)

def downsample_tsplit(train, val, args):
    encoders = ['ViT-L/14@336px', 'ViT-B/32', 'ViT-B/16']
    assert args.feature in encoders
    if args.feature == encoders[0]:
        downsample_dict = {'multioff': {'train': 64, 'val': 53},
                           'memotion3': {'train':926, 'val':231},
                           'figmemes': {'train':751, 'val':255},
                           'mami': {'train':647, 'val':161}}
    elif args.feature == encoders[1]:
        downsample_dict = {'multioff': {'train': 91, 'val': 59},
                           'memotion3': {'train':877, 'val':219},
                           'figmemes': {'train':757, 'val':255},
                           'mami': {'train':705, 'val':176}}
    elif args.feature == encoders[2]:
        downsample_dict = {'multioff': {'train': 78, 'val': 56},
                           'memotion3': {'train':670, 'val':167},
                           'figmemes': {'train':791, 'val':259},
                           'mami': {'train':787, 'val':196}}
    train = train.to_pandas()
    val = val.to_pandas()
    train_size = len(train)
    val_size = len(val)
    train_samp_size = downsample_dict[args.dataset]['train']
    val_samp_size = downsample_dict[args.dataset]['val']

    if train_size <= train_samp_size:
        train = datasets.Dataset.from_pandas(train)
    else:
        samp_train = train.sample(train_size-train_samp_size, random_state=args.seed)
        train = train.drop(samp_train.index)
        train = datasets.Dataset.from_pandas(train)
    if val_size <= val_samp_size:
        val = datasets.Dataset.from_pandas(val)
    else:
        samp_val = val.sample(val_size-val_samp_size, random_state=args.seed)
        val = val.drop(samp_val.index)
        val = datasets.Dataset.from_pandas(val)
    
    return train, val