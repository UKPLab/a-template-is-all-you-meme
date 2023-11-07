import csv
import os
import ast
import datasets
import h5py
import pandas as pd
from tqdm import tqdm

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

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
    LABEL_LIST = ["shaming",	"stereotype",	"objectification", 	"violence"]
    STYLE_LIST = ["art", "real", "infograph", "mixed"]
    folder = os.path.join(args.data_root)
    # TODO comment out
    try:
       dataset = datasets.load_from_disk(os.path.join(folder, "hf_dataset", args.split, args.all_feature_type))
    except FileNotFoundError:
        print("TiMemes HF dataset does not yet exist. Creating it.")
        all_features = {}

        for split in ["training", "test"]:
            with open(os.path.join(folder, f"{split}.csv"), "r", encoding="utf-8-sig") as f:
                for row in tqdm(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE), desc="Annotations"):
                    k = "file_name"
                    name = f'{split}/{row[k]}'
                    all_features[name] = {
                        "img_id": name,
                        "img_path": os.path.join(folder, f"{split}", f'{row[k]}'),
                        "ocr_text": row["Text Transcription"],
                        "labels": [float(row[label]) for label in LABEL_LIST]
                    }
            with open(os.path.join(folder, "style_labels", f"{split.replace('ing', '')}_style_labels.tsv"), "r",
                        encoding="utf-8") as f:
                for row in tqdm(csv.DictReader(f, delimiter="\t"), desc="Annotations"):
                    name = f'{split}/{row["img_id"]}'
                    assert name in all_features
                    for style in STYLE_LIST:
                        if int(row[style]) == 1:
                            all_features[name]["style"] = style
                            break
        with open(os.path.join(folder, f"{args.split}_split.tsv"), "r", encoding="utf-8") as f:
            for row in tqdm(csv.DictReader(f, delimiter="\t"), desc="Annotations"):
                if row["img_name"] not in all_features:
                    print(row["img_name"])
                    continue
                all_features[row["img_name"]]["split"] = row["split"]
                if "year" in row:
                    all_features[row["img_name"]]["year"] = row["year"]
                if "cluster" in row:
                    all_features[row["img_name"]]["cluster"] = row["cluster"]
        if args.all_feature_type:
            for split in ["training", "test"]:
                for feature_type in args.all_feature_type.split(","):
                    with h5py.File(os.path.join(folder, "features", f"{split}_{feature_type}.h5"), "r") as f:
                        for img_id in f.keys():
                            name = f'{split}/{img_id}'
                            if name not in all_features:
                                print(f"Image {name} does not exist in any split.")
                                continue
                            all_features[name][f"{feature_type}_feature"] = f[f'{img_id}/features'][()]
                            all_features[name][f"img_h"] = f[f'{img_id}/img_h'][()]
                            all_features[name][f"img_w"] = f[f'{img_id}/img_w'][()]
                            all_features[name][f"{feature_type}_rect"] = f[f'{img_id}/boxes'][()]
        split_features = {"train": [], "validation": [], "test": []}
        for feature in all_features.values():
            split = feature.pop("split")
            split_features[split].append(feature)
        for split, split_feature in split_features.items():
            split_features[split] = {k: [f[k] for f in split_feature] for k in split_feature[0].keys()}
        dataset = datasets.DatasetDict({split: datasets.Dataset.from_dict(split_features[split]) for split in split_features})
        #dataset.save_to_disk(os.path.join(folder, "hf_dataset", args.split, args.all_feature_type))

    return LABEL_LIST, dataset


def figmemes(args):
    LABEL_LIST = ["allusion", "exaggeration", "irony", "anthrop", "metaphor", "contrast"]
    STYLE_LIST = ["arts", "real", "infograph", "mixed"]
    folder = os.path.join(args.data_root)
    #try:
    #    dataset = datasets.load_from_disk(os.path.join(folder, "hf_dataset", args.split, args.all_feature_type))
    #except FileNotFoundError:
        #print("FigMemes HF dataset does not yet exist. Creating it.")
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
    # for feature_type in args.all_feature_type.split(","):
    #     with h5py.File(os.path.join(folder, "features", f"all_{feature_type}.h5"), "r") as f:
    #         for img_id in f.keys():
    #             if img_id not in all_features:
    #                 print(f"Image {img_id} does not exist in any split.")
    #                 continue
    #             all_features[img_id][f"{feature_type}_feature"] = f[f'{img_id}/features'][()]
    #             all_features[img_id][f"img_h"] = f[f'{img_id}/img_h'][()]
    #             all_features[img_id][f"img_w"] = f[f'{img_id}/img_w'][()]
    #             all_features[img_id][f"{feature_type}_rect"] = f[f'{img_id}/boxes'][()]
    split_features = {"train": [], "validation": [], "test": []}
    for feature in all_features.values():
        split = feature.pop("split")
        split_features[split].append(feature)
    for split, split_feature in split_features.items():
        split_features[split] = {k: [f[k] for f in split_feature] for k in split_feature[0].keys()}
    dataset = datasets.DatasetDict({split: datasets.Dataset.from_dict(split_features[split]) for split in split_features})
        #dataset.save_to_disk(os.path.join(folder, "hf_dataset", args.split, args.all_feature_type))

    return LABEL_LIST, dataset

def multioff(args):
    LABEL_LIST = ["Non-offensiv", "offensive"]
    folder = os.path.join(args.data_root)
    #try:
        #dataset = datasets.load_from_disk(os.path.join(folder,"Split_Dataset", "hf_dataset"))
        #dataset = datasets.load_from_disk(os.path.join(folder, "hf_dataset", args.split, args.all_feature_type))
    #except FileNotFoundError:
        #print("MultiOff HF dataset does not yet exist. Creating it.")
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
        #dataset.save_to_disk(os.path.join(folder, "hf_dataset", args.split, args.all_feature_type))

    return LABEL_LIST, dataset