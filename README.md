# A Template Is All You Meme
#### Disclaimer: Our work should only ever be used for academic purposes.
Source code and data for [A Template Is All You Meme](https://arxiv.org/abs/2311.06649).

Contact person: [Luke Bates](luke.bates@tu-darmstadt.de)

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.

## Project structure
* `finetune_clip.py` -- fine-tuning CLIP with TSplit / Original splits / baseline (downsampling)
* `clip_model.py` -- CLIP model for fine-tuning
* `tsplit.py` -- Template-Aware Splitter
* `tlc.py` -- Template-Label Counter
* `main.py` -- for running TLC
* `memetils.py` -- util code
* `scriptorinos/` -- eda and scraping scripts

## Requirements
Install [clip first](https://github.com/openai/CLIP).

Then, please use the `requirements.txt` file. 

## KYMKB / Data / Embeddings
#### Disclaimer: Our work should only ever be used for academic purposes.
[Our data files](https://knowyourmeme.com/memes/chonk-oh-lawd-he-comin) are some [chonky bois](https://knowyourmeme.com/memes/big-chungus)

For the moment, please contact us about accessing the KYMKB. :)

Remember, sometimes memes are mean. We take no responsiblility if they are offensive nor do they reflect our views in any way.

## Installation
To setup, please follow the instructions below.
```
git clone https://github.com/UKPLab/a-template-is-all-you-meme.git
cd a-template-is-all-you-meme
python -m venv mvenv
source mvenv/bin/activate
pip install --upgrade pip
#install clip here please
pip install -r requirements.txt
```
### Reproduce our results: TSplit
You can finetune CLIP with `python finetune_clip.py`. You can specifiy which configurations by passing arguments to python.
```
--dataset #which dataset from the paper you want to play with
--data_root #where the datafiles are. only relevant for figmemes, mami, and multioff, which you should pass data/annotations, data/MAMI_DATASET, and data/MultiOFF_DATASET respectively
--split #only relevant for figmemes, mami, and multioff, which you should pass standard, task5_style, and standard respectively
--feature_extraction #which encoder? We used ViT-L/14@336px, ViT-B/32, or ViT-B/16
--task #only relevant for Memotion 3 and MAMI 1 = A, 2 = B
--reorganize #original for the original splits, baseline for random downsampling, max for TSplit_max, mean for TSplit_mean, median for TSplit_median, quantile for TSplit_percentile
--batch_size # We used 16
--epochs #fine-tuning epochs. we used 20
--seed # random seed for modelling/sampling. we use 0-4
--sample_train #Downsample TSplit/CLIP Baseline (Table 3) (True or False)
--random_downsample_tsplit #randomly downsample after TSplitting entire dataset (Table 9) (True or False)
--sample_tsplit #TSplit downsampling on entire dataset (Table 9) (True or False)
--overfit #skip model selection and just do test eval on the model fine-tuned for args.epochs (20) epochs. (Table 6) (True or False)
If the preceding 4 arguments are all False, you will TSplit the entire dataset (Table 4)
```
### TSplit expected results
Results will be written to disk in a json file following this structure:
```
clip_results/{args.overfit}/{args.sample_train}/{args.random_downsample_tsplit}/{args.sample_tsplit}/{args.dataset}/{args.reorganize}/{args.feature}/{args.task}/{args.seed}/
```
### Reproduce our results: TLC

You can run TLC with `python main.py`. You can specifiy which configurations by passing arguments to python.
```
--template_path #directory where the KYMKB is located
--dataset #which dataset from the paper you want to play with
--data_root #where the datafiles are. only relevant for figmemes, mami, and multioff, which you should pass data/annotations, data/MAMI_DATASET, and data/MultiOFF_DATASET respectively
--num_neigh #how many neighbors are we talking about
--vote_type #template vs label vote
--split #only relevant for figmemes, mami, and multioff, which you should pass standard, task5_style, and standard respectively
--include_examples #template or templates+examples? True or False, respectively
--feature_extraction #which encoder? ViT-L/14@336px, ViT-B/32, or ViT-B/16
--task #only relevant for Memotion 3 and MAMI 1 = A, 2 = B
--combine #how to model the modalities, None (just template vs memes), concatenate, fusion, latefusion, or fancy (normalize then average)
--just_text #use just about vs OCR? True or False
--need_to_read #use our embeddings or not? True or False
```


### TLC Expected results
Once finished, results will be printed out.

### Citation
If our work was helpful for your work, please be so kind as to cite us:
```
@article{atiaym_2023,
url = {https://arxiv.org/abs/2311.06649},
author = {Luke Bates and Peter Ebert Christensen and Preslav Nakov and Iryna Gurevych},
keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
journal={arXiv preprint arXiv:2311.06649},
title = {A Template Is All You Meme},
publisher = {arXiv},
year = {2023},
}
```

### Update coming soon!
