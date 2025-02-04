# A Template Is All You Meme
#### Disclaimer: Our work should only ever be used for academic purposes.
Source code and data for [A Template Is All You Meme](https://arxiv.org/abs/2311.06649).

Contact person: [Luke Bates](luke.bates@tu-darmstadt.de)

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.

## Project structure
* `tlc.py` -- Template-Label Counter
* `main.py` -- code file that uses the other code files
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

### Reproduce our results

You can run TLC with `python main.py`. You can specifiy which configurations by passing arguments to python.
```
--template_path #directory where the KYMKB is located
--dataset #which dataset from the paper you want to play with
--data_root #where the datafiles are. only relevant for figmemes, mami, and multioff, which you should pass something like data/annotations, data/MAMI_DATASET, and data/MultiOFF_DATASET respectively
--num_neigh #how many neighbors are we talking about
--vote_type #template vs label vote
--split #only relevant for figmemes, mami, and multioff, which you should pass something like standard, task5_style, and standard respectively
--include_examples #template or templates+examples? True or False, respectively
--feature_extraction #which encoder? ViT-L/14@336px, ViT-B/32, or ViT-B/16
--task #only relevant for Memotion 3 1 = A, 2 = B
--combine #how to model the modalities, None (just template vs memes), concatenate, fusion, latefusion, or fancy (normalize then average)
--just_text #use just about vs OCR? True or False
--need_to_read #use our embeddings or note? True or False
```


### Expected results
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
