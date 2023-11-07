import argparse
import clip
from memetils import load_dataset, str2bool
from tlc import TemplateLabelCounter as TLC

parser = argparse.ArgumentParser(description='Template Label Counter')
parser.add_argument('--template_path', action='store', type=str, dest='path', default='data/template_examples/jsons/template_info.json')
parser.add_argument('--dataset', action="store", type=str, dest='dataset', default='figmemes')
parser.add_argument('--data_root', action="store", type=str, dest='data_root', default='data/annotations')
parser.add_argument('--num_neigh', action="store", type=int, dest='num_neigh', default=1)
parser.add_argument('--vote_type', action="store", type=str, dest='vote_type', default='template')
parser.add_argument('--split', action="store", type=str, dest='split', default='standard')
parser.add_argument('--all_feature_type', action="store", type=str, dest='all_feature_type', default='')
parser.add_argument('--include_examples', action="store", type=str2bool, dest='examples', default=False)
parser.add_argument('--feature_extraction', action="store", type=str, dest='feature', default='pixel')
parser.add_argument('--meme_size', action="store", type=int, dest='meme_size', default=64)
parser.add_argument('--task', action="store", type=int, dest='task', default=1)
parser.add_argument('--combine', action="store", type=str, dest='combine', default='None')
parser.add_argument('--just_text', action="store", type=str2bool, dest='just_text', default='False')
parser.add_argument('--need_to_read', action="store", type=str2bool, dest='need_to_read', default='False')

args = parser.parse_args()

dataset = load_dataset(args)

if not args.need_to_read:
    model, preprocess = clip.load(args.feature, download_root='models/clip_models/')
    model.cuda().eval()
    TLC = TLC(args=args, dataset=dataset, model=model, preprocess=preprocess, need_to_read=False)
else:
    TLC = TLC(args=args, dataset=dataset, need_to_read=True)

TLC.run()

print()
print("Job's Done!")
exit()

