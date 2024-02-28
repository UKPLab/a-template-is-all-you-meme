import hashlib
import glob
import json
import os

template_info = []
with open('template_examples/jsons/template_info.json', 'r') as f:
    for line in f:
        template_info.append(dict(json.loads(line)))

for dictionary1 in template_info:
    for key in dictionary1.keys():
        hashes = set()       
        write_path = dictionary1[key]['out_paths'][0].split('/')[:-1]
        write_path = '/'.join(write_path)
        write_path = f'{write_path}/examples/*'
        for file in glob.glob(write_path):
            digest = hashlib.sha1(open(file,'rb').read()).digest()
            if digest not in hashes:
                hashes.add(digest)
            else:
                os.remove(file)