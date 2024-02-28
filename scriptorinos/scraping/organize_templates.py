import json
import glob
import os
import shutil


template_key = dict()
seen_titles = []
with open('memes/jsons/meme_pictures.json', 'r') as f:
    for line in f:
        results = dict(json.loads(line))
        template_title = results['title']
        seen_titles.append(template_title)
        if template_title in template_key:
            template_key[template_title].append(results)
        else:
            template_key[template_title] = [results]
print(len(seen_titles))
print(len(set(seen_titles)))

seen_twice = []
template_out = 'template_examples/templates/'
skips =0
hits = 0
crap_count = 0
ok_count = 0
template_info = dict()
for idx, title in enumerate(seen_titles):
    if title in seen_twice:
        skips+=1
        continue
    template_info[title] = {'title': title,
                            'original_info': [],
                            'out_paths': []}
    
    results = template_key[title]
    if len(results) == 1:
        hits+=1
        results = results[0]
                
        pic_title = results['pic_title']
        folder_title = pic_title.split('_')[1]
        folder_path = f'{template_out}{folder_title[:-4]}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        src = f'memes/{pic_title}'
        dst = f'{folder_path}/{folder_title}'        
        shutil.copyfile(src, dst)
        template_info[title]['original_info'].append(results)
        template_info[title]['out_paths'].append(dst)
    else:
        potential_templates = []
        for jdx, dictionary in enumerate(results):
            p_tempts = dictionary['pic_title']
            with open(f'memes/{p_tempts}', 'rb') as f:
                potential_templates.append((f.read(), dictionary))
        
        first = potential_templates[0][0]
        for new_idx, tup in enumerate(potential_templates):
            pic, results = tup
            pic_title = results['pic_title']
            folder_title = pic_title.split('_')[1]
            folder_path = f'{template_out}{folder_title[:-4]}'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            if pic!= first:
                folder_title = f'{new_idx}_{folder_title}'
                src = f'memes/{pic_title}'
                dst = f'{folder_path}/{folder_title}'
                
            else:
                src = f'memes/{pic_title}'
                dst = f'{folder_path}/{folder_title}'              
           
            shutil.copyfile(src, dst)
            template_info[title]['original_info'].append(results)
            template_info[title]['out_paths'].append(dst)      
    
    
    seen_twice.append(title)



print(skips)
for key in template_info.keys():
    out_json = {key: template_info[key]}
    with open('template_examples/jsons/template_info.json', 'a') as f:
        f.write(json.dumps(out_json)+'\n')


print("jobs done!")