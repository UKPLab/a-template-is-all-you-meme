import json
import glob
import os
import shutil

# Helper function to create folder if not exists
def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Helper function to process and move meme files
def process_meme_files(results, template_out, template_info, title, seen_twice):
    if len(results) == 1:
        # Single result
        process_single_meme(results[0], template_out, template_info, title)
    else:
        # Multiple results - potential template
        process_multiple_memes(results, template_out, template_info, title)
    seen_twice.append(title)

# Process single meme
def process_single_meme(result, template_out, template_info, title):
    pic_title = result['pic_title']
    folder_title = pic_title.split('_')[1]
    folder_path = os.path.join(template_out, folder_title[:-4])

    # Create folder if it does not exist
    create_folder(folder_path)

    # Define file paths
    src = os.path.join('memes', pic_title)
    dst = os.path.join(folder_path, folder_title)

    try:
        shutil.copyfile(src, dst)
        template_info[title]['original_info'].append(result)
        template_info[title]['out_paths'].append(dst)
    except FileNotFoundError as e:
        print(f"File {src} not found. Skipping...")
    
# Process multiple memes (potential templates)
def process_multiple_memes(results, template_out, template_info, title):
    potential_templates = []
    for dictionary in results:
        pic_title = dictionary['pic_title']
        try:
            with open(f'memes/{pic_title}', 'rb') as f:
                potential_templates.append((f.read(), dictionary))
        except FileNotFoundError:
            print(f"File {pic_title} not found. Skipping...")
            continue

    if not potential_templates:
        print(f"No valid templates for title {title}. Skipping...")
        return

    first = potential_templates[0][0]
    for new_idx, (pic, result) in enumerate(potential_templates):
        pic_title = result['pic_title']
        folder_title = pic_title.split('_')[1]
        folder_path = os.path.join(template_out, folder_title[:-4])
        
        # Create folder if it does not exist
        create_folder(folder_path)

        # Define file paths
        if pic != first:
            folder_title = f'{new_idx}_{folder_title}'
        
        src = os.path.join('memes', pic_title)
        dst = os.path.join(folder_path, folder_title)
        
        try:
            shutil.copyfile(src, dst)
            template_info[title]['original_info'].append(result)
            template_info[title]['out_paths'].append(dst)
        except FileNotFoundError:
            print(f"File {src} not found. Skipping...")

# Main function to orchestrate the processing
def process_templates(template_out='template_examples/templates/'):
    template_key = dict()
    seen_titles = []

    # Load meme pictures JSON and organize by title
    with open('jsons/meme_pictures.json', 'r') as f:
        for line in f:
            results = json.loads(line)
            template_title = results['title']
            seen_titles.append(template_title)
            template_key.setdefault(template_title, []).append(results)

    print(f"Total titles: {len(seen_titles)}")
    print(f"Unique titles: {len(set(seen_titles))}")

    seen_twice = []
    skips = 0
    hits = 0
    template_info = {}

    for idx, title in enumerate(seen_titles):
        if title in seen_twice:
            skips += 1
            continue

        template_info[title] = {'title': title, 'original_info': [], 'out_paths': []}
        results = template_key[title]

        try:
            process_meme_files(results, template_out, template_info, title, seen_twice)
            hits += 1
        except Exception as e:
            print(f"Error processing {title}: {e}")
            continue

    print(f"Skipped: {skips}")
    print(f"Processed successfully: {hits}")

    # Save template information to JSON
    with open('template_examples/jsons/template_info.json', 'a') as f:
        for key, info in template_info.items():
            out_json = {key: info}
            f.write(json.dumps(out_json) + '\n')

    print("Jobs done!")

# Run the template processing
process_templates()
