import json
import glob
import os
import shutil
from tqdm import tqdm

# Helper function to ensure the directory exists
def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

# Function to copy example files to their destination
def copy_example_files(examples, write_path):
    for idx, example in enumerate(examples):
        dst = os.path.join(write_path, f'{idx}_{os.path.basename(example)}')
        
        # Check if the destination directory exists, if not, skip copying
        if len(dst.split('/')) != 5:
            print(f"Skipping invalid file: {example}")
            continue
        
        try:
            shutil.copyfile(example, dst)
        except FileNotFoundError:
            print(f"File not found: {example}. Skipping...")
            continue
        except PermissionError:
            print(f"Permission error with file: {example}. Skipping...")
            continue
        
        # Logging every 1000 copies
        if (idx + 1) % 1000 == 0:
            print(f'{idx + 1} files copied.')

# Main function to process template information and copy files
def process_template_info():
    workstation = 'workstation_scrapes/meme_examples/*_'
    cluster = 'memes/meme_examples/*_'
    template_info = []

    # Load template info from JSON
    with open('template_examples/jsons/template_info.json', 'r') as f:
        for line in f:
            template_info.append(dict(json.loads(line)))

    count = 0
    for dictionary1 in tqdm(template_info):
        for key in dictionary1.keys():
            examples, seen_bytes = [], []

            # Prepare the write path and check if the directory exists
            write_path = dictionary1[key]['out_paths'][0].rsplit('/', 1)[0] + '/examples'
            create_directory(write_path)
            
            # Gather all example paths
            station_title_path = f'{workstation}{key}/*'
            cluster_title_path = f'{cluster}{key}/*'
            
            # Add examples from both workstation and cluster directories
            examples.extend(glob.glob(station_title_path))
            examples.extend(glob.glob(cluster_title_path))

            # Copy each example to the new directory
            copy_example_files(examples, write_path)
            count += len(examples)

    print(f'Jobs done! {count} files copied.')

# Run the main function to process the template info
process_template_info()
