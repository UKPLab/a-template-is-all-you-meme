import hashlib
import glob
import json
import os

# Helper function to compute the SHA1 hash of a file
def get_file_hash(file_path):
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha1(f.read()).digest()
        return file_hash
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

# Function to process the template info and remove duplicate files based on hashes
def remove_duplicate_files():
    template_info = []

    # Load template info from JSON
    with open('template_examples/jsons/template_info.json', 'r') as f:
        for line in f:
            template_info.append(dict(json.loads(line)))

    for dictionary1 in template_info:
        for key in dictionary1.keys():
            hashes = set()  # Set to keep track of unique file hashes
            
            # Prepare the write path
            write_path = dictionary1[key]['out_paths'][0].rsplit('/', 1)[0] + '/examples/*'
            
            # Iterate over all files in the directory
            for file in glob.glob(write_path):
                file_hash = get_file_hash(file)
                
                if file_hash is None:
                    continue  # Skip the file if there was an error reading it
                
                # If the file hash is already in the set, it's a duplicate, so remove it
                if file_hash in hashes:
                    try:
                        os.remove(file)
                        print(f"Removed duplicate file: {file}")
                    except Exception as e:
                        print(f"Error removing file {file}: {e}")
                else:
                    hashes.add(file_hash)

    print("Duplicate file removal complete.")

# Run the function to remove duplicates
remove_duplicate_files()
