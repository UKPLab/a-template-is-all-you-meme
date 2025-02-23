# Use Our Scraping Code to Make Your Own KYMKB!

### Step 1: Install Dependencies

Make sure you have installed the environment, particularly **BeautifulSoup** and **Selenium**.

### Step 2: Run `downloader.ipynb`

This step will download and write to disk all the parent entries in the "confirmed" section of Know Your Meme.

#### Code Explanation

- **WebDriver Setup**:  
  The script initializes a Chrome browser using **Selenium WebDriver** and automatically downloads the correct **ChromeDriver** using the **webdriver-manager** package.

- **Navigation**:  
  The script scrapes the list of memes starting from the first page and moves to subsequent pages using the "next page" button. It continuously saves each page's HTML source.

- **Error Handling**:  
  If an issue occurs with locating the "next page" button, the script will gracefully terminate and inform the user.

- **Sleep Time**:  
  The script waits for **2 seconds** after loading each page to ensure that the page is fully loaded before performing actions.

#### Output

The HTML source of each meme list page will be saved in the **`selenium_snapshots`** directory as `page_{page_number}.html`. Each file represents one page of memes.

---

### Step 3: Run `parseandwrite.py`

This script looks at all the parent entries in the "confirmed" section, downloads them, and writes them to disk.

#### Code Explanation

1. **Imports and Setup**  
   The script uses the following libraries:
   - **requests**: For HTTP requests to access the **Wayback Machine API** and fetch snapshots.
   - **BeautifulSoup** from **bs4**: To parse and extract links from HTML pages.
   - **json**: For reading and writing JSON data.
   - **os**: To handle file and directory creation.
   - **tqdm**: To display a progress bar during the process.

2. **Directories and Data Loading**  
   - **snapshots_dir**: Directory for storing the HTML snapshots.
   - **jsons_dir**: Directory for storing metadata.
   - The script loads previously processed meme URLs from **meme_tables.json** to avoid re-processing them. If there are any missed URLs (for which snapshots were not found), they are loaded from **miss.json**.

3. **Fetching Wayback Machine Snapshots**  
   - The `get_wayback_snapshot()` function queries the Wayback Machine’s API for a snapshot of a given meme URL. If a snapshot is found, the function returns the snapshot’s URL and timestamp; otherwise, it returns `None`.

4. **Processing HTML Files**  
   - The script iterates over all HTML files in the **`selenium_snapshots/`** directory.
   - It extracts meme URLs by looking for `<img>` tags with the attribute `data-data-entry-name`.
   - Invalid URLs (those not starting with "http") are ignored.

5. **Handling Already Processed URLs**  
   - URLs that have already been processed (stored in **meme_tables.json**) are excluded from the processing queue.

6. **Fetching and Storing Snapshots**  
   - For each new meme URL, the script fetches the closest **Wayback Machine snapshot**.
   - If a snapshot is found, the script downloads the snapshot content and saves it as an HTML file in the appropriate directory.
   - The script writes the URL of each successfully processed meme snapshot into **meme_tables.json** and logs missed URLs in **miss.json**.

7. **Error Handling**  
   - If a request fails while fetching a Wayback Machine snapshot or HTML content, the script will wait for **10 minutes** before retrying.
   - If no snapshots are found for a meme, it logs the missed URL for future attempts.

---

### Output

- **HTML Files**:  
  Each snapshot of a meme is saved as an HTML file in the **`selenium_snapshots`** directory under a folder named by the timestamp of the snapshot.  
  For example, a snapshot with the timestamp `20230222120000` would be saved in **`selenium_snapshots/20230222120000/`**.  
  The files are named `1_tablecontent.html`, `2_tablecontent.html`, etc., depending on how many snapshots have been processed.

- **JSON Files**:  
  - **meme_tables.json**: This file stores metadata about the successfully processed meme URLs and their associated snapshots. Each entry contains the `url` of the snapshot and the `count` of how many memes have been processed.
  - **miss.json**: This file contains URLs for which snapshots could not be found. Each entry in this file includes the `url` and the `count` of missed URLs.
 

### Step 4: Run `get_images.py`

This script will find all the meme template images and metadata (title, description, etc.), downloading the images and parsing the pages for additional information. This step will take a very long time.

This Python script scrapes meme images and metadata (such as titles, descriptions, and sources) from HTML files stored in the **`selenium_snapshots/`** directory. It uses the **Wayback Machine** to find snapshots of meme pages and saves the corresponding images and metadata for further use.

#### Code Explanation

1. **Imports and Setup**
   - The script uses the following libraries:
     - **`requests`**: To send HTTP requests to the Wayback Machine API and download images.
     - **`BeautifulSoup`** from **`bs4`**: To parse the HTML files and extract metadata.
     - **`json`**: To save metadata about the images in JSON files.
     - **`time`**: To introduce delays for rate-limiting and retries.
     - **`os`**: For file and directory handling.
     - **`re`**: For sanitizing file names (removing non-alphanumeric characters).
     - **`glob`**: For listing files in the **`selenium_snapshots/`** directory.
   - The script creates necessary directories:  
     **`pictures`** for storing images, and **`jsons`** for metadata.

2. **Requesting Wayback Machine Snapshot**
   - The function **`request_get_snapshot(url)`** sends a request to the Wayback Machine API to retrieve the closest snapshot for a given URL.
   - If a snapshot is found, the function returns the snapshot's URL and metadata; otherwise, it returns `None`.

3. **Processing HTML Files**
   - The script iterates over all HTML files stored in the **`selenium_snapshots/`** directory, specifically looking for files that match the pattern `*_tablecontent.html`.
   - For each HTML file, the script:
     1. Extracts the image URL from the `og:image` meta tag.
     2. Extracts other metadata fields (title, description, origin) using the `og:title`, `og:description`, and `og:site_name` meta tags.
     3. Collects other meta tags from the HTML and stores them in a dictionary (`other_meta`).

4. **Handling Image Downloads**
   - The script attempts to download the image from the extracted URL using **`requests.get()`**.
   - If the image download fails, it retries after waiting for **10 minutes**.
   - If the download still fails, the URL is recorded in the **`miss_images.json`** file.

5. **Sanitizing Image Titles**
   - The image titles are cleaned by removing non-alphanumeric characters using a regular expression (`re.sub`).
   - The sanitized title is used to save the image in the **`pictures`** directory.

6. **Saving Metadata and Images**
   - The script saves the image metadata (URL, title, description, etc.) in the **`meme_pictures.json`** file.
   - The meme image is saved in the **`pictures`** directory with a sanitized title.

7. **Retries and Rate Limiting**
   - To avoid overloading the server, the script includes delays (**`time.sleep(1)`**) between requests.
   - If the image download fails, it retries after waiting for **10 minutes** (**`time.sleep(60*10)`**).

---

### Output

- **Metadata JSON (`meme_pictures.json`)**:  
  Each entry in the **`meme_pictures.json`** file contains the following metadata for a meme image:
  ```json
  {
    "url": "https://example.com/meme.jpg",
    "title": "Meme Title",
    "about": "Meme Description",
    "origin": "Origin of the Meme",
    "pic_title": "pictures/1_Meme_Title.jpg",
    "other_meta": {
      "author": "Author Name",
      "tags": "tag1, tag2"
    }
  }

### Step 5 (optional): Review `read_images.py`

This script processes meme images, resizes them to a fixed size, and stores them in a template array for further use. It reads meme image data from a JSON file, processes the images (including resizing and handling potential issues with GIFs), and saves the processed images to a specified directory.

#### Code Explanation

1. **Imports and Setup**
   - The script uses several libraries:
     - **`json`**: To handle meme image metadata.
     - **`cv2` (OpenCV)**: To process image files and resize them.
     - **`os`**: For file and directory handling.
     - **`re`**: For sanitizing filenames.
     - **`numpy`**: To store the resized image data in an array.
     - **`matplotlib.pyplot`**: For saving images.
     - **`skimage.transform.resize`**: To resize images.
     - **`tqdm`**: For displaying a progress bar.
     - **`logging`**: For error and progress logging.
   
   - The script defines some constants, such as **`meme_size`** for resizing images and **`double_check`** for storing the processed images.

2. **Image Processing Function**
   - The **`process_image()`** function reads an image (or GIF) and resizes it to the desired **`meme_size`** (64x64).
   - It handles both static images (using **`cv2.imread()`**) and animated GIFs (by reading all frames and resizing the first frame).
   - If the image cannot be read, it logs an error and returns `None`.

3. **Loading Meme Data**
   - The script loads the meme metadata from the **`jsons/meme_pictures.json`** file.
   - It filters out duplicate entries based on the image URL to ensure each meme is processed only once.

4. **Processing Each Meme**
   - The script processes each meme image by:
     1. Retrieving the image path from the metadata.
     2. Attempting to process and resize the image using **`process_image()`**.
     3. If the image is valid, the script saves it with a sanitized filename (removing non-alphanumeric characters).
     4. The processed image is then stored in a **NumPy array** (**`templates`**).
   
   - If any errors occur during processing (such as issues with resizing or saving the image), the script logs the errors and adds the problematic meme to a **`problems`** list for later review.

5. **Saving Processed Images**
   - After resizing, the image is saved in the **`double_check/`** directory with a filename that includes its index and sanitized title.
   - The **`plt.imsave()`** function is used to save the image to disk.
   
6. **Storing Processed Images in Templates**
   - The resized images are stored in the **`templates`** NumPy array.
   - The **`templates`** array is then flattened to a **2D array** (**`templates_flatten`**) for future use in machine learning or image analysis tasks.

7. **Error Handling and Logging**
   - The script logs various stages of the processing, including errors encountered while reading, resizing, or saving images.
   - The **`problems`** list is used to store dictionaries of problematic meme images, which can be reviewed later.

8. **Final Output**
   - The script prints a log message (`'Job done!'`) when all meme images have been processed and saved successfully.

---

### Output

- **Processed Meme Images:**
  - Resized meme images are saved in the **`double_check/`** directory with filenames based on their index and sanitized titles (e.g., `0_Meme_Title.jpg`).
  
- **Problems List (`problems`):**
  - If any images cannot be processed, the corresponding meme metadata (URL, title, etc.) is added to the **`problems`** list.
  - This allows you to review which meme images failed to process, with details available for troubleshooting.

- **Logging Output:**
  - Throughout the script, various log messages are generated, indicating the progress of image processing and any errors that occur.

### Step 6: Run `get_examples.py`

Run `get_examples.py` to visit each template page and download/write to disk all available examples. This will take a very, very long time.

---

#### Code Explanation

1. **Imports and Setup**
   - **Libraries Used:**
     - **`requests`**: For downloading content from URLs.
     - **`BeautifulSoup`** from **`bs4`**: To parse HTML files and extract metadata.
     - **`json`**: For handling JSON data.
     - **`time`**: For adding delays between requests.
     - **`os`**: For file and directory management.
     - **`re`**: For regex-based text processing.
     - **`tqdm`**: For displaying a progress bar.
     - **`glob`**: For file pattern matching.
     - **`utils.request_get_snapshot`**: For fetching the Wayback Machine snapshot of a URL.

2. **Function Definitions**

   - **`process_images(hit_count, pic_title, example_count, image_snapshot, html_file, title, about)`**
     - Downloads the image from the Wayback Machine snapshot and stores it in a dedicated directory.
     - Saves the image metadata (URL, title, etc.) to a JSON file (**`example_pictures.json`**).
     - Saves the image to disk in a folder based on the **`hit_count`** and **`pic_title`**.
   
   - **`process_misses(html_file, title, miss_count, if_statement=False)`**
     - Handles cases where an image download or snapshot retrieval fails.
     - Logs the failure to the **`miss_examples.json`** file.

   - **`save_json(data, filename)`**
     - Helper function to append JSON data to a specified file.

   - **`process_files(html_file, hit_files, miss_files, seen_files, hit_count, miss_count)`**
     - Processes each HTML file to extract image metadata (e.g., title, description).
     - Extracts the image URL from `<img>` tags in the HTML and requests a Wayback Machine snapshot.
     - If the snapshot is found, it processes and saves the image; if not, logs the failure.

3. **Main Workflow**
   - **`main()`**
     - Loads previously processed hit and miss files from the **`example_pictures.json`** and **`miss_examples.json`** files.
     - Tracks already seen files to prevent re-processing.
     - Loops over all HTML files matching the pattern **`selenium_snapshots/*/*_tablecontent.html`**.
     - For each file, the script calls **`process_files()`** to extract image URLs, retrieve snapshots, and download images.
     - Updates the **`hit_count`** and **`miss_count`** to reflect the processed files.

4. **Error Handling**
   - The script includes robust error handling. If an image cannot be processed or a snapshot cannot be retrieved, it attempts a retry after waiting for 10 minutes (**`time.sleep(60*10)`**).
   - If the retry also fails, it logs the failure in the **`miss_examples.json`** file.

5. **Output Files**
   - **`example_pictures.json`**: Stores metadata for successfully processed images, including URLs, titles, descriptions, and image file paths.
   
   - **`miss_examples.json`**: Logs failed processing attempts, with information about the HTML file and the reason for the failure.

6. **File Organization**
   - The processed images are saved in directories within the **`meme_examples`** folder. Each directory is named using a combination of the **`hit_count`** and **`pic_title`**.
   - The images are saved with filenames such as **`1_Meme_Title.jpg`**, where **`1`** is the index of the processed example.

7. **Running the Script**
   - The script iterates over the HTML files located in the **`selenium_snapshots/`** directory and processes them one by one.
   - It utilizes the **`tqdm`** progress bar to track the overall progress during execution.

---

#### Example JSON Format for Processed Memes

For each successfully processed meme, the following metadata is stored in the **`example_pictures.json`** file:

```json
{
  "url": "https://example.com/meme.html",
  "title": "Meme Title",
  "image": "https://web.archive.org/web/20250223/https://example.com/meme.jpg",
  "about": "Description of the meme",
  "pic_directory": "meme_examples/1_Meme_Title/",
  "example_title": "1_Meme_Title.jpg"
}

#### Step 7
#### Step 7
Run `org_temps.py` to get the templates in the final format we used for our experiments.

# Meme Template Processing and Organization

This Python script processes meme files, organizes them by title, and creates directories for each meme template. The script manages both single meme files and multiple memes that may represent a template. The processed memes and their metadata are saved in a structured directory, and information is logged into a JSON file for future reference.

## Code Explanation

### 1. **Imports and Helper Functions**
   - **Libraries Used:**
     - `json` for handling JSON data.
     - `glob` for pattern matching files.
     - `os` for file and directory management.
     - `shutil` for file copying and moving.

   - **Helper Functions:**
     - **`create_folder(folder_path)`**:
       - Checks if a folder exists and creates it if it doesn't.
     
     - **`process_meme_files(results, template_out, template_info, title, seen_twice)`**:
       - Handles the processing of meme files. If there is only one result, it processes it as a single meme. If there are multiple results, it considers them as potential templates and processes them accordingly.

     - **`process_single_meme(result, template_out, template_info, title)`**:
       - Processes a single meme by copying the meme image from the `memes` folder to the specified template output directory. It also stores the original information and output path in the `template_info` dictionary.
     
     - **`process_multiple_memes(results, template_out, template_info, title)`**:
       - Processes multiple meme images, treating them as potential templates. It stores the images in directories and tracks the original information and output paths.

### 2. **Main Template Processing Function**
   - **`process_templates(template_out='template_examples/templates/')`**
     - The main function that orchestrates the processing of templates.
     - Loads meme data from the `meme_pictures.json` file and organizes it by title.
     - Iterates over the titles and processes the meme files accordingly.
     - The function tracks successfully processed titles and skips duplicates.
     - It uses the helper functions to handle the actual processing of meme files (both single and multiple).
     - It saves the processed template information in `template_info.json` for later use.

### 3. **Error Handling and Logging**
   - The script handles errors such as missing files (`FileNotFoundError`) and logs them with appropriate messages.
   - It also tracks how many titles were skipped (if they were already processed) and how many were processed successfully.

### 4. **JSON Output Format**
   After processing, the script saves the metadata for each template in a JSON file (`template_info.json`). The format for each entry is as follows:

   ```json
   {
     "template_title": {
       "title": "Template Title",
       "original_info": [
         {"pic_title": "image1.jpg", "other_info": "details"},
         {"pic_title": "image2.jpg", "other_info": "details"}
       ],
       "out_paths": [
         "template_examples/templates/folder1/image1.jpg",
         "template_examples/templates/folder1/image2.jpg"
       ]
     }
   }

#### Step 8
Run `org_examples.py` to get the template examples in the final format we use for our experiments.

# Template Example File Processing and Copying

This Python script processes template information, gathers meme examples, and copies them to new directories for further processing. It ensures that the necessary directories exist and handles potential errors such as missing files or permission issues. The script also logs its progress and provides a summary at the end.

## Code Explanation

### 1. **Imports and Helper Functions**
   - **Libraries Used:**
     - `json` for reading and writing JSON data.
     - `glob` for finding file paths that match a given pattern.
     - `os` for file and directory management.
     - `shutil` for file copying and moving.
     - `tqdm` for progress tracking during long operations.

   - **Helper Functions:**
     - **`create_directory(directory_path)`**:
       - Ensures the specified directory exists by creating it if it doesn't already.
     
     - **`copy_example_files(examples, write_path)`**:
       - Takes a list of example file paths and copies them to the specified destination (`write_path`).
       - Skips copying files that have an invalid path or cannot be found.
       - Logs progress every 1000 files copied.

### 2. **Main Template Processing Function**
   - **`process_template_info()`**:
     - The main function that processes template information and copies example meme files.
     - **File Paths and Directories:**
       - It looks for example files in two directories: `workstation_scrapes/meme_examples/` and `memes/meme_examples/`.
       - It creates the target directory for each template using the `out_paths` from the template metadata.
     
     - **Template Info Loading:**
       - The function loads the template information from `template_info.json`, which contains the metadata for each meme template.
     
     - **Example File Gathering:**
       - For each template, it gathers example file paths from the workstation and cluster directories.
       - The `glob.glob` method is used to gather all the example file paths matching the pattern.
     
     - **File Copying:**
       - After gathering the example files, the function calls `copy_example_files` to copy the files to the appropriate directory.
     
     - **Logging Progress:**
       - The script logs how many files have been copied during the process.
     
     - **Completion Summary:**
       - Once the process is complete, it prints a summary of how many files were copied.

### 3. **Error Handling**
   - The script handles errors such as:
     - **FileNotFoundError**: If an example file is not found, it skips that file and logs the error.
     - **PermissionError**: If there are permission issues while copying a file, it skips that file and logs the error.
     - **Invalid File Path**: If the path doesn't match the expected structure, the file is skipped.

### 4. **Logging**
   - Every 1000 files copied, the script logs the progress to inform the user of the current status.
   
### 5. **Script Execution**
   - The script iterates over the template information loaded from the JSON file.
   - For each template, it gathers the example files and copies them to the appropriate directory.
   - After processing all templates, it prints the total number of files copied.

### 6. **JSON Output Format**
   After processing, the script assumes the following structure for the `template_info.json`:

   ```json
   {
     "template_title": {
       "title": "Template Title",
       "original_info": [
         {"pic_title": "image1.jpg", "other_info": "details"},
         {"pic_title": "image2.jpg", "other_info": "details"}
       ],
       "out_paths": [
         "template_examples/templates/folder1/image1.jpg",
         "template_examples/templates/folder1/image2.jpg"
       ]
     }
   }

#### Step 9
Run `remove_dups.py` to get rid of redundant data.

# Duplicate File Removal Based on SHA1 Hashes

This Python script helps identify and remove duplicate files in a template directory based on their SHA1 hash. It processes the template information, computes the hash of each file, and removes any file that has the same hash as a previously encountered file. This ensures that only unique files are kept in the target directory.

## Code Explanation

### 1. **Imports**
   - **`hashlib`**: Used to compute the SHA1 hash of files, which is a common method for detecting duplicates based on file content.
   - **`glob`**: Used to gather file paths matching a specific pattern.
   - **`json`**: Used to load the template information from a JSON file.
   - **`os`**: Used for file operations like removing duplicates.

### 2. **Helper Function: `get_file_hash(file_path)`**
   - **Purpose**: Computes the SHA1 hash of a file.
   - **Parameters**: 
     - `file_path`: The path to the file whose hash is to be computed.
   - **Logic**:
     - Opens the file in binary read mode (`'rb'`).
     - Reads the file content and computes its SHA1 hash using `hashlib.sha1()`.
     - Returns the hash as a binary digest if successful; otherwise, prints an error message and returns `None`.

### 3. **Main Function: `remove_duplicate_files()`**
   - **Purpose**: Processes template files, computes their hashes, and removes duplicates based on these hashes.
   
   - **Steps**:
     1. **Load Template Information**: 
        - Reads the `template_info.json` file, which contains metadata about template files.
        - Loads the data into the `template_info` list.
        
     2. **Iterate Over Each Template**:
        - For each template in the `template_info`, the function prepares to check files in the corresponding `out_paths` directory.
        
     3. **Track Unique File Hashes**:
        - Initializes an empty set `hashes` to keep track of unique file hashes.
        
     4. **Check Each File**:
        - Uses `glob.glob(write_path)` to find all files in the target directory (defined by `write_path`).
        - For each file, it calculates its SHA1 hash using the `get_file_hash()` function.
        - If the hash has already been encountered (i.e., the file is a duplicate), it removes the file and prints a log message.
        - If the file is not a duplicate, it adds the hash to the set of seen hashes.
        
     5. **Completion**:
        - Once all files are processed, a message is printed indicating that the duplicate removal process is complete.

### 4. **Error Handling**
   - **File Hash Calculation**: If the script encounters an error while reading a file (e.g., file not found or permission error), it skips that file and continues with the next one.
   - **File Deletion**: If an error occurs while attempting to delete a file, it prints the error message but does not stop the process.

### 5. **Usage**
   This script can be run to ensure that only unique files are kept in the template directories, which is useful for saving disk space and maintaining a clean file structure.

### Example Command to Run the Script:

```bash
python remove_dups.py
