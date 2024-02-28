### Use our scraping code.
#### Step 1
Make sure you have installed the environment, in particular BeautifulSoup and Selenium.
#### Step 2
Run the code in `downloader.ipynb`. This will download and write to disk all the parent entries in the "confirmed" section of Know Your Meme.
#### Step 3
Run `parseandwrite.py`. This is look at all the parent entries in the "confirmed" section, download them, and write them to disk.
#### Step 4
Run `get_images.py`. This with find all the meme template download about and download the image/parse the page for the title/about section.`
This will take a very long time.
#### Step 5 (optional)
You can take a look at `read_images.py` for how we read in images/handle exceptions
#### Step 6
Run `get_examples.py` to visit each template page and download/write to disk all available examples. This will take a very, very long time.
#### Step 7
Run `organize_templates.py` to get the templates in the final format we used for our experiments.
#### Step 8
Run `organize_examples.py` to get the template examples in the final format we use for our experiments.
#### Step 9
Run `remove_dups.py`to get rid of redundant data.
