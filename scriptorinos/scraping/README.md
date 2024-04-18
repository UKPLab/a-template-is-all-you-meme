### What we did to create the KYMKB.
#### Step 1
We installed the environment, in particular BeautifulSoup and Selenium.
#### Step 2
We ran the code in `downloader.ipynb` for crawling all the parent entries in the "confirmed" section of Know Your Meme.
#### Step 3
Ran `parseandwrite.py`. This looked at parent entries in the "confirmed" section and crawled them.
#### Step 4
Ran `get_images.py`. This with found all the meme templates abouts and the image/parse the page for the title/about section.
This took take a very long time.
#### Step 5 (optional)
We used `read_images.py` to images/handle exceptions
#### Step 6
We ran `get_examples.py` to visit each template page and crawl all available examples. This took a very, very long time.
#### Step 7
We ran `organize_templates.py` to format templates for our experiments.
#### Step 8
We ran `organize_examples.py` to format template examples for our experiments.
#### Step 9
We ran `remove_dups.py` for redundancies.
