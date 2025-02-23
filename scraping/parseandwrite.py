import requests
from bs4 import BeautifulSoup
import json
import time
import os
from tqdm import tqdm

# Define directories
snapshots_dir = "selenium_snapshots"
jsons_dir = "jsons"
os.makedirs(jsons_dir, exist_ok=True)

# Load previously saved meme tables
meme_table_path = os.path.join(jsons_dir, "meme_tables.json")
seen_urls = set()
meme_count = 0

if os.path.isfile(meme_table_path):
    with open(meme_table_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            seen_urls.add(entry["url"])
            meme_count = max(meme_count, entry["count"])

# Load missed URLs
missed_path = os.path.join(jsons_dir, "miss.json")
missed_count = sum(1 for _ in open(missed_path, "r")) if os.path.isfile(missed_path) else 0

# Function to get the closest Wayback Machine snapshot
def get_wayback_snapshot(url):
    api_url = f"http://archive.org/wayback/available?url={url}"
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException:
        print(f"Request failed for {url}. Retrying in 10 minutes...")
        time.sleep(600)
        response = requests.get(api_url)
        data = response.json()

    if "closest" in data.get("archived_snapshots", {}):
        snapshot = data["archived_snapshots"]["closest"]
        if snapshot.get("available") and snapshot.get("url"):
            return snapshot
    return None

# Loop over all HTML files in selenium_snapshots/
html_files = [os.path.join(snapshots_dir, f) for f in os.listdir(snapshots_dir) if f.endswith(".html")]
meme_urls = set()

# Load and parse an example file
test_file = html_files[0]  # Pick the first file in selenium_snapshots
with open(test_file, "rb") as f:
    soup = BeautifulSoup(f, "html.parser")

# Print the full parsed page
print(soup.prettify())  

# Find all links
all_links = soup.find_all("a", href=True)
print('pretty soup')

# Print first 10 links for inspection
for link in all_links[:10]:
    print(link)

# Process HTML files
for html_file in tqdm(html_files, desc="Processing HTML pages"):
    with open(html_file, "rb") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Extract meme URLs from <img> tags with the 'data-data-entry-name' attribute
    meme_images = soup.find_all('img', {'data-data-entry-name': True})

    if meme_images:
        for img in meme_images:
            meme_url = img['src']  # Extract the image URL from the 'src' attribute
            if meme_url.startswith("http"):
                meme_urls.add(meme_url)  # Add the valid meme URL
            else:
                print(f"Invalid URL: {meme_url}")
    
# **Check if meme_urls were found**
if not meme_urls:
    print("⚠️ No meme URLs found in any of the HTML files. Check if the structure of the page has changed!")
else:
    print(f"✅ Found {len(meme_urls)} meme URLs.")

# Remove already processed URLs
meme_urls -= seen_urls

# **Check if there's anything left to process**
if not meme_urls:
    print("⚠️ No new meme URLs to process. Everything has already been processed.")
else:
    print(f"✅ Processing {len(meme_urls)} new meme URLs...")

# Process each meme URL
for meme_url in tqdm(meme_urls, desc="Fetching Wayback Snapshots"):
    snapshot = get_wayback_snapshot(meme_url)

    if snapshot:
        timestamp = snapshot["timestamp"]
        snapshot_url = snapshot["url"]

        outdir = os.path.join(snapshots_dir, timestamp)
        os.makedirs(outdir, exist_ok=True)

        try:
            response = requests.get(snapshot_url)
            response.raise_for_status()
        except requests.exceptions.RequestException:
            print(f"Failed to fetch {snapshot_url}. Retrying in 10 minutes...")
            time.sleep(600)
            response = requests.get(snapshot_url)

        meme_count += 1
        filename = os.path.join(outdir, f"{meme_count}_tablecontent.html")
        with open(filename, "wb") as f:
            f.write(response.content)

        # **Write JSON immediately to ensure updates are saved**
        with open(meme_table_path, "a") as f:
            json.dump({"url": snapshot_url, "count": meme_count}, f)
            f.write("\n")

        print(f"✅ Saved: {filename}")
        time.sleep(1)

    else:
        missed_count += 1
        with open(missed_path, "a") as f:
            json.dump({"url": meme_url, "count": missed_count}, f)
            f.write("\n")
