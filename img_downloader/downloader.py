import os
import requests
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import threading

# ===== CONFIG =====
INPUT_FILE = "new_papal_imgs.txt"
OUTPUT_DIR = "downloaded_images"
FAILED_FILE = "failed_downloads.txt"
MAX_WORKERS = 32
TIMEOUT = 15
RETRY_FAILED_PASS = True
MAX_IMAGES = None   # Set to None to download all
# ==================

os.makedirs(OUTPUT_DIR, exist_ok=True)

failed_urls = []
failed_lock = threading.Lock()

# Create persistent session with retries
def create_session():
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(
        max_retries=retries,
        pool_connections=100,
        pool_maxsize=100,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

session = create_session()

def is_valid_image(content):
    try:
        Image.open(BytesIO(content)).verify()
        return True
    except Exception:
        return False

def download_image(url):
    try:
        filename = os.path.basename(urlparse(url).path)

        if not filename:
            return False

        save_path = os.path.join(OUTPUT_DIR, filename)

        # Resume support: skip if already exists
        if os.path.exists(save_path):
            return True

        response = session.get(url, timeout=TIMEOUT)
        response.raise_for_status()

        content = response.content

        # Validate image
        if not is_valid_image(content):
            raise ValueError("Invalid image file")

        with open(save_path, "wb") as f:
            f.write(content)

        return True

    except Exception:
        with failed_lock:
            failed_urls.append(url)
        return False


def run_download(urls, pass_name="Main pass"):
    print(f"\nStarting {pass_name} ({len(urls)} images)")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(download_image, url) for url in urls]

        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass


# Load URLs
with open(INPUT_FILE, "r") as f:
    urls = [line.strip() for line in f if line.strip()]

# Apply optional limit
if MAX_IMAGES is not None:
    urls = urls[:MAX_IMAGES]
    print(f"Limiting download to first {len(urls)} images.")

# MAIN PASS
run_download(urls, "Main pass")

# RETRY PASS
if RETRY_FAILED_PASS and failed_urls:
    retry_urls = failed_urls.copy()
    failed_urls.clear()
    run_download(retry_urls, "Retry pass")

# Save final failed list
if failed_urls:
    with open(FAILED_FILE, "w") as f:
        for url in failed_urls:
            f.write(url + "\n")

print("\nDownload complete.")
print(f"Total URLs: {len(urls)}")
print(f"Failed after retry: {len(failed_urls)}")
