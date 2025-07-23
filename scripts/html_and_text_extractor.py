import asyncio
import os
import json
from urllib.parse import urlparse
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd

SAVE_DIR = "scripts/extracted_website_data_for_feature_engineering"
# Check before doing anything else
if os.path.exists(SAVE_DIR) and len(os.listdir(SAVE_DIR)) > 0:
    print(f"[!] Directory '{SAVE_DIR}' already contains extracted data. Aborting to prevent overwrite.")
    exit(0)
else:
    os.makedirs(SAVE_DIR, exist_ok=True)

CONCURRENT_TASKS = 5  # Set this based on your system capabilities, 3 for starter
semaphore = asyncio.Semaphore(CONCURRENT_TASKS)

def sanitize_filename(url):
    parsed = urlparse(url)
    netloc = parsed.netloc.replace(".", "_")
    path = parsed.path.replace("/", "_")
    return f"{netloc}{path}.json"

def normalize_url(url):
    if not url.startswith(("http://", "https://")):
        return "https://" + url
    return url

async def extract_and_save(playwright, url):
    async with semaphore:
        try:
            browser = await playwright.firefox.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/114.0.0.0 Safari/537.36",
                viewport={"width": 1280, "height": 800},
                java_script_enabled=True
            )
            page = await context.new_page()
            url = normalize_url(url)
            await page.goto(url, timeout=60000)
            await page.wait_for_timeout(5000)

            html_content = await page.content()
            soup = BeautifulSoup(html_content, "html.parser")
            text_content = soup.get_text(separator=' ', strip=True)

            await context.close()
            await browser.close()

            data = {
                "url": url,
                "html": html_content,
                "text": text_content
            }
            filename = sanitize_filename(url)
            filepath = os.path.join(SAVE_DIR, filename)
            with open(filepath, "w", encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"[!] Failed to process {url}: {e}")

async def process_urls_concurrently(url_list):
    async with async_playwright() as playwright:
        tasks = [extract_and_save(playwright, url) for url in url_list]
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            await f

if __name__ == "__main__":
    ecommerce_urls = pd.read_csv('scripts/e-commerce_websites.csv')['URL'].to_list()
    nonecommerce_urls = pd.read_csv('scripts/non-ecommerce_websites.csv')['URL'].to_list()
    url_list = ecommerce_urls + nonecommerce_urls
    asyncio.run(process_urls_concurrently(url_list))
