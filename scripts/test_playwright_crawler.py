import asyncio
import os
import hashlib
from urllib.parse import urlparse, urljoin
from datetime import datetime
from playwright.async_api import async_playwright
import json
import aiohttp
import re

# Config
OUTPUT_DIR = "C:/Users/mmia43/Desktop/Longitudinal Study Research/Fradulent_E-Commerce_Website_Data/"
VIEWPORT = {"width": 1280, "height": 960}
HEADLESS = True
TIMEOUT = 120000  # 2 minutes for page navigation
DOWNLOAD_TIMEOUT = 30 # seconds for individual file downloads
MAX_JS_FILES = 10
MAX_SUSPICIOUS_FILES = 10
MAX_IMAGES_TO_DOWNLOAD = 10 # Variable for number of images

SUSPICIOUS_EXTENSIONS = {".exe", ".bin", ".dll", ".scr", ".apk", ".vbs", ".rb", ".pl", ".py", ".run", ".sh", ".pkg", ".dmg", ".app", ".pif", ".cpl", ".jse", ".ps1", ".cmd", ".bat", ".msi"}

def hash_string(value: str) -> str:
    """Generates a SHA256 hash for a given string."""
    return hashlib.sha256(value.encode()).hexdigest()

def get_fqdn_and_hash(url: str):
    """Extracts FQDN from a URL and returns its hash."""
    fqdn = urlparse(url).netloc
    return fqdn, hash_string(fqdn)

def get_formatted_timestamp():
    """Returns a formatted timestamp string."""
    now = datetime.now()
    return f"{now.year}_{now.month:02d}_{now.day:02d}_{now.hour:02d}_{now.minute:02d}"

def get_filename_from_url(url, default_name):
    """
    Extracts a filename from a URL. If no clear filename, uses default_name.
    Ensures filename is safe for file systems.
    """
    path = urlparse(url).path
    filename = os.path.basename(path)
    if not filename or filename.endswith("/"):
        filename = default_name
    
    # Sanitize filename (remove/replace invalid characters)
    filename = re.sub(r'[\\/:*?"<>|]', '_', filename) # Replace forbidden chars with underscore
    filename = filename.strip() # Remove leading/trailing whitespace
    return filename

class WebsiteCrawler:
    """
    A class to crawl websites, capture screenshots, HTML, text, and download
    various associated files (JS, images, suspicious files).
    Manages a single Playwright browser instance for efficiency across multiple crawls.
    """
    def __init__(self, output_dir=OUTPUT_DIR, viewport=VIEWPORT, headless=HEADLESS, 
                 timeout=TIMEOUT, download_timeout=DOWNLOAD_TIMEOUT, 
                 max_js_files=MAX_JS_FILES, max_suspicious_files=MAX_SUSPICIOUS_FILES,
                 max_images_to_download=MAX_IMAGES_TO_DOWNLOAD):
        
        self.output_dir = output_dir
        self.viewport = viewport
        self.headless = headless
        self.timeout = timeout
        self.download_timeout = download_timeout
        self.max_js_files = max_js_files
        self.max_suspicious_files = max_suspicious_files
        self.max_images_to_download = max_images_to_download

        self.suspicious_extensions = SUSPICIOUS_EXTENSIONS # Use the global config

        self.browser = None  # Playwright browser instance
        self.context = None  # Playwright browser context
        self.playwright_instance = None # To store the playwright context manager

        # Initialize aiohttp session for file downloads.
        # It's better to manage this session per class instance as well.
        self.session = None 

    async def initialize(self):
        """Initializes the Playwright browser, context, and aiohttp session."""
        if self.browser is None:
            print("Initializing Playwright browser...")
            self.playwright_instance = await async_playwright().start()
            self.browser = await self.playwright_instance.chromium.launch(headless=self.headless)
            self.context = await self.browser.new_context(
                viewport=self.viewport,
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36"
            )
            print("Playwright browser initialized.")
        
        if self.session is None:
            print("Initializing aiohttp client session...")
            self.session = aiohttp.ClientSession()
            print("aiohttp client session initialized.")

    async def close(self):
        """Closes the Playwright browser and aiohttp session."""
        if self.browser:
            print("Closing Playwright browser...")
            await self.browser.close()
            if self.playwright_instance:
                await self.playwright_instance.stop()
            self.browser = None
            self.context = None
            self.playwright_instance = None
            print("Playwright browser closed.")
        
        if self.session:
            print("Closing aiohttp client session...")
            await self.session.close()
            self.session = None
            print("aiohttp client session closed.")

    async def _download_file(self, url, save_path):
        """Internal helper to download a single file using the shared aiohttp session."""
        try:
            async with self.session.get(url, timeout=self.download_timeout) as resp:
                if resp.status == 200:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    with open(save_path, "wb") as f:
                        f.write(await resp.read())
                    # print(f"Downloaded: {url} to {save_path}")
                else:
                    print(f"⚠️ Failed to download {url}: HTTP Status {resp.status}")
        except aiohttp.ClientError as e:
            print(f"❌ Client error downloading file {url}: {e}")
        except asyncio.TimeoutError:
            print(f"❌ Timeout downloading file {url}")
        except Exception as e:
            print(f"❌ General error downloading file {url}: {e}")

    async def crawl_url(self, url: str):
        """
        Crawls a single URL, collects data, and saves it to a structured directory.
        Uses the shared Playwright browser instance and aiohttp session.
        """
        fqdn, fqdn_hash = get_fqdn_and_hash(url)
        timestamp = get_formatted_timestamp()
        save_path = os.path.join(self.output_dir, fqdn_hash, timestamp)
        os.makedirs(save_path, exist_ok=True)

        js_folder = os.path.join(save_path, "js_scripts")
        suspicious_folder = os.path.join(save_path, "suspicious_files")
        images_folder = os.path.join(save_path, "images")

        os.makedirs(js_folder, exist_ok=True)
        os.makedirs(suspicious_folder, exist_ok=True)
        os.makedirs(images_folder, exist_ok=True)

        if self.browser is None or self.context is None or self.session is None:
            print("Crawler not initialized. Calling initialize()...")
            await self.initialize() # Ensure it's initialized if somehow not already

        page = await self.context.new_page() # Create a new page for this URL
        logs = []
        page.on("console", lambda msg: logs.append(f"[{msg.type}] {msg.text}"))

        try:
            response = await page.goto(url, wait_until="networkidle", timeout=self.timeout)
            if not response:
                print(f"No response received for {url}.")
                return

            await page.wait_for_selector("body", state="visible", timeout=30000)
            await asyncio.sleep(5) # Give some time for dynamic content to load

            await page.screenshot(path=os.path.join(save_path, "screenshot.png"))

            rendered_html = await page.content()
            with open(os.path.join(save_path, "rendered_html.html"), "w", encoding="utf-8") as f:
                f.write(rendered_html)

            text = await page.inner_text("body")
            with open(os.path.join(save_path, "text.txt"), "w", encoding="utf-8") as f:
                f.write(text)

            with open(os.path.join(save_path, "response_headers.json"), "w") as f:
                json.dump(response.headers, f, indent=2)
            with open(os.path.join(save_path, "request_headers.json"), "w") as f:
                json.dump(response.request.headers, f, indent=2)

            csp = response.headers.get("content-security-policy", "")
            with open(os.path.join(save_path, "csp_headers.txt"), "w", encoding="utf-8") as f:
                f.write(csp)

            cookies = await self.context.cookies() # Get cookies from the context
            with open(os.path.join(save_path, "cookies.json"), "w") as f:
                json.dump(cookies, f, indent=2)

            with open(os.path.join(save_path, "console_logs.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(logs))

            links = await page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
            internal = [l for l in links if urlparse(l).netloc == fqdn]
            external = [l for l in links if urlparse(l).netloc != fqdn]

            with open(os.path.join(save_path, "internal_links.txt"), "w") as f:
                f.write("\n".join(internal))
            with open(os.path.join(save_path, "external_links.txt"), "w") as f:
                f.write("\n".join(external))

            script_urls = await page.eval_on_selector_all(
                "script[src]", 
                "els => els.map(e => e.src).filter(src => src && (!src.startsWith('http') || src.includes(location.hostname)))"
            )
            # Filter internal scripts
            script_urls = [url for url in script_urls if urlparse(url).netloc == fqdn]
            script_urls = script_urls[:self.max_js_files]

            for idx, js_url in enumerate(script_urls):
                filename = get_filename_from_url(js_url, f"script_{idx + 1}.js")
                save_js_path = os.path.join(js_folder, filename)
                await self._download_file(js_url, save_js_path)

            # Download Images
            image_urls = await page.eval_on_selector_all(
                "img[src]",
                "els => els.map(e => e.src).filter(src => src)"
            )
            base_url_for_images = response.url
            absolute_image_urls = []
            for img_url in image_urls:
                absolute_img_url = urljoin(base_url_for_images, img_url)
                if urlparse(absolute_img_url).netloc == fqdn:
                    absolute_image_urls.append(absolute_img_url)
            
            images_to_download = absolute_image_urls[:self.max_images_to_download]

            print(f"Attempting to download {len(images_to_download)} images for {url}...")
            for idx, img_url in enumerate(images_to_download):
                img_filename_raw = get_filename_from_url(img_url, f"image_{idx + 1}.png")
                if not os.path.splitext(img_filename_raw)[1]:
                    img_filename_raw += ".png"
                
                final_img_filename = f"{idx + 1}_{img_filename_raw}"
                save_img_path = os.path.join(images_folder, final_img_filename)
                await self._download_file(img_url, save_img_path)

            suspicious_urls = []
            for ext in self.suspicious_extensions:
                # Select both src and href for suspicious extensions
                selector = f'[src$="{ext}" i], [href$="{ext}" i]' # 'i' for case-insensitive
                found_urls = await page.eval_on_selector_all(selector, "els => els.map(e => e.src || e.href).filter(Boolean)")
                suspicious_urls.extend(found_urls)

            # Filter for internal suspicious files and limit
            suspicious_urls = [url for url in suspicious_urls if urlparse(url).netloc == fqdn][:self.max_suspicious_files]

            for idx, sus_url in enumerate(suspicious_urls):
                filename = get_filename_from_url(sus_url, f"suspicious_{idx + 1}{os.path.splitext(sus_url)[1]}")
                save_sus_path = os.path.join(suspicious_folder, filename)
                await self._download_file(sus_url, save_sus_path)

            metadata = {
                "original_url": url,
                "fqdn": fqdn,
                "fqdn_hash": fqdn_hash,
                "timestamp": timestamp,
                "js_files_count": len(script_urls),
                "suspicious_files_count": len(suspicious_urls),
                "images_downloaded_count": len(images_to_download)
            }
            with open(os.path.join(save_path, "metadata.txt"), "w", encoding="utf-8") as f:
                for k, v in metadata.items():
                    f.write(f"{k}: {v}\n")

            print(f"✅ Successfully crawled: {url}")
            print(f"📁 Data saved to: {save_path}")

        except Exception as e:
            print(f"❌ Error crawling {url}: {e}")

        finally:
            await page.close() # Close the page, but keep browser/context open

async def main(urls, concurrency=5):
    """
    Main function to orchestrate crawling of multiple URLs with concurrency.
    Manages a single WebsiteCrawler instance.
    """
    # Ensure the base output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Instantiate the crawler class
    crawler = WebsiteCrawler(output_dir=OUTPUT_DIR, headless=HEADLESS, 
                             max_images_to_download=MAX_IMAGES_TO_DOWNLOAD)

    # Initialize the single browser and aiohttp session
    await crawler.initialize()

    semaphore = asyncio.Semaphore(concurrency)

    async def sem_crawl(url):
        async with semaphore:
            await crawler.crawl_url(url) # Call the method on the class instance

    tasks = [sem_crawl(url) for url in urls]
    await asyncio.gather(*tasks)

    # Close the single browser and aiohttp session when all tasks are done
    await crawler.close()
    print("\nAll crawling tasks completed and resources released.")

# --- Entry Point ---
if __name__ == "__main__":
    URLS_TO_TEST = [     
        "https://birthbday.com/"
    ]
    max_urls_for_parallel_process = 5

    asyncio.run(main(URLS_TO_TEST, concurrency=max_urls_for_parallel_process))