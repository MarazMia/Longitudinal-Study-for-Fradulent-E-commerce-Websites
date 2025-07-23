# import asyncio
# import pickle
# from datetime import datetime
# from urllib.parse import urlparse
# from bs4 import BeautifulSoup
# from playwright.async_api import async_playwright
# from html_and_text_feature_extractor import extract_all_features
# import pandas as pd
# import re

# # Load ML assets
# with open("scripts/tfidf_vectorizer.pkl", "rb") as f:
#     tfidf_vectorizer = pickle.load(f)

# with open("scripts/top_keywords.pkl", "rb") as f:
#     top_keywords = pickle.load(f)

# with open("scripts/xgb_ecommerce_detector_voting.pkl", "rb") as f:
#     model = pickle.load(f)

# with open("scripts/feature_scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)

# # Load the expected feature column order
# with open("scripts/feature_column_order.pkl", "rb") as f:
#     expected_columns = pickle.load(f)

# # Config
# MAX_INTERNAL_PAGES = 10
# PAGE_CATEGORIES = {
#     "about": {"our story", "who we are", "mission", "team", "company history", "about us"},
#     "contact": {"email us", "contact form", "get in touch", "phone number", "reach us", "contact us"},
#     "payment": {"payment methods", "credit card", "checkout", "paypal", "billing", "secure payment"},
#     "home": {"welcome", "featured products", "latest arrivals"},
#     "faq": {"frequently asked questions", "how do i", "support topics", "faq"},
#     "shipping": {"shipping policy", "delivery time", "tracking number", "shipping info"}
# }

# def normalize_url(url):
#     return url if url.startswith(("http://", "https://")) else "https://" + url

# def get_fqdn(url):
#     return urlparse(url).netloc

# def filter_priority_links(all_links, fqdn, original_url):
#     priority_keywords = {
#         "about": ["about", "about-us", "aboutus", "chisiamo", "uberuns", "sobrenosotros", "sobrenos"],
#         "contact": ["contact", "contact-us", "kontaktieren", "contactez", "kontakt", "contattaci"],
#         "payment": ["payment", "checkout", "pagamento", "zahlung", "kasse", "paiement", "pago"]
#     }

#     internal_links = list(set([link for link in all_links if urlparse(link).netloc == fqdn]))
#     selected = set()

#     fqdn_home = f"https://{fqdn}/"
#     if original_url.rstrip("/") != fqdn_home.rstrip("/"):
#         selected.add(fqdn_home)

#     for keywords in priority_keywords.values():
#         for keyword in keywords:
#             for link in internal_links:
#                 if re.search(rf"\\b{keyword}\\b", link.lower()) and link not in selected:
#                     selected.add(link)
#                     break

#     selected.add(original_url)

#     while len(selected) < MAX_INTERNAL_PAGES and internal_links:
#         next_link = internal_links.pop()
#         if next_link not in selected:
#             selected.add(next_link)

#     # Convert to list preserving order and apply the cleaning logic:
#     unique_pages = list(dict.fromkeys(selected))

#     # Remove the base URL itself if present
#     unique_pages = [p for p in unique_pages if p.rstrip('/') != original_url.rstrip('/')]

#     # Include fqdn root only if different from original_url
#     if fqdn_home.rstrip('/') != original_url.rstrip('/'):
#         if fqdn_home not in unique_pages:
#             unique_pages.insert(0, fqdn_home)
#     else:
#         # Remove fqdn_home if same as original_url
#         # unique_pages = [p for p in unique_pages if p.rstrip('/') != fqdn_home.rstrip('/')]
#         pass

#     # Limit to MAX_INTERNAL_PAGES (usually 10)
#     return unique_pages[:MAX_INTERNAL_PAGES]

# async def fetch_html_and_links(url):
#     async with async_playwright() as p:
#         browser = await p.chromium.launch(headless=True)
#         context = await browser.new_context(
#             user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
#                        "AppleWebKit/537.36 (KHTML, like Gecko) "
#                        "Chrome/114.0.0.0 Safari/537.36"
#         )
#         page = await context.new_page()
#         try:
#             await page.goto(url, timeout=30000)
#             await page.wait_for_timeout(3000)
#             html = await page.content()
#             all_links = await page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
#         except Exception as e:
#             print(f"[!] Error fetching {url}: {e}")
#             html, all_links = "", []
#         await browser.close()
#         return html, all_links

# def classify_new_url_with_ml(url: str, blacklist_name="unknown"):
#     url = normalize_url(url)
#     fqdn = get_fqdn(url)
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     html, all_links = asyncio.run(fetch_html_and_links(url))
#     if not html.strip():
#         return {
#             "url": url,
#             "fqdn": fqdn,
#             "initial_URL_Entry_time": timestamp,
#             "blacklist_name": blacklist_name,
#             "selected_Internal_pages_list": [],
#             "error": "Empty HTML content"
#         }

#     soup = BeautifulSoup(html, "html.parser")
#     text = soup.get_text(separator=" ", strip=True)

#     features_dict = extract_all_features(html, text, tfidf_vectorizer, top_keywords)
#     selected_pages = filter_priority_links(all_links, fqdn, url)

#     features_df = pd.DataFrame([features_dict])

#     # Add missing columns as 0
#     for col in expected_columns:
#         if col not in features_df.columns:
#             features_df[col] = 0

#     # Drop unexpected columns and reorder
#     features_df = features_df[expected_columns] 

#     features_scaled = scaler.transform(features_df)
#     pred = model.predict(features_scaled)[0]
#     proba = model.predict_proba(features_scaled)[0][1] if hasattr(model, "predict_proba") else None

#     return {
#         "url": url,
#         "fqdn": fqdn,
#         "initial_URL_Entry_time": timestamp,
#         "blacklist_name": blacklist_name,
#         "selected_pages_to_crawl": selected_pages,
#         "is_ecommerce": "yes" if pred == 1 else "no",
#         "confidence_score": round(proba, 3) if proba is not None else "N/A"
#     }



# print(classify_new_url_with_ml('https://smishtank.com/'))




import asyncio
import pickle
from datetime import datetime
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from html_and_text_feature_extractor import extract_all_features
import pandas as pd
import re
import os
import json
import warnings # This is the correct import for warning control

class URLClassifier:
    """
    A class to classify URLs as e-commerce or not using a pre-trained ML model.
    It handles web scraping, feature extraction, and prediction.

    ML assets are loaded once when the object is instantiated to optimize performance
    for classifying multiple URLs. A single Playwright browser instance is also
    managed to reduce overhead.
    """
    def __init__(self, scripts_dir="scripts"):
        """
        Initializes the URLClassifier by loading all necessary ML assets and
        starting a single Playwright browser instance.

        Args:
            scripts_dir (str): The directory where ML model assets are stored.
        """
        self.scripts_dir = scripts_dir
        self.browser = None  # Will hold the Playwright browser instance
        self.context = None  # Will hold the Playwright browser context
        self.playwright_instance = None # To store the playwright context manager

        self._load_ml_assets()

        self.MAX_INTERNAL_PAGES = 10
        self.PAGE_CATEGORIES = {
            "about": {"our story", "who we are", "mission", "team", "company history", "about us"},
            "contact": {"email us", "contact form", "get in touch", "phone number", "reach us", "contact us"},
            "payment": {"payment methods", "credit card", "checkout", "paypal", "billing", "secure payment"},
            "home": {"welcome", "featured products", "latest arrivals"},
            "faq": {"frequently asked questions", "how do i", "support topics", "faq"},
            "shipping": {"shipping policy", "delivery time", "tracking number", "shipping info"}
        }

    async def _initialize_browser(self):
        """Initializes the Playwright browser and context."""
        if self.browser is None:
            # Start the async_playwright context manager explicitly
            self.playwright_instance = await async_playwright().start()
            self.browser = await self.playwright_instance.chromium.launch(headless=True)
            self.context = await self.browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/114.0.0.0 Safari/537.36"
            )
            print("Playwright browser initialized.")

    async def close_browser(self):
        """Closes the Playwright browser instance."""
        if self.browser:
            await self.browser.close()
            # Stop the playwright context manager
            if self.playwright_instance:
                await self.playwright_instance.stop()
            self.browser = None
            self.context = None
            self.playwright_instance = None
            print("Playwright browser closed.")

    def _load_ml_assets(self):
        """Loads all pre-trained machine learning assets."""
        with open(os.path.join(self.scripts_dir, "tfidf_vectorizer.pkl"), "rb") as f:
            self.tfidf_vectorizer = pickle.load(f)

        with open(os.path.join(self.scripts_dir, "top_keywords.pkl"), "rb") as f:
            self.top_keywords = pickle.load(f)

        with open(os.path.join(self.scripts_dir, "xgb_ecommerce_detector_voting.pkl"), "rb") as f:
            self.model = pickle.load(f)

        with open(os.path.join(self.scripts_dir, "feature_scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)

        with open(os.path.join(self.scripts_dir, "feature_column_order.pkl"), "rb") as f:
            self.expected_columns = pickle.load(f)
        print("ML assets loaded successfully.")

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Ensures the URL has a scheme (http/https)."""
        return url if url.startswith(("http://", "https://")) else "https://" + url

    @staticmethod
    def _get_fqdn(url: str) -> str:
        """Extracts the Fully Qualified Domain Name (FQDN) from a URL."""
        return urlparse(url).netloc

    def _filter_priority_links(self, all_links: list, fqdn: str, original_url: str) -> list:
        priority_keywords = {
            "about": ["about", "about-us", "aboutus", "chisiamo", "uberuns", "sobrenosotros", "sobrenos"],
            "contact": ["contact", "contact-us", "kontaktieren", "contactez", "kontakt", "contattaci"],
            "payment": ["payment", "checkout", "pagamento", "zahlung", "kasse", "paiement", "pago"]
        }

        internal_links = list(set([link for link in all_links if urlparse(link).netloc == fqdn]))
        selected = set()

        fqdn_home = f"https://{fqdn}/"
        if original_url.rstrip("/") != fqdn_home.rstrip("/"):
            selected.add(fqdn_home)

        for keywords in priority_keywords.values():
            for keyword in keywords:
                for link in internal_links:
                    if re.search(rf"\b{keyword}\b", link.lower()) and link not in selected:
                        selected.add(link)
                        break

        selected.add(original_url)

        temp_internal_links = [link for link in internal_links if link not in selected]
        while len(selected) < self.MAX_INTERNAL_PAGES and temp_internal_links:
            next_link = temp_internal_links.pop(0)
            if next_link not in selected:
                selected.add(next_link)

        unique_ordered_pages = list(dict.fromkeys(selected))

        final_pages = []
        if fqdn_home.rstrip('/') != original_url.rstrip('/'):
            if fqdn_home not in final_pages:
                final_pages.append(fqdn_home)

        if original_url not in final_pages:
            final_pages.append(original_url)

        for p in unique_ordered_pages:
            if p.rstrip('/') != fqdn_home.rstrip('/') and p.rstrip('/') != original_url.rstrip('/'):
                final_pages.append(p)

        return final_pages[:self.MAX_INTERNAL_PAGES]

    async def _fetch_html_and_links(self, url: str) -> tuple[str, list]:
        """
        Asynchronously fetches HTML content and all links from a given URL using
        the shared Playwright browser instance.
        """
        if self.browser is None or self.context is None:
            await self._initialize_browser() # Ensure browser is initialized

        page = await self.context.new_page()
        try:
            await page.goto(url, timeout=30000)
            await page.wait_for_timeout(5000)
            html = await page.content()
            all_links = await page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
        except Exception as e:
            print(f"[!] Error fetching {url}: {e}")
            html, all_links = "", []
        finally:
            await page.close()
        return html, all_links

    async def classify_url_async(self, url: str, blacklist_name: str = "unknown") -> dict:
        """
        Asynchronously classifies a single URL to determine if it's an e-commerce site.
        """
        url = self._normalize_url(url)
        fqdn = self._get_fqdn(url)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html, all_links = await self._fetch_html_and_links(url)

        if not html.strip():
            return {
                "url": url,
                "fqdn": fqdn,
                "initial_URL_Entry_time": timestamp,
                "blacklist_name": blacklist_name,
                "selected_pages_to_crawl": [],
                "is_ecommerce": 0,
                "confidence_score": 0.0,
                "error": "Empty HTML content or failed to fetch"
            }

        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator=" ", strip=True)

        features_dict = extract_all_features(html, text, self.tfidf_vectorizer, self.top_keywords)
        selected_pages = self._filter_priority_links(all_links, fqdn, url)

        features_df = pd.DataFrame([features_dict])

        for col in self.expected_columns:
            if col not in features_df.columns:
                features_df[col] = 0

        features_df = features_df[self.expected_columns] 

        # Suppress the UserWarning from scikit-learn regarding feature names
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning) # Use built-in UserWarning
            features_scaled = self.scaler.transform(features_df)
            pred = self.model.predict(features_scaled)[0]
            proba = self.model.predict_proba(features_scaled)[0][1] if hasattr(self.model, "predict_proba") else None

        return {
            "url": url,
            "fqdn": fqdn,
            "initial_URL_Entry_time": timestamp,
            "blacklist_name": blacklist_name,
            "selected_pages_to_crawl": selected_pages,
            "is_ecommerce": "yes" if pred == 1 else "no",
            "confidence_score": round(proba, 3) if proba is not None else 0.0
        }

    def classify_url(self, url: str, blacklist_name: str = "unknown") -> dict:
        """
        Synchronous wrapper for the asynchronous classification method.
        This method will handle the async context for you, including browser initialization
        and closure if used for a single call, but it's less efficient for multiple calls.
        """
        async def _run_and_close():
            # If the classifier wasn't initialized in an async loop, do it here
            if self.browser is None:
                await self._initialize_browser()
            result = await self.classify_url_async(url, blacklist_name)
            return result
        
        return asyncio.run(_run_and_close())


async def main():
    print("Application Start: Initializing URLClassifier...")
    url_classifier = URLClassifier(scripts_dir="scripts")

    # IMPORTANT: Initialize the Playwright browser instance after creating the classifier object.
    # This must be awaited because Playwright operations are async.
    await url_classifier._initialize_browser()
    print("URLClassifier and Playwright browser ready for use.\n")

    urls_to_check = [
        'https://smishtank.com/',
        'https://www.nike.com/',          
        'https://www.wikipedia.org/'
    ]

    for url in urls_to_check:
        print(f"\nProcessing URL: {url}")
        result = await url_classifier.classify_url_async(url)
        print(json.dumps(result, indent=4))

    await url_classifier.close_browser()
    print("\nApplication End: Playwright browser closed.")

if __name__ == "__main__":
    if not os.path.exists("scripts"):
        print("Error: 'scripts' directory not found. Please ensure it exists and contains your .pkl files.")
    else:
        asyncio.run(main())