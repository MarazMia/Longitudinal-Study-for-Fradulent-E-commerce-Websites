import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import asyncio
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np

class EcommerceDetector:
    def __init__(self):
        # E-commerce content patterns
        self.ecommerce_patterns = {
            'shopping_cart': re.compile(r'shopping[-\s]?cart|cart|basket', re.IGNORECASE),
            'checkout': re.compile(r'check\s?out|proceed to payment', re.IGNORECASE),
            'add_to_cart': re.compile(r'add\s?(to|2)\s?(cart|basket|bag)', re.IGNORECASE),
            'product_page': re.compile(r'product\s?(details|info|description|page)', re.IGNORECASE),
            'price_display': re.compile(r'\$\d+\.\d{2}|\d+\s?(USD|EUR|GBP)', re.IGNORECASE),
            'quantity_selector': re.compile(r'quantity|qty|how many', re.IGNORECASE),
            'inventory_status': re.compile(r'in stock|out of stock|only \d+ left', re.IGNORECASE),
            'product_variants': re.compile(r'select (size|color|option|variant)', re.IGNORECASE),
            'shipping_info': re.compile(r'shipping|delivery|estimated arrival', re.IGNORECASE),
            'return_policy': re.compile(r'return policy|free returns|\d+-day return', re.IGNORECASE)
        }

        self.platform_signatures = {
            'shopify': re.compile(r'shopify', re.IGNORECASE),
            'woocommerce': re.compile(r'woocommerce', re.IGNORECASE),
            'magento': re.compile(r'magento', re.IGNORECASE),
            'bigcommerce': re.compile(r'bigcommerce', re.IGNORECASE),
            'cart_scripts': re.compile(r'/cart\.js|/cart/add\.js', re.IGNORECASE),
            'opencart': re.compile(r'opencart', re.IGNORECASE),
            'squarespace': re.compile(r'squarespace\.com', re.IGNORECASE),
            'wix': re.compile(r'wix\.com', re.IGNORECASE),
            'prestashop': re.compile(r'prestashop', re.IGNORECASE),
            'volusion': re.compile(r'volusion', re.IGNORECASE),
            'weebly': re.compile(r'weebly', re.IGNORECASE),
            'ecwid': re.compile(r'ecwid', re.IGNORECASE),
            '3dcart': re.compile(r'3dcart', re.IGNORECASE),
            'sellfy': re.compile(r'sellfy', re.IGNORECASE),
            'gumroad': re.compile(r'gumroad', re.IGNORECASE)
        }

        self.payment_indicators = {
            'paypal': re.compile(r'paypal', re.IGNORECASE),
            'stripe': re.compile(r'stripe', re.IGNORECASE),
            'credit_cards': re.compile(r'visa|mastercard|amex|american express', re.IGNORECASE),
            'payment_buttons': re.compile(r'pay now|place order|complete purchase', re.IGNORECASE)
        }

        self.schema_markup = re.compile(r'Product|Offer|AggregateOffer', re.IGNORECASE)

        self.user_agent = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36'
        }

        self.min_confidence_threshold = 3  # Minimum signals needed

    def analyze_page_content(self, soup):
        confidence_score = 0
        detected_features = []

        page_text = soup.get_text().lower()

        for pattern_name, pattern in self.ecommerce_patterns.items():
            if pattern.search(page_text):
                confidence_score += 1
                detected_features.append(pattern_name)

        for platform, pattern in self.platform_signatures.items():
            if pattern.search(page_text):
                confidence_score += 2
                detected_features.append(f"platform:{platform}")

        for payment, pattern in self.payment_indicators.items():
            if pattern.search(page_text):
                confidence_score += 1
                detected_features.append(f"payment:{payment}")

        ld_json_tags = soup.find_all('script', {'type': 'application/ld+json'})
        for tag in ld_json_tags:
            if tag.string and self.schema_markup.search(tag.string):
                confidence_score += 2
                detected_features.append("schema:product")

        product_elements = soup.find_all(class_=re.compile(r'product|item|prod', re.IGNORECASE))
        if len(product_elements) > 3:
            confidence_score += 1
            detected_features.append("multiple_products")

        price_elements = soup.find_all(class_=re.compile(r'price|amount|cost', re.IGNORECASE))
        if len(price_elements) > 3:
            confidence_score += 1
            detected_features.append("multiple_prices")

        return confidence_score, detected_features, len(price_elements)


class EcommerceDetectorWithJS(EcommerceDetector):
    async def fetch_page_with_playwright(self, url):
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(user_agent=self.user_agent['User-Agent'])
                page = await context.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                await asyncio.sleep(2)
                html = await page.content()
                await browser.close()
                return html
        except Exception as e:
            return f"<html><body>Error loading {url}: {str(e)}</body></html>"

    async def is_ecommerce_async(self, url):
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        parsed_url = urlparse(url)
        if not parsed_url.netloc:
            return url, False, 0, ["Invalid URL"]

        html = await self.fetch_page_with_playwright(url)
        soup = BeautifulSoup(html, "html.parser")
        score, features, price_count = self.analyze_page_content(soup)

        # Fallback logic for multilingual e-commerce sites
        if score >= 2 and price_count > 3:
            features.append("fallback:price_detection")
            is_ecom = True
        else:
            is_ecom = score >= self.min_confidence_threshold

        return url, is_ecom, score, features


async def run_detector_on_urls(urls, concurrency=5):
    detector = EcommerceDetectorWithJS()
    sem = asyncio.Semaphore(concurrency)

    async def process_url(url):
        async with sem:
            return await detector.is_ecommerce_async(url)

    tasks = [process_url(url) for url in urls]

    # tqdm_asyncio provides native support for asyncio.gather() with progress bar
    results = await tqdm_asyncio.gather(*tasks, desc="🔍 Analyzing", total=len(tasks))
    return results


if __name__ == "__main__":
    data_ecommerce = pd.read_csv('scripts/e-commerce_websites.csv')['URL'].to_list()
    data_non_ecommerce = pd.read_csv('scripts/non-ecommerce_websites.csv')['URL'].to_list()

    # Create lists to store all relevant data
    all_urls = []
    true_labels = []

    # Mix the two types of data and associate them with their true labels
    for url in data_ecommerce:
        all_urls.append(url)
        true_labels.append(1) # E-commerce

    for url in data_non_ecommerce:
        all_urls.append(url)
        true_labels.append(0) # Non-e-commerce

    # Combine URLs and true labels for shuffling
    combined_data = list(zip(all_urls, true_labels))
    np.random.shuffle(combined_data)
    
    # Unzip them back into separate lists (test_urls will be the shuffled URLs)
    test_urls, true_labels = zip(*combined_data) 
    
    # Convert to list because run_detector_on_urls expects a list
    test_urls_list = list(test_urls)

    results = asyncio.run(run_detector_on_urls(test_urls_list, concurrency=4))

    # Initialize lists to store details for FP and FN
    false_positives_data = []
    false_negatives_data = []
    
    predicted_labels = []

    # Iterate through the results to collect predicted labels and identify FP/FN
    for i, (url, is_ecom, score, features) in enumerate(results):
        predicted_label = 1 if is_ecom else 0
        predicted_labels.append(predicted_label)
        
        original_url = url # The URL from the detector's output
        true_label = true_labels[i] # The true label corresponding to this URL in the shuffled list

        # Identify False Positives (True Label = 0, Predicted Label = 1)
        if true_label == 0 and predicted_label == 1:
            false_positives_data.append({
                "URL": original_url,
                "True_Label": true_label,
                "Predicted_Label": predicted_label,
                "Detector_Score": score,
                "Features": features # Uncomment if you want to save features too, but it might be verbose
            })

        # Identify False Negatives (True Label = 1, Predicted Label = 0)
        elif true_label == 1 and predicted_label == 0:
            false_negatives_data.append({
                "URL": original_url,
                "True_Label": true_label,
                "Predicted_Label": predicted_label,
                "Detector_Score": score,
                "Features": features # Uncomment if you want to save features too
            })

    # Generate the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    print("\n" + "="*30)
    print("Confusion Matrix Results:")
    print("="*30)
    print(cm)
    print("\nInterpretation of Confusion Matrix:")
    print("Rows represent True Labels, Columns represent Predicted Labels")
    print("[[True Negative (TN), False Positive (FP)]")
    print(" [False Negative (FN), True Positive (TP)]]")
    print(f"\n  • **True Positives (TP)**: {cm[1, 1]} (Correctly predicted as e-commerce)")
    print(f"  • **True Negatives (TN)**: {cm[0, 0]} (Correctly predicted as non-e-commerce)")
    print(f"  • **False Positives (FP)**: {cm[0, 1]} (Incorrectly predicted as e-commerce, but was non-e-commerce - Type I error)")
    print(f"  • **False Negatives (FN)**: {cm[1, 0]} (Incorrectly predicted as non-e-commerce, but was e-commerce - Type II error)")
    print("="*30)

    # Save FP and FN URLs to separate CSV files
    if false_positives_data:
        df_fp = pd.DataFrame(false_positives_data)
        df_fp.to_csv('scripts/false_positives_urls.csv', index=False)
        print(f"\nSaved {len(false_positives_data)} False Positive URLs to 'false_positives_urls.csv'")
    else:
        print("\nNo False Positives found.")

    if false_negatives_data:
        df_fn = pd.DataFrame(false_negatives_data)
        df_fn.to_csv('scripts/false_negatives_urls.csv', index=False)
        print(f"Saved {len(false_negatives_data)} False Negative URLs to 'false_negatives_urls.csv'")
    else:
        print("No False Negatives found.")
