import os
import json
import re
import pandas as pd
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
from tqdm import tqdm

# --- Configuration ---
SAVE_DIR = "scripts/extracted_website_data_for_feature_engineering"
ECOMMERCE_CSV = "scripts/e-commerce_websites.csv"
NONECOMMERCE_CSV = "scripts/non-ecommerce_websites.csv"
TOP_N_TFIDF_FEATURES = 20

# --- Helpers ---
def sanitize_filename(url):
    parsed = urlparse(url)
    netloc = parsed.netloc.replace(".", "_")
    path = parsed.path.replace("/", "_")
    return f"{netloc}{path}.json"

def normalize_url(url):
    return url if url.startswith(("http://", "https://")) else "https://" + url

def detect_ecommerce_platform_exists(html):
    patterns = {
        'shopify': [r'shopify', r'cdn\.shopify\.com', r'shopify-checkout', r'X-ShopId'],
        'woocommerce': [r'woocommerce', r'wc_cart', r'wc_fragments'],
        'magento': [r'magento', r'mage-init', r'Magento_'],
        'bigcommerce': [r'bigcommerce', r'cdn\.bigcommerce\.com'],
        'cart_scripts': [r'/cart\.js', r'/cart/add\.js'],
        'opencart': [r'opencart'],
        'squarespace': [r'squarespace\.com'],
        'wix': [r'wix\.com'],
        'prestashop': [r'prestashop', r'blockcart', r'/modules/'],
        'volusion': [r'volusion'],
        'weebly': [r'weebly'],
        'ecwid': [r'ecwid'],
        '3dcart': [r'3dcart'],
        'sellfy': [r'sellfy'],
        'gumroad': [r'gumroad']
    }
    for regexes in patterns.values():
        if any(re.search(rgx, html, re.IGNORECASE) for rgx in regexes):
            return 1
    return 0

# --- Feature Extractor ---
def extract_features(html, text):
    soup = BeautifulSoup(html, 'html.parser')
    features = {}

    features['ecommerce_platform_exists'] = detect_ecommerce_platform_exists(html)

    cart_keywords = ["cart", "shopping_cart", "cart-icon", "icon-cart", "mini-cart", "cart-wrapper", "minicart", "carrinho", "briefcase",
                     "cart-container", "widget_cart", "cart-dropdown", "add-to-cart", "checkout", "cart-shopping", "icon-shopping-cart", "cart-link"]
    features['cart_element_found'] = int(any(re.search(k, html, re.IGNORECASE) for k in cart_keywords))

    currency_patterns = [r"\$\s?\d+", r"€\s?\d+", r"£\s?\d+", r"\d+\s?(USD|EUR|GBP|৳|¥|₹|₦)"]
    found_prices = [m.group() for pat in currency_patterns for m in re.finditer(pat, text)]
    features['price_pattern_found'] = int(bool(found_prices))

    features['num_currencies_found'] = len(set(re.findall(r"\b(USD|EUR|GBP|৳|¥|usd|eur|gbp|৳|¥|₹|₦)\b", text, re.IGNORECASE)))

    intent_keywords = ["add to cart", "checkout", "buy now", "purchase", "pay", "order now", "shop now", "discount", "rating", "star"]
    features['ecommerce_keywords_count'] = sum(bool(re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE)) for kw in intent_keywords)

    all_anchors = soup.find_all('a')
    num_links = len(all_anchors)
    anchor_words = {'buy', 'shop', 'add', 'cart', 'checkout', 'order', "bag"}
    matched_links = sum(1 for a in all_anchors if a.get_text() and any(word in a.get_text().lower() for word in anchor_words))
    features['anchor_buy_ratio'] = round(matched_links / num_links, 3) if num_links else 0.0

    features['num_html_tags'] = len(soup.find_all())
    features['text_to_html_ratio'] = round(len(text) / len(html), 3) if html else 0.0
    features['word_count'] = len(text.split())
    features['anchor_count'] = num_links

    # Price tag positions
    tags = soup.find_all()
    price_positions = [i for i, tag in enumerate(tags) if any(re.search(pat, str(tag), re.IGNORECASE) for pat in currency_patterns)]
    features['avg_price_dom_position'] = round(sum(price_positions) / len(price_positions), 3) if price_positions else -1

    return features

# --- Dataset Builder ---
def build_dataset():
    ecommerce_urls = pd.read_csv(ECOMMERCE_CSV)['URL'].dropna().tolist()
    nonecommerce_urls = pd.read_csv(NONECOMMERCE_CSV)['URL'].dropna().tolist()

    all_rows = []
    all_urls = [(url, 1) for url in ecommerce_urls] + [(url, 0) for url in nonecommerce_urls]

    for url, label in tqdm(all_urls, desc="Building dataset"):
        norm_url = normalize_url(url)
        filename = sanitize_filename(norm_url)
        filepath = os.path.join(SAVE_DIR, filename)
        if not os.path.isfile(filepath):
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        html = data.get("html", "")
        text = data.get("text", "")
        features = extract_features(html, text)
        features['url'] = url
        features['text'] = text
        features['label'] = label
        all_rows.append(features)

    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    df_all = build_dataset()

    print("Computing TF-IDF and training classifier...")
    tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    X_tfidf = tfidf_vectorizer.fit_transform(df_all['text'])
    tfidf_features = tfidf_vectorizer.get_feature_names_out()

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_tfidf, df_all['label'])

    top_n_indices = clf.feature_importances_.argsort()[::-1][:TOP_N_TFIDF_FEATURES]
    top_keywords = [tfidf_features[i] for i in top_n_indices]

    X_binary_keywords = (X_tfidf[:, top_n_indices] > 0).astype(int)
    tfidf_df = pd.DataFrame(X_binary_keywords.toarray(), columns=[f'has_kw_{kw}' for kw in top_keywords])

    df_final = pd.concat([df_all.drop(columns=['text']), tfidf_df], axis=1)
    df_final.to_csv("scripts/ecommerce_feature_dataset_with_keywords.csv", index=False)
    print(f"✅ Saved: scripts/ecommerce_feature_dataset_with_keywords.csv")

    with open("scripts/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
    with open("scripts/top_keywords.pkl", "wb") as f:
        pickle.dump(top_keywords, f)
    print("✅ Saved TF-IDF vectorizer and top keywords for future prediction tasks.")
