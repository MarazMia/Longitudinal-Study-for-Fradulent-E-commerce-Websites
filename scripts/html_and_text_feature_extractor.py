import re
from bs4 import BeautifulSoup
import pandas as pd

# Precompile currency patterns
CURRENCY_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"\$\s?\d+", r"€\s?\d+", r"£\s?\d+", r"\d+\s?(USD|EUR|GBP|৳|¥|₹|₦)"
]]
CURRENCY_SYMBOLS = re.compile(r"\b(USD|EUR|GBP|৳|¥|usd|eur|gbp|₹|₦)\b", re.IGNORECASE)
INTENT_KEYWORDS = [re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) for kw in [
    "add to cart", "checkout", "buy now", "purchase", "pay", "order now", "shop now", "discount", "rating", "star"
]]
CART_KEYWORDS = [re.compile(k, re.IGNORECASE) for k in [
    "cart", "shopping_cart", "cart-icon", "icon-cart", "mini-cart", "cart-wrapper", "minicart", "carrinho", "briefcase",
    "cart-container", "widget_cart", "cart-dropdown", "add-to-cart", "checkout", "cart-shopping", "icon-shopping-cart", "cart-link"
]]

ANCHOR_WORDS = {"buy", "shop", "add", "cart", "checkout", "order", "bag"}

def detect_ecommerce_platform_exists(html):
    # Combine all patterns for fast search
    platform_regexes = [
        r'shopify', r'cdn\.shopify\.com', r'shopify-checkout', r'X-ShopId',
        r'woocommerce', r'wc_cart', r'wc_fragments',
        r'magento', r'mage-init', r'Magento_',
        r'bigcommerce', r'cdn\.bigcommerce\.com',
        r'/cart\.js', r'/cart/add\.js',
        r'opencart', r'squarespace\.com', r'wix\.com',
        r'prestashop', r'blockcart', r'/modules/',
        r'volusion', r'weebly', r'ecwid', r'3dcart', r'sellfy', r'gumroad'
    ]
    return int(any(re.search(p, html, re.IGNORECASE) for p in platform_regexes))


def extract_handcrafted_features(html, text):
    soup = BeautifulSoup(html, 'html.parser')
    features = {}

    features['ecommerce_platform_exists'] = detect_ecommerce_platform_exists(html)

    # Cart element
    features['cart_element_found'] = int(any(p.search(html) for p in CART_KEYWORDS))

    # Currency and prices
    found_prices = [m.group() for pat in CURRENCY_PATTERNS for m in pat.finditer(text)]
    features['price_pattern_found'] = int(bool(found_prices))
    features['num_currencies_found'] = len(set(CURRENCY_SYMBOLS.findall(text)))

    # Intent keywords
    features['ecommerce_keywords_count'] = sum(bool(p.search(text)) for p in INTENT_KEYWORDS)

    # Anchor tags
    all_anchors = soup.find_all('a')
    num_links = len(all_anchors)
    matched_links = sum(
        1 for a in all_anchors
        if (text_ := a.get_text(strip=True).lower()) and any(word in text_ for word in ANCHOR_WORDS)
    )
    features['anchor_buy_ratio'] = round(matched_links / num_links, 3) if num_links else 0.0
    features['anchor_count'] = num_links

    # DOM stats
    all_tags = soup.find_all()
    features['num_html_tags'] = len(all_tags)
    features['text_to_html_ratio'] = round(len(text) / len(html), 3) if html else 0.0
    features['word_count'] = len(text.split())

    # Price tag positions
    price_positions = [
        i for i, tag in enumerate(all_tags)
        if any(pat.search(str(tag)) for pat in CURRENCY_PATTERNS)
    ]
    features['avg_price_dom_position'] = round(sum(price_positions) / len(price_positions), 3) if price_positions else -1

    return features


def extract_tfidf_boolean_features(text, vectorizer, top_keywords):
    vector = vectorizer.transform([text])
    keyword_set = set(top_keywords)
    feature_names = vectorizer.get_feature_names_out()
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}

    return {
        f'has_kw_{kw}': int(vector[0, name_to_idx[kw]] > 0) if kw in name_to_idx else 0
        for kw in keyword_set
    }


def extract_all_features(html, text, vectorizer, top_keywords):
    features = extract_handcrafted_features(html, text)
    features.update(extract_tfidf_boolean_features(text, vectorizer, top_keywords))
    return features
