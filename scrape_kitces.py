#!/usr/bin/env python3
"""
Kitces Best Of Posts Scraper
Scrapes Best Of blog posts from kitces.com and filters by category.
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from urllib.parse import urlparse, urljoin, urlunparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from url_cache import URLCache

BEST_OF_URL = "https://www.kitces.com/best-of-posts/"
OUTPUT_DIR = "output/kitces"
FAILED_URLS_FILE = "output/failed_kitces.json"
DELAY_BETWEEN_REQUESTS = 0.5 # seconds

ALLOWED_CATEGORIES = {
    "annuities",
    "debt & liabilities",
    "estate planning",
    "general planning",
    "insurance",
    "investments",
    "retirement planning",
    "taxes"
}

OUTRO_PREFIXES = [
    "We really do use your feedback to shape our future content!",
    "Quality? Nerdy? Relevant?"
]

DEBUG_MODE = False

os.makedirs(OUTPUT_DIR, exist_ok=True)


def handle_warning(message):
    """Handle warning based on debug mode."""
    print(f"WARNING: {message}")
    if DEBUG_MODE:
        print("Debug mode enabled - terminating on warning")
        raise SystemExit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Scrape Kitces Best Of blog posts")
    parser.add_argument("--url", type=str, help="Scrape a single specific post URL")
    parser.add_argument("--retry-failed", action="store_true", help="Retry failed URLs")
    parser.add_argument("--failed-file", default=FAILED_URLS_FILE, help="Failed URLs JSON file")
    parser.add_argument("--force", action="store_true", help="Force re-scrape even if URL is already cached")
    parser.add_argument("--debug", action="store_true", help="Debug mode: terminate on warnings")
    return parser.parse_args()


def clean_url(url):
    """Normalize a URL by stripping fragments and sorting components."""
    parsed = urlparse(url)
    cleaned = parsed._replace(fragment="")
    return urlunparse(cleaned)


def is_article_url(url):
    """Basic filter for Kitces article URLs."""
    parsed = urlparse(url)
    if parsed.netloc and "kitces.com" not in parsed.netloc:
        return False
    if not parsed.path:
        return False
    path = parsed.path.rstrip("/")
    if path == "/best-of-posts":
        return False
    if any(seg in path for seg in ["/category/", "/tag/", "/author/", "/page/", "/feed/"]):
        return False
    if re.search(r"\.(pdf|png|jpg|jpeg|gif|svg)$", path, re.IGNORECASE):
        return False
    return "/blog/" in path


def fetch_listing():
    """Fetch and parse the Best Of listing page for candidate URLs."""
    print(f"Fetching Best Of listing: {BEST_OF_URL}")
    try:
        response = requests.get(BEST_OF_URL, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Error fetching listing: {exc}")
        return []

    urls = []
    link_matches = re.findall(
        r"<li>\s*<a\s+href=\"([^\"]+)\"[^>]*>([^<]+)</a>\s*</li>",
        response.text,
        flags=re.IGNORECASE,
    )
    for href, _title in link_matches:
        if not href:
            continue
        full_url = urljoin(BEST_OF_URL, href.strip())
        full_url = clean_url(full_url)
        if is_article_url(full_url):
            urls.append(full_url)

    # Deduplicate preserving order
    unique_urls = []
    seen = set()
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)

    print(f"Found {len(unique_urls)} candidate URLs")
    return unique_urls


def load_failed_urls(failed_file):
    if not os.path.exists(failed_file):
        return []
    try:
        with open(failed_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        urls = data.get("failed_urls", [])
        print(f"Loaded {len(urls)} failed URLs from {failed_file}")
        return urls
    except (json.JSONDecodeError, OSError) as exc:
        handle_warning(f"Could not load failed URLs: {exc}")
        return []


def save_failed_urls(failed_urls, failed_file):
    data = {
        "failed_urls": failed_urls,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "count": len(failed_urls),
    }
    with open(failed_file, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    print(f"Saved {len(failed_urls)} failed URLs to {failed_file}")


def fetch_article(url):
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.text
    except requests.RequestException as exc:
        print(f"Error fetching {url}: {exc}")
        return None


def extract_title(soup):
    h1 = soup.find("h1")
    if h1:
        title_text = h1.get_text(strip=True)
        if title_text:
            return title_text
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return og_title.get("content").strip()
    return None


def extract_categories(soup):
    categories = []
    container = soup.find(class_="entry-categories")
    if not container:
        return categories
    for item in container.find_all(["a", "span"]):
        text = item.get_text(" ", strip=True)
        if text:
            categories.append(text)
    if not categories:
        raw_text = container.get_text(" ", strip=True)
        if raw_text:
            categories = [raw_text]
    normalized = []
    seen = set()
    for category in categories:
        cleaned = re.sub(r"\s+", " ", category).strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            normalized.append(cleaned)
            seen.add(key)
    return normalized


def has_allowed_category(categories):
    for category in categories:
        if category.lower() in ALLOWED_CATEGORIES:
            return True
    return False


def extract_publish_date(soup):
    date_meta = (
        soup.find("meta", property="article:published_time")
        or soup.find("meta", attrs={"name": "publish_date"})
        or soup.find("meta", attrs={"name": "date"})
    )
    if date_meta and date_meta.get("content"):
        return date_meta.get("content")
    time_tag = soup.find("time", attrs={"datetime": True})
    if time_tag:
        return time_tag.get("datetime")
    return None


def format_pub_date(pub_date):
    if not pub_date:
        return None
    for fmt in [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%f%z",
    ]:
        try:
            cleaned = pub_date.split("+")[0].split("Z")[0]
            dt = datetime.strptime(cleaned, fmt)
            return dt.strftime("%y%m%d")
        except ValueError:
            continue
    return None


def extract_author(soup):
    meta_author = soup.find("meta", attrs={"name": "author"})
    if meta_author and meta_author.get("content"):
        return meta_author.get("content").strip()

    author_candidates = [
        soup.find(class_="author-name"),
        soup.find(class_="entry-author"),
        soup.find("span", class_="author"),
        soup.find("a", rel="author"),
    ]
    for candidate in author_candidates:
        if not candidate:
            continue
        text = candidate.get_text(" ", strip=True)
        if text:
            return text
    return None


def extract_executive_summary(soup):
    summary_block = soup.find(class_="executive-summary")
    if not summary_block:
        return None
    paragraphs = [p.get_text(" ", strip=True) for p in summary_block.find_all("p")]
    paragraphs = [p for p in paragraphs if p]
    if paragraphs:
        return paragraphs
    text = summary_block.get_text(" ", strip=True)
    return [text] if text else None


def extract_content(soup):
    content_container = soup.find(class_="pf-content") or soup.find(class_="entry-content") or soup.find(class_="content")
    if not content_container:
        return []

    for block in content_container.find_all(class_=["executive-summary", "author-block"]):
        block.decompose()

    # Remove all table elements
    for table in content_container.find_all("table"):
        table.decompose()

    text_blocks = []
    for element in content_container.find_all(["p", "li"], recursive=True):
        text = element.get_text(" ", strip=True)
        if text:
            text_blocks.append(text)
    
    # Remove lines containing "Click to Tweet"
    text_blocks = [block for block in text_blocks if "Click To Tweet" not in block]

    # Check last 10 lines for outro prefixes and remove them and all following lines
    if len(text_blocks) > 0:
        check_count = min(10, len(text_blocks))
        for i in range(len(text_blocks) - check_count, len(text_blocks)):
            for outro_prefix in OUTRO_PREFIXES:
                if text_blocks[i].startswith(outro_prefix):
                    # Remove this line and all following lines
                    text_blocks = text_blocks[:i]
                    return text_blocks
    
    return text_blocks


def extract_article(html, url):
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")

    title = extract_title(soup)
    if not title:
        return None

    categories = extract_categories(soup)
    if not categories:
        handle_warning(f"No categories found for {url}")
        return None
    if not has_allowed_category(categories):
        return "FILTERED"

    executive_summary = extract_executive_summary(soup)
    content = extract_content(soup)
    if not content:
        handle_warning(f"No content found for {url}")
        return None

    pub_date_raw = extract_publish_date(soup)
    pub_date = format_pub_date(pub_date_raw)
    author = extract_author(soup)

    result = {
        "title": title,
        "pub_date": pub_date,
        "url": url,
        "summary": executive_summary,
        "content": content,
    }

    return result


def create_filename_from_title(title, pub_date=None):
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", title)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if len(cleaned) > 100:
        cleaned = cleaned[:100].rstrip("_")
    if pub_date:
        return f"{pub_date}_{cleaned}.json"
    return f"{cleaned}.json"


def scrape_single(url, url_cache=None, force_rescrape=False):
    """Scrape a single article and save to file.
    
    Args:
        url: URL of the article to scrape
        url_cache: URLCache instance for tracking scraped URLs
        force_rescrape: If True, scrape even if URL is already cached
    
    Returns:
        Article data dict if successful, "FILTERED" if filtered out, "CACHED" if skipped, None on failure
    """
    # Check cache first (unless force_rescrape is enabled)
    if url_cache and not force_rescrape:
        if url_cache.is_cached(url):
            if url_cache.is_filtered(url):
                return "CACHED_FILTERED"
            return "CACHED"
    
    html = fetch_article(url)
    article = extract_article(html, url)
    if article == "FILTERED":
        # Add filtered URL to cache so we don't check it again
        if url_cache:
            url_cache.add_filtered_url(url)
            url_cache.save_cache()
        return "FILTERED"
    if not article:
        return None

    filename = create_filename_from_title(article["title"], article.get("pub_date"))
    output_path = os.path.join(OUTPUT_DIR, filename)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(article, handle, ensure_ascii=False, indent=2)
    
    # Update cache and save to disk immediately
    if url_cache:
        url_cache.add_url(url, output_path)
        url_cache.save_cache()
    
    return article


def scrape_all(urls, url_cache=None, force_rescrape=False):
    """Scrape all articles with progress tracking and rate limiting.
    
    Args:
        urls: List of article URLs to scrape
        url_cache: URLCache instance for tracking scraped URLs
        force_rescrape: If True, scrape even if URLs are already cached
    
    Returns:
        Tuple of (successful_articles, failed_urls)
    """
    if not urls:
        print("No candidate URLs found.")
        return [], []

    successful = []
    failed = []
    skipped_cached = 0
    skipped_filtered = 0
    skipped_previously_filtered = 0

    for url in tqdm(urls, desc="Scraping posts"):
        article = scrape_single(url, url_cache, force_rescrape)
        if article == "CACHED":
            skipped_cached += 1
        elif article == "CACHED_FILTERED":
            skipped_previously_filtered += 1
        elif article == "FILTERED":
            skipped_filtered += 1
        elif article:
            successful.append(article["title"])
        else:
            failed.append(url)
            handle_warning(f"Failed to scrape {url}")
        if url != urls[-1] and article not in ["CACHED", "CACHED_FILTERED", "FILTERED"]:
            time.sleep(DELAY_BETWEEN_REQUESTS)

    print(f"\n✓ Successfully scraped: {len(successful)} articles")
    if skipped_cached > 0:
        print(f"⊘ Skipped (already cached): {skipped_cached} articles")
    if skipped_previously_filtered > 0:
        print(f"⊘ Skipped (previously filtered): {skipped_previously_filtered} articles")
    if skipped_filtered > 0:
        print(f"⊘ Skipped (filtered by category): {skipped_filtered} articles")

    return successful, failed


def main():
    global DEBUG_MODE
    args = parse_args()
    DEBUG_MODE = args.debug

    print("=" * 70)
    print("Kitces Best Of Posts Scraper")
    if DEBUG_MODE:
        print("(Debug mode enabled - will terminate on warnings)")
    if args.force:
        print("(Force rescrape enabled - will overwrite cached URLs)")
    print("=" * 70)

    # Initialize URL cache
    url_cache = URLCache()
    print(f"Loaded URL cache with {url_cache.get_cache_size()} entries\n")

    if args.url:
        print("Mode: Scraping single post")
        print(f"URL: {args.url}")
        article = scrape_single(args.url, url_cache, args.force)
        if article == "CACHED":
            print(f"\n⊘ Article already cached (use --force to re-scrape)")
            return 0
        elif article == "CACHED_FILTERED":
            print(f"\n⊘ Article previously filtered (use --force to re-check)")
            return 0
        elif article == "FILTERED":
            print(f"\n⊘ Article filtered out (category not in allowed list)")
            return 0
        elif article:
            filename = create_filename_from_title(article["title"], article.get("pub_date"))
            print(f"\n✓ Success: {article['title']}")
            print(f"Saved to: {os.path.join(OUTPUT_DIR, filename)}")
            url_cache.save_cache()
            return 0
        print("\n✗ Failed to scrape post")
        return 1

    if args.retry_failed:
        urls = load_failed_urls(args.failed_file)
        if not urls:
            print("No failed URLs to retry.")
            return 1
    else:
        urls = fetch_listing()
        if not urls:
            print("Failed to load Best Of listing.")
            return 1

    successful, failed = scrape_all(urls, url_cache, args.force)

    # Save URL cache
    url_cache.save_cache()
    print(f"\n✓ URL cache updated ({url_cache.get_cache_size()} total entries)")

    if failed:
        save_failed_urls(failed, args.failed_file)
    elif os.path.exists(args.failed_file) and not args.retry_failed:
        os.remove(args.failed_file)
        print(f"✓ Cleared {args.failed_file} (no failures)")

    if not successful and not url_cache.get_cache_size():
        print("\n✗ No posts were successfully scraped.")
        return 1

    print("\n" + "=" * 70)
    print("Process complete!")
    print(f"Articles saved in '{OUTPUT_DIR}/' directory")
    if failed:
        print(f"Failed URLs saved to '{args.failed_file}'")
        print("Retry with: python scrape_kitces.py --retry-failed")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
