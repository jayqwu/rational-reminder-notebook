#!/usr/bin/env python3
"""
Rational Reminder Podcast Transcript Scraper
Scrapes all podcast transcripts from rationalreminder.ca.
Episode links are automatically discovered from the podcast directory.
Episode numbers are extracted from page titles.
"""

import argparse
import os
import json
import re
import time
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Configuration
PODCAST_DIRECTORY_URL = "https://rationalreminder.ca/podcast-directory"
TRANSCRIPTS_DIR = "transcripts"
FAILED_URLS_FILE = "failed_episodes.json"
DELAY_BETWEEN_REQUESTS = 0.5  # seconds

DISCLAIMER_PREFIXES = [
    "Disclosure:",
    "Portfolio management and brokerage services in Canada are offered exclusively by PWL Capital",
    "Is there an error in the transcript? Let us know! Email us at info@rationalreminder.ca.",
    "Be sure to add the episode number for reference",
]

SPEAKER_FIX_RE = re.compile(r"^([A-Z][A-Za-z .'-]+):(?=\S)")

# Create directory if it doesn't exist
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Scrape Rational Reminder podcast transcripts")
    parser.add_argument("--url", type=str,
                       help="Scrape a single specific episode URL")
    parser.add_argument("--retry-failed", action="store_true", 
                       help="Only retry previously failed URLs")
    parser.add_argument("--failed-file", default=FAILED_URLS_FILE,
                       help="Path to failed URLs JSON file")
    return parser.parse_args()


def load_failed_urls(failed_file):
    """Load list of previously failed URLs."""
    if not os.path.exists(failed_file):
        return []
    
    try:
        with open(failed_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            urls = data.get('failed_urls', [])
            print(f"✓ Loaded {len(urls)} failed URLs from {failed_file}")
            return urls
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load failed URLs: {e}")
        return []


def save_failed_urls(failed_urls, failed_file):
    """Save list of failed URLs to file."""
    data = {
        'failed_urls': failed_urls,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'count': len(failed_urls)
    }
    
    with open(failed_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Saved {len(failed_urls)} failed URLs to {failed_file}")


def fetch_podcast_directory():
    """Fetch all episode links from the podcast directory page."""
    print(f"Fetching episode links from {PODCAST_DIRECTORY_URL}...")
    
    try:
        response = requests.get(PODCAST_DIRECTORY_URL, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all links that point to podcast episodes
        episode_urls = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Episode links follow pattern: /podcast/... or https://rationalreminder.ca/podcast/...
            if '/podcast/' in href:
                # Normalize to full URL
                if href.startswith('http'):
                    full_url = href
                elif href.startswith('/'):
                    full_url = f"https://rationalreminder.ca{href}"
                else:
                    continue
                
                # Skip the main podcast page
                if full_url.rstrip('/') == 'https://rationalreminder.ca/podcast':
                    continue
                
                episode_urls.append(full_url)
        
        # Remove duplicates while preserving order
        unique_urls = []
        seen = set()
        for url in episode_urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
        
        print(f"✓ Found {len(unique_urls)} episode links")
        return unique_urls
        
    except requests.RequestException as e:
        print(f"Error fetching podcast directory: {e}")
        return []


def fetch_episode_page(episode_url):
    """Fetch the HTML content of an episode page."""
    try:
        response = requests.get(episode_url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching {episode_url}: {e}")
        return None


def extract_transcript(html_content, episode_url):
    """Extract transcript text, metadata, and YouTube info from episode HTML."""
    if not html_content:
        return None
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract episode title - find first non-empty h1 or use meta tag
    title = None
    for h1 in soup.find_all('h1'):
        h1_text = h1.get_text(strip=True)
        if h1_text:
            title = h1_text
            break
    
    # Fallback to og:title meta tag if h1 is still empty
    if not title:
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            title = og_title.get('content').replace(' — Rational Reminder', '')
    
    if not title:
        return None
    
    # Extract YouTube video information
    youtube_data = {}
    
    # Look for YouTube iframe embed
    youtube_iframe = soup.find('iframe', src=re.compile(r'youtube\.com/embed/'))
    if youtube_iframe:
        iframe_src = youtube_iframe.get('src', '')
        # Extract video ID from embed URL (format: youtube.com/embed/VIDEO_ID)
        video_id_match = re.search(r'youtube\.com/embed/([^?&/]+)', iframe_src)
        if video_id_match:
            video_id = video_id_match.group(1)
            youtube_data['video_id'] = video_id
            youtube_data['video_url'] = f"https://www.youtube.com/watch?v={video_id}"
            youtube_data['embed_url'] = iframe_src
    
    # Also check for direct YouTube links
    if not youtube_data:
        for link in soup.find_all('a', href=True):
            href = link['href']
            if 'youtube.com/watch' in href or 'youtu.be/' in href:
                # Extract video ID
                if 'youtube.com/watch' in href:
                    video_id_match = re.search(r'[?&]v=([^&]+)', href)
                elif 'youtu.be/' in href:
                    video_id_match = re.search(r'youtu\.be/([^?&/]+)', href)
                
                if video_id_match:
                    video_id = video_id_match.group(1)
                    youtube_data['video_id'] = video_id
                    youtube_data['video_url'] = f"https://www.youtube.com/watch?v={video_id}"
                    break
    
    # Extract publication date from page
    pub_date = None
    # Try various meta tags and structured data
    date_meta = soup.find('meta', property='article:published_time')
    if not date_meta:
        date_meta = soup.find('meta', attrs={'name': 'publish_date'})
    if not date_meta:
        date_meta = soup.find('meta', attrs={'name': 'date'})
    if not date_meta:
        date_meta = soup.find('time', attrs={'datetime': True})
        if date_meta:
            pub_date = date_meta.get('datetime')
    
    if date_meta and not pub_date:
        pub_date = date_meta.get('content') or date_meta.get('datetime')
    
    # Parse and format date as YYMMDD
    formatted_date = None
    if pub_date:
        try:
            # Handle various date formats
            for fmt in ['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S.%f%z']:
                try:
                    dt = datetime.strptime(pub_date.split('+')[0].split('Z')[0], fmt)
                    formatted_date = dt.strftime('%y%m%d')
                    break
                except ValueError:
                    continue
        except Exception as e:
            print(f"Warning: Could not parse date '{pub_date}': {e}")
    
    # Try to get YouTube video title from page
    if youtube_data:
        # Look for video title in various places
        # Sometimes it's in the title itself or in meta tags
        youtube_data['page_title'] = title
    
    # Find transcript section - look for the heading "Read The Transcript:"
    transcript_text = []
    transcript_found = False
    
    # Try to find transcript heading
    for heading in soup.find_all(['h2', 'h3']):
        if 'transcript' in heading.get_text().lower():
            transcript_found = True
            # Get all content after this heading until next major section
            current = heading.find_next_sibling()
            while current:
                if current.name in['h2', 'h3'] and current != heading:
                    break
                if current.name == 'p':
                    text = current.get_text(" ", strip=True)
                    if text:
                        transcript_text.append(text)
                current = current.find_next_sibling()
            break
    
    if not transcript_found or not transcript_text:
        # Fallback 1: Look for specific content headings that precede transcripts
        for heading in soup.find_all(['h2', 'h3']):
            heading_text = heading.get_text().lower()
            if any(phrase in heading_text for phrase in ['rapid fire', 'listener questions', 'questions and answers', 'q&a', 'interview']):
                # Get all content after this heading
                current = heading.find_next_sibling()
                while current:
                    if current.name in ['h2', 'h3'] and current != heading:
                        break
                    if current.name == 'p':
                        text = current.get_text(" ", strip=True)
                        if text:
                            transcript_text.append(text)
                    current = current.find_next_sibling()
                if transcript_text:
                    transcript_found = True
                    break
    
    if not transcript_found or not transcript_text:
        # Fallback 1b: Look for paragraph-based section markers
        paragraphs = soup.find_all('p')
        for idx, p in enumerate(paragraphs):
            p_text = p.get_text(" ", strip=True)
            # Check if this paragraph is a section marker (short text that looks like a heading)
            if len(p_text) < 100 and any(phrase in p_text.lower() for phrase in ['rapid fire', 'listener questions', 'questions and answers', 'q&a', 'interview', 'read the transcript']):
                # Collect all paragraphs after this marker
                for para in paragraphs[idx + 1:]:
                    para_text = para.get_text(" ", strip=True)
                    if para_text:
                        transcript_text.append(para_text)
                if transcript_text:
                    transcript_found = True
                    break
    
    if not transcript_found or not transcript_text:
        # Fallback 2: Look for transcript after "Key Points" section
        # Key Points entries have timestamps like [0:58:06]
        paragraphs = soup.find_all('p')
        last_key_points_idx = -1
        
        # Find last paragraph with timestamp pattern
        timestamp_pattern = r'\[\d{1,2}:\d{2}:\d{2}\]'
        
        for idx, p in enumerate(paragraphs):
            text = p.get_text(" ", strip=True)
            if re.search(timestamp_pattern, text):
                last_key_points_idx = idx
        
        # If we found Key Points section, transcript starts after it
        if last_key_points_idx >= 0 and last_key_points_idx + 1 < len(paragraphs):
            # Collect all paragraphs after Key Points
            for p in paragraphs[last_key_points_idx + 1:]:
                text = p.get_text(" ", strip=True)
                if text:
                    transcript_text.append(text)
            
            if transcript_text:
                transcript_found = True
    
    if not transcript_found or not transcript_text:
        # Fallback: look for common transcript patterns
        paragraphs = soup.find_all('p')
        for i, p in enumerate(paragraphs):
            text = p.get_text(" ", strip=True)
            # Check if this looks like a transcript (has speaker names followed by colon)
            if ':' in text and any(name in text.split(':')[0] for name in ['Ben Felix', 'Cameron Passmore', 'Mark McGrath','Dan Bortolotti','Ben Wilson']):
                # Found start of transcript, collect all following paragraphs
                for para in paragraphs[i:]:
                    para_text = para.get_text(" ", strip=True)
                    if para_text:
                        transcript_text.append(para_text)
                break
    
    if not transcript_text:
        return None

    def _normalize_for_match(value):
        if not value:
            return ""
        normalized = value
        normalized = normalized.replace("\u2019", "'").replace("\u2018", "'")
        normalized = normalized.replace("\u201c", '"').replace("\u201d", '"')
        normalized = normalized.replace("\u00a0", " ")
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _is_disclaimer(value):
        normalized = _normalize_for_match(value)
        if not normalized:
            return False
        return any(normalized.startswith(prefix) for prefix in DISCLAIMER_PREFIXES)

    def normalize_paragraph(text: str) -> str:
        if not isinstance(text, str):
            return text
        return SPEAKER_FIX_RE.sub(r"\1: ", text)

    transcript_text = [
        normalize_paragraph(paragraph) for paragraph in transcript_text if not _is_disclaimer(paragraph)
    ]
    
    result = {
        'title': title,
        'episode_url': episode_url,
        'transcript': transcript_text
    }
    
    # Add publication date if found
    if formatted_date:
        result['pub_date'] = formatted_date
    
    # Add YouTube data if found
    if youtube_data:
        result['youtube'] = youtube_data
    
    return result


def create_filename_from_title(title, pub_date=None):
    """Create a filesystem-safe filename from title and publication date.
    
    Args:
        title: Episode title string
        pub_date: Publication date in YYMMDD format (optional)
    
    Returns:
        Filename string in format: YYMMDD_cleaned_title.json or cleaned_title.json
    """
    # Remove 'Episode N:' prefix if present
    cleaned = re.sub(r'^Epi(?:[sd]+)?o[ds]e\s+\d+:\s*', '', title, flags=re.IGNORECASE)
    
    # Replace non-alphanumeric characters with underscores
    cleaned = re.sub(r'[^a-zA-Z0-9]+', '_', cleaned)
    
    # Remove leading/trailing underscores and collapse multiple underscores
    cleaned = re.sub(r'_+', '_', cleaned).strip('_')
    
    # Limit length to avoid filesystem issues
    if len(cleaned) > 100:
        cleaned = cleaned[:100].rstrip('_')
    
    # Add date prefix if available
    if pub_date:
        filename = f"{pub_date}_{cleaned}.json"
    else:
        filename = f"{cleaned}.json"
    
    return filename


def scrape_episode(episode_url):
    """Scrape a single episode and save to cache."""
    # We need to fetch first to get title and date
    html = fetch_episode_page(episode_url)
    transcript_data = extract_transcript(html, episode_url)
    
    if not transcript_data:
        return None
    
    # Create filename from title and date
    title = transcript_data['title']
    pub_date = transcript_data.get('pub_date')
    cache_file = os.path.join(TRANSCRIPTS_DIR, create_filename_from_title(title, pub_date))
    
    # Check if already cached
    if os.path.exists(cache_file):
        # Return cached version
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Save to cache
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(transcript_data, f, ensure_ascii=False, indent=2)
    
    return transcript_data


def scrape_all_episodes(episode_urls):
    """Scrape all episodes with progress tracking and rate limiting."""
    if not episode_urls:
        print("No episodes found to scrape.")
        return [], []
    
    print(f"\nScraping {len(episode_urls)} episodes from Rational Reminder podcast...")
    print(f"Results will be cached in '{TRANSCRIPTS_DIR}/' directory\n")
    
    successful = []
    failed_urls = []
    
    for episode_url in tqdm(episode_urls, desc="Scraping episodes"):
        transcript_data = scrape_episode(episode_url)
        
        if transcript_data:
            successful.append(transcript_data['title'])
        else:
            failed_urls.append(episode_url)
            print(f"\n⚠️  Warning: Failed to extract transcript from {episode_url}")
        
        # Rate limiting - don't delay on last episode
        if episode_url != episode_urls[-1]:
            time.sleep(DELAY_BETWEEN_REQUESTS)
    
    print(f"\n✓ Successfully scraped: {len(successful)} episodes")
    if failed_urls:
        print(f"✗ Failed to scrape: {len(failed_urls)} URLs")
        for url in failed_urls[:10]:  # Show first 10 failures
            print(f"  - {url}")
        if len(failed_urls) > 10:
            print(f"  ... and {len(failed_urls) - 10} more")
    
    return successful, failed_urls


def main():
    """Main execution function."""
    args = parse_args()
    
    print("="*70)
    print("Rational Reminder Podcast Transcript Scraper")
    print("="*70)
    print()
    
    # Step 1: Get list of URLs to scrape
    if args.url:
        # Single URL mode
        print(f"Mode: Scraping single episode")
        print(f"URL: {args.url}")
        print()
        
        transcript_data = scrape_episode(args.url)
        if transcript_data:
            title = transcript_data['title']
            pub_date = transcript_data.get('pub_date', 'unknown')
            filename = create_filename_from_title(title, transcript_data.get('pub_date'))
            print(f"\n✓ Successfully scraped: {title}")
            print(f"  Publication date: {pub_date}")
            print(f"  Saved to: {filename}")
            return 0
        else:
            print(f"\n✗ Failed to scrape episode from {args.url}")
            return 1
    
    elif args.retry_failed:
        print("Mode: Retrying previously failed episodes")
        episode_urls = load_failed_urls(args.failed_file)
        if not episode_urls:
            print("\n✗ No failed URLs found to retry.")
            return 1
    else:
        print("Mode: Scraping all episodes from directory")
        episode_urls = fetch_podcast_directory()
        if not episode_urls:
            print("\n✗ Failed to fetch episode URLs from podcast directory.")
            return 1
    
    # Step 2: Scrape all episodes
    successful_episodes, failed_urls = scrape_all_episodes(episode_urls)
    
    # Step 3: Save failed URLs for future retry
    if failed_urls:
        save_failed_urls(failed_urls, args.failed_file)
    elif os.path.exists(args.failed_file) and not args.retry_failed:
        # Clear the failed file if we had no failures on a full run
        os.remove(args.failed_file)
        print(f"✓ Cleared {args.failed_file} (no failures)")
    
    if not successful_episodes:
        print("\n✗ No episodes were successfully scraped.")
        return 1
    
    print("\n" + "="*70)
    print("Process complete!")
    print(f"Transcripts saved in '{TRANSCRIPTS_DIR}/' directory")
    if failed_urls:
        print(f"Failed URLs saved to '{args.failed_file}'")
        print(f"  Retry with: python scrape_transcripts.py --retry-failed")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
