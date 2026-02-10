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
from url_cache import URLCache

# Configuration
PODCAST_DIRECTORY_URL = "https://rationalreminder.ca/podcast-directory"
TRANSCRIPTS_DIR = "output/rational_reminder"
FAILED_URLS_FILE = "output/failed_rr.json"
DELAY_BETWEEN_REQUESTS = 0.5  # seconds

OUTRO_PREFIXES = [
    "[AFTER SHOW]",
    "Disclosure:",
    "Disclaimer:",
    "Policies and Disclaimer",
    "Portfolio management and brokerage services in Canada are offered exclusively by PWL Capital",
    "Announcer: Portfolio management and brokerage services in Canada are offered exclusively by PWL Capital Inc.",
    "Is there an error in the transcript? Let us know! Email us at info@rationalreminder.ca.",
]

SPEAKER_FIX_RE = re.compile(r"^([A-Z][A-Za-z .'-]+):(?=\S)")

# Create directory if it doesn't exist
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)

# Global debug mode flag
DEBUG_MODE = False


def handle_warning(message):
    """Handle warning based on debug mode.
    
    In debug mode, raises SystemExit. Otherwise, prints warning.
    """
    print(f"⚠️  WARNING: {message}")
    if DEBUG_MODE:
        print("\n✗ Debug mode enabled - terminating on warning")
        raise SystemExit(1)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Scrape Rational Reminder podcast transcripts")
    parser.add_argument("--url", type=str,
                       help="Scrape a single specific episode URL")
    parser.add_argument("--retry-failed", action="store_true", 
                       help="Only retry previously failed URLs")
    parser.add_argument("--failed-file", default=FAILED_URLS_FILE,
                       help="Path to failed URLs JSON file")
    parser.add_argument("--force", action="store_true",
                       help="Force re-scrape even if URL is already cached")
    parser.add_argument("--debug", action="store_true",
                       help="Debug mode: terminate on any warnings")
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
        handle_warning(f"Could not load failed URLs: {e}")
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


def extract_summary(soup):
    """Extract key points from the episode page.
    
    Key points are lines that contain timestamps. Timestamps match pattern:
    \\d{1,2}:\\d{2}(:\\d{2})? with optional brackets/parens at beginning or end.
    
    Examples: (0:21:19), [0:11:43], 0:09:33), 1:23:05], [01:43], (0:32)
    
    Returns:
        List of key points with timestamps removed, or empty list if not found.
    """
    summary = []
    
    # Timestamp patterns at start or end of line
    # Format: optional bracket/paren, MM:SS or HH:MM:SS, optional decimal seconds, optional bracket/paren
    # Examples: [0:03:52.2], (0:03:52), [01:43], (0:32.5)
    timestamp_at_start = r'^[\[\(]?\d{1,2}:\d{1,2}(:\d{1,2})?(\.\d+)?(:\d+)?[\]\)]?\s*'
    timestamp_at_end = r'\s*[\[\(]?\d{1,2}:\d{1,2}(:\d{1,2})?(\.\d+)?(:\d+)?[\]\)]?$'
    
    # Look for "Key Points" heading - check h2/h3 first (newer format)
    summary_heading = None
    for heading in soup.find_all(['h2', 'h3']):
        heading_text = heading.get_text(strip=True).lower()
        if 'key points' in heading_text:
            summary_heading = heading
            break
    
    # If not found in h2/h3, look for <strong> tags with "Key Points" (older format)
    if not summary_heading:
        for strong in soup.find_all('strong'):
            strong_text = strong.get_text(strip=True).lower()
            if 'key points' in strong_text:
                # For older format, the <strong> is inside a <p>, use the <p> as the reference
                summary_heading = strong.parent
                break
    
    # If still no heading found, look for a cluster of paragraphs with timestamps
    # This handles episodes without explicit "Key Points" heading
    if not summary_heading:
        paragraphs = soup.find_all('p')
        
        # Find sequences of paragraphs that start with timestamps
        for i, p in enumerate(paragraphs):
            p_text = p.get_text(" ", strip=True)
            # Check if this paragraph starts with a timestamp
            if re.match(timestamp_at_start, p_text):
                # Check if there are multiple consecutive paragraphs with timestamps
                consecutive_count = 1
                for j in range(i + 1, min(i + 10, len(paragraphs))):
                    next_text = paragraphs[j].get_text(" ", strip=True)
                    if re.match(timestamp_at_start, next_text):
                        consecutive_count += 1
                    elif len(next_text) > 20:  # Skip very short paragraphs
                        break
                
                # If we found at least 3 consecutive timestamped paragraphs, use this as key points
                if consecutive_count >= 3:
                    # Process from this paragraph onward
                    for idx in range(i, len(paragraphs)):
                        para_text = paragraphs[idx].get_text(" ", strip=True)
                        has_timestamp = (re.search(timestamp_at_start, para_text) or 
                                       re.search(timestamp_at_end, para_text))
                        
                        if has_timestamp:
                            clean_text = re.sub(timestamp_at_start, '', para_text)
                            clean_text = re.sub(timestamp_at_end, '', clean_text).strip()
                            if clean_text and len(clean_text) > 5:
                                summary.append(clean_text)
                        elif len(para_text) > 50:
                            # Stop if we hit a non-timestamp paragraph with substantial content
                            break
                    
                    return summary
        
        # If we didn't find key points via cluster detection, return empty list
        return summary
    
    # If we found a key points heading, collect all text content between it and next major section
    content_blocks = []
    current = summary_heading.find_next_sibling()
    
    while current:
        # Stop at next major heading
        if current.name in ['h2', 'h3']:
            break
        
        # Extract text from various container types
        if current.name == 'p':
            text = current.get_text(" ", strip=True)
            if text and not text.startswith('Read The Transcript'):
                content_blocks.append(text)
        elif current.name == 'li':
            text = current.get_text(" ", strip=True)
            if text:
                content_blocks.append(text)
        elif current.name in ['ul', 'ol']:
            for li in current.find_all('li', recursive=False):
                li_text = li.get_text(" ", strip=True)
                if li_text:
                    content_blocks.append(li_text)
        elif current.name == 'div':
            text = current.get_text(" ", strip=True)
            if text and len(text) > 5:
                content_blocks.append(text)
        
        current = current.find_next_sibling()
    
    # Process each content block to find lines with timestamps
    for block in content_blocks:
        # Split by newlines first
        lines = block.split('\n')
        
        # If single line with multiple timestamps, try to split by timestamp boundaries
        # Look for patterns like "text) (0:12:34" or "text] [0:12:34"
        # Only split if we detect MULTIPLE timestamps (multiple [ or ( followed by time pattern)
        if len(lines) == 1:
            # Count timestamps in the block - look for [ or ( followed by \d:\d pattern
            timestamp_count = len(re.findall(r'[\[\(]\d{1,2}:\d{2}', block))
            if timestamp_count > 1:
                # Try splitting directly on leading timestamp pattern
                # Pattern matches: optional whitespace, opening bracket/paren, timestamp, closing bracket/paren, optional whitespace
                # This captures the full timestamp marker: [0:12:34] or (1:23:45)
                parts = re.split(r'\s*([\[\(]\d{1,2}:\d{1,2}(?::\d{1,2})?(?:\.\d+)?[\]\)])\s*', block)
                
                # Filter out empty parts and reconstruct by prepending timestamp to following text
                reconstructed = []
                i = 0
                while i < len(parts):
                    part = parts[i].strip()
                    if part:
                        # Check if this looks like a timestamp
                        if re.match(r'^[\[\(]\d{1,2}:\d{1,2}', part):
                            # This is a timestamp, combine with next part if it exists
                            if i + 1 < len(parts) and parts[i + 1].strip():
                                reconstructed.append(part + ' ' + parts[i + 1].strip())
                                i += 2
                                continue
                            else:
                                # Timestamp alone, just add it
                                reconstructed.append(part)
                        else:
                            # Not a timestamp, just add it
                            reconstructed.append(part)
                    i += 1
                
                if len(reconstructed) > 1:
                    lines = reconstructed
                else:
                    # Fallback 1: Split where text ends (with period or similar) and next timestamp begins
                    # Pattern: look for ". (" or ". [" or ") (" or "] [" before a timestamp
                    parts = re.split(r'(?<=[.!?)\]])\s+(?=[\[\(]\d{1,2}:\d{2})', block)
                    if len(parts) > 1:
                        lines = parts
                    else:
                        # Fallback 2: Split where closing bracket/paren is followed by opening bracket/paren with timestamp
                        parts = re.split(r'([\]\)])\s*(?=[\[\(]\d{1,2}:\d{2})', block)
                        # Reconstruct by joining pairs
                        reconstructed = []
                        current_line = ""
                        for part in parts:
                            current_line += part
                            if part in [')', ']']:
                                reconstructed.append(current_line)
                                current_line = ""
                        if current_line:
                            reconstructed.append(current_line)
                        lines = reconstructed if len(reconstructed) > 1 else lines
        
        for line in lines:
            line = line.strip()

            # Check if line contains timestamp at beginning or end
            has_timestamp = (re.search(timestamp_at_start, line) or 
                           re.search(timestamp_at_end, line))
            
            if has_timestamp:
                # Remove timestamps from both ends
                clean_text = re.sub(timestamp_at_start, '', line)
                clean_text = re.sub(timestamp_at_end, '', clean_text).strip()
                
                # Remove trailing brackets and bullet points
                clean_text = clean_text.replace("•", "").strip()
                clean_text = re.sub(r'\s*(\[|\()\s*$', '', clean_text).strip()
                
                if clean_text and len(clean_text) > 5:
                    summary.append(clean_text)
    
    return summary

def clean_transcript(transcript_text):

    def normalize_paragraph(text: str) -> str:
        if not isinstance(text, str):
            return text
        return SPEAKER_FIX_RE.sub(r"\1: ", text)

    transcript_text = [
        normalize_paragraph(paragraph) for paragraph in transcript_text
    ]

    def trim_transcript_by_marker(paragraphs, markers):
        """Trim transcript by any of the provided markers (mutually exclusive).
        
        Args:
            paragraphs: List of transcript paragraphs
            markers: List of marker strings to search for
        
        Returns:
            Tuple of (trimmed_paragraphs, was_trimmed)
        """
        for marker in markers:
            marker_positions = [
                idx for idx, paragraph in enumerate(paragraphs)
                if isinstance(paragraph, str) and paragraph.strip() == marker
            ]
            if not marker_positions:
                continue
            if len(marker_positions) == 1:
                return paragraphs[marker_positions[0] + 1:], True
            first_idx = marker_positions[0]
            last_idx = marker_positions[-1]
            return paragraphs[first_idx + 1:last_idx], True
        
        return paragraphs, False
    
    # Trim transcript text by known markers to remove intro and outro content
    # Markers are mutually exclusive, so check all at once
    transcript_text, trimmed_by_marker = trim_transcript_by_marker(
        transcript_text, 
        ["***", "[EPISODE]", "[INTERVIEW]"]
    )

    # Remove lines ending with "welcome to the Rational Reminder Podcast" and all preceding lines
    # Only check lines 5 to 100 of the transcript
    # Skip this check if any marker-based trimming already occurred
    if not trimmed_by_marker:
        check_start = min(5, len(transcript_text))
        check_end = min(100, len(transcript_text))
        for idx in range(check_start, check_end):
            line = transcript_text[idx]
            if isinstance(line, str) and re.search(r'welcome[^(\.|\?|!)]+the rational reminder podcast(\.|\?|!)$', line, re.IGNORECASE):
                # Remove this line and all preceding lines
                transcript_text = transcript_text[idx + 1:]
                break

    # Remove lines containing "Disclosure:" or "Disclaimer:" and all following lines
    # Only check the last 50 lines of the transcript
    check_start = max(0, len(transcript_text) - 50)
    for idx in range(check_start, len(transcript_text)):
        line = transcript_text[idx]
        if any(phrase in line for phrase in OUTRO_PREFIXES) or line.startswith('Is there an error in the transcript?'):
            # Remove this line and all following lines
            transcript_text = transcript_text[:idx]
            break

    # Remove lines ending with "How do you define success in your life?" and all following lines
    check_start = max(0, len(transcript_text) - 50)
    for idx in range(check_start, len(transcript_text)):
        line = transcript_text[idx]
        if line.lower().endswith('how do you define success in your life?'):
            # Remove this line and all following lines
            transcript_text = transcript_text[:idx]
            break
            
    # Remove all instances of " : " from transcript text
    transcript_text = [paragraph.replace(" : ", ": ") for paragraph in transcript_text]
    # Remove lines that are exactly "***"
    transcript_text = [paragraph for paragraph in transcript_text if paragraph.strip() != "***"]
    # Remove all instances of " : " from transcript text
    transcript_text = [paragraph.replace(" : ", ": ") for paragraph in transcript_text]
    # Remove lines that are exactly "***"
    transcript_text = [paragraph for paragraph in transcript_text if paragraph.strip() != "***"]
    # Normalize [inaudible ...] to just [inaudible]
    transcript_text = [re.sub(r'\[inaudible[^\]]*\]', '[inaudible]', paragraph) for paragraph in transcript_text]

    return transcript_text

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
            handle_warning(f"Could not parse date '{pub_date}': {e}")
    
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
                if current.name == 'p' or current.get_text() == "***":
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
        # Key Points entries have timestamps like [0:58:06] or (0:58)
        paragraphs = soup.find_all('p')
        last_summary_idx = -1
        
        # Find last paragraph with timestamp pattern (handles [/( and with/without seconds)
        timestamp_pattern = r'[\[\(]\d{1,2}:\d{2}(:\d{2})?[\]\)]'
        
        for idx, p in enumerate(paragraphs):
            text = p.get_text(" ", strip=True)
            if re.search(timestamp_pattern, text):
                last_summary_idx = idx
        
        # If we found Key Points section, transcript starts after it
        if last_summary_idx >= 0 and last_summary_idx + 1 < len(paragraphs):
            # Collect all paragraphs after Key Points
            for p in paragraphs[last_summary_idx + 1:]:
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
    
    cleaned_transcript=clean_transcript(transcript_text)
    
    # Extract key points
    summary = extract_summary(soup)
    
    # Build result with fields in specific order: title, pub_date, summary, episode_url, transcript, youtube
    result = {
        "title": title,
        "pub_date": formatted_date,
        "url": episode_url,
        "summary": summary,
        "content": cleaned_transcript,
    }

    if youtube_data:
        result["youtube"] = youtube_data
    
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
    cleaned = re.sub(r'Episode ','Ep', title)
    cleaned = re.sub(r'^Epi(d|o|s|e){4} ', 'Ep', cleaned)
    cleaned = re.sub(r'Understanding Crypto ','UC', cleaned)
    
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


def scrape_episode(episode_url, url_cache=None, force_rescrape=False):
    """Scrape a single episode and save to file.
    
    Args:
        episode_url: URL of the episode to scrape
        url_cache: URLCache instance for tracking scraped URLs
        force_rescrape: If True, scrape even if URL is already cached
    
    Returns:
        Transcript data dict if successful, None otherwise
    """
    # Check cache first (unless force_rescrape is enabled)
    if url_cache and not force_rescrape:
        if url_cache.is_cached(episode_url):
            return "CACHED"
    
    # Fetch episode page and extract transcript
    html = fetch_episode_page(episode_url)
    transcript_data = extract_transcript(html, episode_url)
    
    if not transcript_data:
        return None
    
    # Create filename from title and date
    title = transcript_data['title']
    pub_date = transcript_data.get('pub_date')
    
    output_file = os.path.join(TRANSCRIPTS_DIR, create_filename_from_title(title, pub_date))
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transcript_data, f, ensure_ascii=False, indent=2)
    
    # Update cache
    if url_cache:
        url_cache.add_url(episode_url, output_file)
    
    return transcript_data


def scrape_all_episodes(episode_urls, url_cache=None, force_rescrape=False):
    """Scrape all episodes with progress tracking and rate limiting.
    
    Args:
        episode_urls: List of episode URLs to scrape
        url_cache: URLCache instance for tracking scraped URLs
        force_rescrape: If True, scrape even if URLs are already cached
    
    Returns:
        Tuple of (successful_episodes, failed_urls)
    """
    if not episode_urls:
        print("No episodes found to scrape.")
        return [], []
    
    print(f"\nScraping {len(episode_urls)} episodes from Rational Reminder podcast...")
    print(f"Results will be saved in '{TRANSCRIPTS_DIR}/' directory\n")
    
    successful = []
    failed_urls = []
    skipped_cached = 0
    
    for episode_url in tqdm(episode_urls, desc="Scraping episodes"):
        transcript_data = scrape_episode(episode_url, url_cache, force_rescrape)
        
        if transcript_data == "CACHED":
            skipped_cached += 1
        elif transcript_data:
            successful.append(transcript_data['title'])
        else:
            failed_urls.append(episode_url)
            handle_warning(f"Failed to extract transcript from {episode_url}")
        
        # Rate limiting - don't delay on last episode or cached episodes
        if episode_url != episode_urls[-1] and transcript_data != "CACHED":
            time.sleep(DELAY_BETWEEN_REQUESTS)
    
    print(f"\n✓ Successfully scraped: {len(successful)} episodes")
    if skipped_cached > 0:
        print(f"⊘ Skipped (already cached): {skipped_cached} episodes")
    if failed_urls:
        print(f"✗ Failed to scrape: {len(failed_urls)} URLs")
        for url in failed_urls[:10]:  # Show first 10 failures
            print(f"  - {url}")
        if len(failed_urls) > 10:
            print(f"  ... and {len(failed_urls) - 10} more")
    
    return successful, failed_urls


def main():
    """Main execution function."""
    global DEBUG_MODE
    args = parse_args()
    DEBUG_MODE = args.debug
    
    print("="*70)
    print("Rational Reminder Podcast Transcript Scraper")
    if DEBUG_MODE:
        print("(Debug mode enabled - will terminate on any warnings)")
    if args.force:
        print("(Force rescrape enabled - will overwrite cached URLs)")
    print("="*70)
    print()
    
    # Initialize URL cache
    url_cache = URLCache()
    print(f"Loaded URL cache with {url_cache.get_cache_size()} entries\n")
    
    # Step 1: Get list of URLs to scrape
    if args.url:
        # Single URL mode
        print(f"Mode: Scraping single episode")
        print(f"URL: {args.url}")
        print()
        
        transcript_data = scrape_episode(args.url, url_cache, args.force)
        if transcript_data == "CACHED":
            print(f"\n⊘ Episode already cached (use --force to re-scrape)")
            return 0
        elif transcript_data:
            title = transcript_data['title']
            pub_date = transcript_data.get('pub_date', 'unknown')
            filename = create_filename_from_title(title, transcript_data.get('pub_date'))
            print(f"\n✓ Successfully scraped: {title}")
            print(f"  Publication date: {pub_date}")
            print(f"  Saved to: {os.path.join(TRANSCRIPTS_DIR, filename)}")
            url_cache.save_cache()
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
    successful_episodes, failed_urls = scrape_all_episodes(episode_urls, url_cache, args.force)
    
    # Step 3: Save URL cache
    url_cache.save_cache()
    print(f"\n✓ URL cache updated ({url_cache.get_cache_size()} total entries)")
    
    # Step 4: Save failed URLs for future retry
    if failed_urls:
        save_failed_urls(failed_urls, args.failed_file)
    elif os.path.exists(args.failed_file) and not args.retry_failed:
        # Clear the failed file if we had no failures on a full run
        os.remove(args.failed_file)
        print(f"✓ Cleared {args.failed_file} (no failures)")
    
    if not successful_episodes and not url_cache.get_cache_size():
        print("\n✗ No episodes were successfully scraped.")
        return 1
    
    print("\n" + "="*70)
    print("Process complete!")
    print(f"Transcripts saved in '{TRANSCRIPTS_DIR}/' directory")
    if failed_urls:
        print(f"Failed URLs saved to '{args.failed_file}'")
        print(f"  Retry with: python scrape_rationalreminder.py --retry-failed")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
