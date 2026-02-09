#!/usr/bin/env python3
"""Fetch YouTube metrics for videos referenced in transcript files.

Scans transcript JSON files, extracts YouTube video IDs, and fetches metrics.
Uses transcript titles as identifiers for linking.
Requires YOUTUBE_API_KEY environment variable for fresh data fetch.

Usage:
  python fetch_youtube_metrics.py                                    # Fetch from YouTube and cache
  python fetch_youtube_metrics.py --use-cache                        # Use cache if available
  python fetch_youtube_metrics.py --cache-only                       # Error if cache doesn't exist
"""

import argparse
import csv
import glob
import json
import bisect
import math
import os
import sys
import time
from datetime import datetime, timedelta
from statistics import mean, stdev
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    # Fallback if dotenv not installed
    load_dotenv = lambda path=None: None

import requests

API_BASE = "https://www.googleapis.com/youtube/v3"
CACHE_FILE = "output/youtube_data_cache.json"
TRANSCRIPTS_DIR = "output/rational_reminder"
SLEEP_BETWEEN_REQUESTS = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch YouTube metrics for videos in transcript files")
    parser.add_argument("--output", default="output/youtube_metrics.csv", help="Output CSV file")
    parser.add_argument("--use-cache", action="store_true", help="Use cached YouTube data if available, otherwise fetch")
    parser.add_argument("--cache-only", action="store_true", help="Only use cached data, error if cache doesn't exist")
    parser.add_argument("--cache-file", default=CACHE_FILE, help="Cache file path")
    return parser.parse_args()


# ============================================================================
# Transcript Scanning Functions
# ============================================================================

def scan_transcript_files(transcripts_dir: str) -> List[Dict[str, str]]:
    """Scan transcript JSON files and extract video IDs and titles.
    
    Returns:
        List of dicts with 'video_id' and 'title' keys
    """
    video_refs = []
    
    json_files = glob.glob(os.path.join(transcripts_dir, '*.json'))
    print(f"Scanning {len(json_files)} transcript files...")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if this transcript has YouTube metadata
            youtube_data = data.get('youtube')
            if youtube_data and 'video_id' in youtube_data:
                video_id = youtube_data['video_id']
                title = data.get('title', '')
                
                video_refs.append({
                    'video_id': video_id,
                    'title': title
                })
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read {json_file}: {e}")
            continue
    
    print(f"✓ Found {len(video_refs)} videos with YouTube metadata")
    return video_refs


# ============================================================================
# YouTube API Functions
# ============================================================================

def api_get(endpoint: str, params: Dict[str, str]) -> Dict:
    url = f"{API_BASE}/{endpoint}"
    response = requests.get(url, params=params, timeout=20)
    if response.status_code != 200:
        try:
            payload = response.json()
        except ValueError:
            payload = {"error": {"message": response.text}}
        message = payload.get("error", {}).get("message", "Unknown error")
        raise RuntimeError(f"YouTube API error: {message}")
    return response.json()


def chunked(items: List[str], size: int) -> List[List[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def save_youtube_cache(videos_dict: Dict[str, Dict], cache_file: str) -> None:
    """Save raw YouTube video data to cache file (keyed by video_id)."""
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(videos_dict, f, indent=2)
    print(f"✓ Cached {len(videos_dict)} videos to {cache_file}")


def load_youtube_cache(cache_file: str) -> Optional[Dict[str, Dict]]:
    """Load cached YouTube video data (dict keyed by video_id).
    
    Handles both old list format and new dict format.
    """
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle old list format - convert to dict
        if isinstance(data, list):
            videos_dict = {v['video_id']: v for v in data if 'video_id' in v}
            print(f"✓ Loaded {len(videos_dict)} videos from cache (converted from list format)")
            return videos_dict
        elif isinstance(data, dict):
            print(f"✓ Loaded {len(data)} videos from cache ({cache_file})")
            return data
        else:
            print(f"Warning: Unexpected cache format: {type(data)}")
            return None
            
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load cache: {e}")
        return None


def fetch_video_stats(api_key: str, video_ids: List[str], sleep_s: float) -> Dict[str, Dict[str, str]]:
    """Fetch YouTube stats for specific video IDs.
    
    Returns:
        Dict keyed by video_id with stats as values
    """
    results: Dict[str, Dict[str, str]] = {}
    
    print(f"Fetching stats for {len(video_ids)} videos from YouTube API...")
    for batch in chunked(video_ids, 50):
        params = {
            "part": "snippet,statistics",
            "id": ",".join(batch),
            "key": api_key,
        }
        data = api_get("videos", params)
        for item in data.get("items", []):
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            
            video_id = item.get("id", "")
            results[video_id] = {
                "video_id": video_id,
                "published_at": snippet.get("publishedAt", ""),
                "view_count": stats.get("viewCount", "0"),
                "url": f"https://www.youtube.com/watch?v={video_id}",
            }
        time.sleep(sleep_s)

    return results


# ============================================================================
# Metrics Calculation Functions
# ============================================================================


def calculate_baseline(videos: List[Dict], target_index: int, 
                      earliest_date: datetime, latest_date: datetime,
                      edge_start_avg: float, edge_end_avg: float,
                      half_window_days: int = 180) -> float:
    """Calculate baseline using symmetric rolling window with edge handling.
    
    Args:
        videos: List of all videos
        target_index: Index of target video
        earliest_date: Earliest video publication date
        latest_date: Latest video publication date
        edge_start_avg: Baseline for videos at the start edge
        edge_end_avg: Baseline for videos at the end edge
        half_window_days: Half-width of the rolling window (default 180 days)
    
    Returns:
        Baseline view count for the target video
    """
    target_date = datetime.fromisoformat(videos[target_index]['published_at'].replace('Z', '+00:00'))
    
    # Check if this video is in the start edge (first half-window period)
    if (target_date - earliest_date).days < half_window_days:
        return edge_start_avg
    
    # Check if this video is in the end edge (last half-window period)
    if (latest_date - target_date).days < half_window_days:
        return edge_end_avg
    
    # Calculate rolling average (videos within ±half_window)
    start_date = target_date - timedelta(days=half_window_days)
    end_date = target_date + timedelta(days=half_window_days)
    
    window_views = []
    for video in videos:
        video_date = datetime.fromisoformat(video['published_at'].replace('Z', '+00:00'))
        if start_date <= video_date <= end_date and video != videos[target_index]:
            try:
                views = int(video['view_count'])
                window_views.append(views)
            except (ValueError, KeyError):
                continue
    
    if not window_views:
        # Fallback to edge average if no videos in window
        return edge_start_avg if target_date < (earliest_date + latest_date) / 2 else edge_end_avg
    
    return mean(window_views)


def calculate_percentile(value: float, sorted_values: List[float]) -> float:
    """Calculate percentile rank (0-100) using right-inclusive ranking."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return 100.0
    rank = bisect.bisect_right(sorted_values, value)
    return 100.0 * (rank - 1) / (len(sorted_values) - 1)


def calculate_metrics(videos: List[Dict]) -> List[Dict]:
    """Calculate baselines, popularity scores, and percentile ranks."""
    # Sort videos by publication date
    videos_sorted = sorted(videos, key=lambda x: x['published_at'])
    earliest_date = datetime.fromisoformat(videos_sorted[0]['published_at'].replace('Z', '+00:00'))
    latest_date = datetime.fromisoformat(videos_sorted[-1]['published_at'].replace('Z', '+00:00'))
    
    # Define rolling window half-width
    half_window_days = 180
    
    # Calculate edge baselines for start and end periods
    edge_start_cutoff = earliest_date + timedelta(days=half_window_days)
    edge_end_cutoff = latest_date - timedelta(days=half_window_days)
    
    # Start edge: average of videos in first half-window period
    edge_start_views = []
    for video in videos_sorted:
        video_date = datetime.fromisoformat(video['published_at'].replace('Z', '+00:00'))
        if video_date <= edge_start_cutoff:
            try:
                views = int(video['view_count'])
                edge_start_views.append(views)
            except (ValueError, KeyError):
                continue
    
    edge_start_avg = mean(edge_start_views) if edge_start_views else 1.0
    
    # End edge: average of videos in last half-window period
    edge_end_views = []
    for video in videos_sorted:
        video_date = datetime.fromisoformat(video['published_at'].replace('Z', '+00:00'))
        if video_date >= edge_end_cutoff:
            try:
                views = int(video['view_count'])
                edge_end_views.append(views)
            except (ValueError, KeyError):
                continue
    
    edge_end_avg = mean(edge_end_views) if edge_end_views else edge_start_avg
    
    print(f"Edge baseline (first {half_window_days} days): {edge_start_avg:.0f} views")
    print(f"Edge baseline (last {half_window_days} days): {edge_end_avg:.0f} views")
    print(f"Rolling window: ±{half_window_days} days")
    
    # Calculate baselines and popularity for each video
    for i, video in enumerate(videos):
        try:
            views = int(video['view_count'])
        except (ValueError, KeyError):
            video['baseline'] = ''
            video['popularity'] = ''
            continue
        
        baseline = calculate_baseline(videos, i, earliest_date, latest_date,
                                     edge_start_avg, edge_end_avg, half_window_days)
        video['baseline'] = f"{baseline:.2f}"
        
        # Popularity = log10(views/baseline)
        if baseline > 0:
            popularity = math.log10(views / baseline)
            video['popularity'] = f"{popularity:.4f}"
        else:
            video['popularity'] = ''
    
    # Calculate percentile ranks of popularity
    popularity_scores = []
    for video in videos:
        if video.get('popularity'):
            try:
                popularity_scores.append(float(video['popularity']))
            except ValueError:
                continue
    
    if popularity_scores:
        sorted_scores = sorted(popularity_scores)
        pop_mean = mean(popularity_scores)
        pop_stdev = stdev(popularity_scores) if len(popularity_scores) > 1 else 1.0
        print(f"Popularity mean: {pop_mean:.4f}, stdev: {pop_stdev:.4f}")

        for video in videos:
            if video.get('popularity'):
                try:
                    popularity = float(video['popularity'])
                    percentile = calculate_percentile(popularity, sorted_scores)
                    video['percentile'] = f"{percentile:.2f}"
                except ValueError:
                    video['percentile'] = ''
            else:
                video['percentile'] = ''
    else:
        for video in videos:
            video['percentile'] = ''
    
    return videos


def write_metrics_csv(path: str, videos: List[Dict]) -> None:
    """Write final metrics CSV."""
    fieldnames = [
        'video_id', 'title', 'published_at', 
        'view_count', 'baseline', 'popularity', 'percentile', 'url'
    ]
    
    # Sort by publication date for better readability
    videos_sorted = sorted(videos, key=lambda x: x.get('published_at', ''))
    
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for video in videos_sorted:
            writer.writerow(video)
    
    print(f"\n✓ Saved metrics to {path}")
    
    # Print stats
    with_popularity = sum(1 for v in videos if v.get('popularity'))
    print(f"  Total videos: {len(videos)}")
    print(f"  Videos with popularity scores: {with_popularity}/{len(videos)}")


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    # Load environment variables from .env file
    load_dotenv()
    
    args = parse_args()
    cache_file = args.cache_file
    
    # Step 1: Scan transcript files to get video IDs and titles
    video_refs = scan_transcript_files(TRANSCRIPTS_DIR)
    if not video_refs:
        print("Error: No videos found in transcript files")
        return 1
    
    # Step 2: Load cache or prepare to fetch
    cached_videos = None
    if args.cache_only:
        if not os.path.exists(cache_file):
            print(f"Error: --cache-only specified but cache file not found: {cache_file}")
            return 1
        cached_videos = load_youtube_cache(cache_file)
        if cached_videos is None:
            return 1
    elif args.use_cache:
        cached_videos = load_youtube_cache(cache_file)
    
    # Step 3: Determine which videos need to be fetched
    videos_dict = cached_videos if cached_videos else {}
    missing_video_ids = [ref['video_id'] for ref in video_refs if ref['video_id'] not in videos_dict]
    
    if missing_video_ids:
        if args.cache_only:
            print(f"Warning: {len(missing_video_ids)} videos not in cache (will be skipped)")
            for vid in missing_video_ids[:5]:
                print(f"  - {vid}")
            if len(missing_video_ids) > 5:
                print(f"  ... and {len(missing_video_ids) - 5} more")
        else:
            print(f"Need to fetch {len(missing_video_ids)} missing videos from YouTube API...")
            
            # Check for API key
            api_key = os.getenv("YOUTUBE_API_KEY")
            if not api_key:
                print("Error: YOUTUBE_API_KEY environment variable is not set.")
                print("Missing video IDs:")
                for vid in missing_video_ids[:10]:
                    print(f"  - {vid}")
                if len(missing_video_ids) > 10:
                    print(f"  ... and {len(missing_video_ids) - 10} more")
                return 1
            
            new_videos = fetch_video_stats(api_key, missing_video_ids, SLEEP_BETWEEN_REQUESTS)
            
            # Merge with cache
            videos_dict.update(new_videos)
            
            # Save updated cache
            save_youtube_cache(videos_dict, cache_file)
    else:
        print("✓ All videos found in cache")
    
    # Step 4: Build final video list with transcript titles
    videos = []
    for ref in video_refs:
        video_id = ref['video_id']
        if video_id in videos_dict:
            video_data = videos_dict[video_id].copy()
            video_data['title'] = ref['title']  # Use transcript title as identifier
            videos.append(video_data)
        else:
            print(f"Warning: Video {video_id} not found in data")
    
    # Step 5: Calculate metrics
    print("Calculating metrics...")
    videos_with_metrics = calculate_metrics(videos)
    
    # Step 6: Write output
    write_metrics_csv(args.output, videos_with_metrics)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
