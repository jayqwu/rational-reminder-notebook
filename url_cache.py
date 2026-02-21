#!/usr/bin/env python3
"""
Shared URL cache module for tracking scraped URLs across different scrapers.
Prevents duplicate scraping and allows force-rescrape functionality.
"""

import json
import os
from typing import Dict, Optional, Set
from urllib.parse import urlsplit, urlunsplit

URL_CACHE_FILE = "output/cache.json"


def normalize_cache_url(url: str) -> Optional[str]:
    """Normalize URL for cache key usage."""
    if not url:
        return None
    try:
        parsed = urlsplit(str(url).strip())
    except Exception:
        return None
    if not parsed.netloc:
        return None
    normalized = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        query="",
        fragment="",
    )
    normalized_url = urlunsplit(normalized).rstrip("/")
    return normalized_url or None


def _is_url_key(value: str) -> bool:
    if not isinstance(value, str):
        return False
    return value.startswith("https://") or value.startswith("http://")

class URLCache:
    """Manages a cache of scraped URLs to avoid duplicate scraping."""
    
    def __init__(self, cache_file: str = URL_CACHE_FILE):
        """Initialize the URL cache.
        
        Args:
            cache_file: Path to the JSON file storing URL to file mappings
        """
        self.cache_file = cache_file
        self.url_entries: Dict[str, Dict[str, object]] = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load the cache from disk if it exists."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                if not isinstance(cache_data, dict):
                    raise ValueError("Cache must be a JSON object")

                self.url_entries = {}

                # Flat shape: top-level URL keys
                for url, entry in cache_data.items():
                    if not _is_url_key(url) or not isinstance(entry, dict):
                        continue
                    normalized_url = normalize_cache_url(url)
                    if not normalized_url:
                        continue

                    existing = self.url_entries.get(normalized_url, {})
                    merged = dict(existing)
                    merged.update(entry)
                    self.url_entries[normalized_url] = merged
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load URL cache from {self.cache_file}: {e}")
                self.url_entries = {}
            except ValueError as e:
                print(f"Warning: Could not load URL cache from {self.cache_file}: {e}")
                self.url_entries = {}
        else:
            # Initialize empty cache
            self.url_entries = {}

    def _load_full_cache_data(self) -> Dict[str, object]:
        if not os.path.exists(self.cache_file):
            return {}

        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

        if not isinstance(cache_data, dict):
            return {}
        return cache_data
    
    def save_cache(self):
        """Save the current cache to disk."""
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

        try:
            cache_data = self._load_full_cache_data()

            for url, entry in self.url_entries.items():
                if not _is_url_key(url):
                    continue
                existing = cache_data.get(url)
                base = dict(existing) if isinstance(existing, dict) else {}
                if not isinstance(entry, dict):
                    entry = {}
                if "status" in entry:
                    base["status"] = entry["status"]
                else:
                    base.pop("status", None)
                if "file" in entry:
                    base["file"] = entry["file"]
                else:
                    base.pop("file", None)
                cache_data[url] = base

            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, sort_keys=True)
        except IOError as e:
            print(f"Error: Could not save URL cache to {self.cache_file}: {e}")
    
    def is_cached(self, url: str) -> bool:
        """Check if a URL has already been scraped.
        
        Args:
            url: The URL to check
            
        Returns:
            True if the URL exists in cache, False otherwise
        """
        normalized_url = normalize_cache_url(url)
        if not normalized_url:
            return False
        entry = self.url_entries.get(normalized_url)
        if not isinstance(entry, dict):
            return False
        return entry.get("status") in {"filtered", "scraped"}
    
    def get_cached_file(self, url: str) -> Optional[str]:
        """Get the file path for a cached URL.
        
        Args:
            url: The URL to look up
            
        Returns:
            File path if URL is cached, None otherwise
        """
        normalized_url = normalize_cache_url(url)
        if not normalized_url:
            return None
        entry = self.url_entries.get(normalized_url)
        if not entry:
            return None
        if entry.get("status") == "filtered":
            return "FILTERED"
        file_path = entry.get("file")
        if isinstance(file_path, str):
            return file_path
        return None
    
    def add_url(self, url: str, file_path: str):
        """Add a URL and its corresponding file to the cache.
        
        Args:
            url: The URL that was scraped
            file_path: Path to the file where content was saved, or "FILTERED" for filtered URLs
        """
        normalized_url = normalize_cache_url(url)
        if not normalized_url:
            return
        existing = self.url_entries.get(normalized_url)
        base = dict(existing) if isinstance(existing, dict) else {}
        base.update({
            "status": "scraped",
            "file": file_path,
        })
        self.url_entries[normalized_url] = base
    
    def add_filtered_url(self, url: str):
        """Mark a URL as filtered (skipped due to category/content filtering).
        
        Args:
            url: The URL that was filtered out
        """
        normalized_url = normalize_cache_url(url)
        if not normalized_url:
            return
        existing = self.url_entries.get(normalized_url)
        base = dict(existing) if isinstance(existing, dict) else {}
        base["status"] = "filtered"
        base.pop("file", None)
        self.url_entries[normalized_url] = base
    
    def is_filtered(self, url: str) -> bool:
        """Check if a URL was filtered (not actually scraped).
        
        Args:
            url: The URL to check
            
        Returns:
            True if URL was filtered, False otherwise
        """
        normalized_url = normalize_cache_url(url)
        if not normalized_url:
            return False
        entry = self.url_entries.get(normalized_url)
        return bool(entry and entry.get("status") == "filtered")
    
    def remove_url(self, url: str) -> bool:
        """Remove a URL from the cache.
        
        Args:
            url: The URL to remove
            
        Returns:
            True if URL was removed, False if it wasn't in cache
        """
        normalized_url = normalize_cache_url(url)
        if not normalized_url:
            return False
        if normalized_url in self.url_entries:
            del self.url_entries[normalized_url]
            return True
        return False
    
    def get_all_urls(self) -> Set[str]:
        """Get set of all cached URLs.
        
        Returns:
            Set of all URLs in the cache
        """
        return set(self.url_entries.keys())
    
    def get_cache_size(self) -> int:
        """Get the number of URLs in the cache.
        
        Returns:
            Number of cached URLs
        """
        return len(self.url_entries)
    
    def clear_cache(self):
        """Clear all entries from the cache."""
        self.url_entries = {}