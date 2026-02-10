#!/usr/bin/env python3
"""
Shared URL cache module for tracking scraped URLs across different scrapers.
Prevents duplicate scraping and allows force-rescrape functionality.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Set

URL_CACHE_FILE = "output/url_index.json"

class URLCache:
    """Manages a cache of scraped URLs to avoid duplicate scraping."""
    
    def __init__(self, cache_file: str = URL_CACHE_FILE):
        """Initialize the URL cache.
        
        Args:
            cache_file: Path to the JSON file storing URL to file mappings
        """
        self.cache_file = cache_file
        self.url_to_file: Dict[str, str] = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load the cache from disk if it exists."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.url_to_file = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load URL cache from {self.cache_file}: {e}")
                self.url_to_file = {}
        else:
            # Initialize empty cache
            self.url_to_file = {}
    
    def save_cache(self):
        """Save the current cache to disk."""
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.url_to_file, f, indent=2, sort_keys=True)
        except IOError as e:
            print(f"Error: Could not save URL cache to {self.cache_file}: {e}")
    
    def is_cached(self, url: str) -> bool:
        """Check if a URL has already been scraped.
        
        Args:
            url: The URL to check
            
        Returns:
            True if the URL exists in cache, False otherwise
        """
        return url in self.url_to_file
    
    def get_cached_file(self, url: str) -> Optional[str]:
        """Get the file path for a cached URL.
        
        Args:
            url: The URL to look up
            
        Returns:
            File path if URL is cached, None otherwise
        """
        return self.url_to_file.get(url)
    
    def add_url(self, url: str, file_path: str):
        """Add a URL and its corresponding file to the cache.
        
        Args:
            url: The URL that was scraped
            file_path: Path to the file where content was saved, or "FILTERED" for filtered URLs
        """
        self.url_to_file[url] = file_path
    
    def add_filtered_url(self, url: str):
        """Mark a URL as filtered (skipped due to category/content filtering).
        
        Args:
            url: The URL that was filtered out
        """
        self.url_to_file[url] = "FILTERED"
    
    def is_filtered(self, url: str) -> bool:
        """Check if a URL was filtered (not actually scraped).
        
        Args:
            url: The URL to check
            
        Returns:
            True if URL was filtered, False otherwise
        """
        return self.url_to_file.get(url) == "FILTERED"
    
    def remove_url(self, url: str) -> bool:
        """Remove a URL from the cache.
        
        Args:
            url: The URL to remove
            
        Returns:
            True if URL was removed, False if it wasn't in cache
        """
        if url in self.url_to_file:
            del self.url_to_file[url]
            return True
        return False
    
    def get_all_urls(self) -> Set[str]:
        """Get set of all cached URLs.
        
        Returns:
            Set of all URLs in the cache
        """
        return set(self.url_to_file.keys())
    
    def get_cache_size(self) -> int:
        """Get the number of URLs in the cache.
        
        Returns:
            Number of cached URLs
        """
        return len(self.url_to_file)
    
    def clear_cache(self):
        """Clear all entries from the cache."""
        self.url_to_file = {}