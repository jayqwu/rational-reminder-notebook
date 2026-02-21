#!/usr/bin/env python3
"""Compile sources into category-specific markdown files using semantic similarity"""

import argparse
import csv
import json
import os
import hashlib
from dotenv import load_dotenv
import torch
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import shutil
import re
from url_cache import normalize_cache_url

DEFAULT_SOURCE_DIRS = ["output/rational_reminder", "output/kitces"]
METRICS_FILE = "output/youtube_metrics.csv"
TAXONOMY_PATH = Path("taxonomy.json")
OUTPUT_MATCHES_CSV = Path("output/episode_category_matches.csv")
OUTPUT_SIMILARITIES_CSV = Path("output/episode_similarity_matrix.csv")
CACHE_FILE = Path("output/cache.json")

# Similarity thresholds for relevance levels (these can be tuned based on observed score distribution)
SIMILARITY_VERY_HIGH = 0.5
SIMILARITY_HIGH = 0.4
SIMILARITY_AVERAGE = 0.3
SIMILARITY_LOW = 0.2

# Minimum number of words in a content paragraph to appending episode number for citation purposes
CITATION_LEN = 35

# Weights for combining title and summary similarities (can be tuned)
WEIGHT_TITLE = 0.20
WEIGHT_SUMMARY = 0.80

# Exclude episodes whose titles contain any of these strings (case-insensitive)
EXCLUDE_TITLES = [
    "Year in Review",
    "Scott Galloway",
    "Comprehensive Overview",
    "Retrospective",
    "Episode 389: How the Rational Reminder Podcast is Made",
    "Bonus Episode: Jim Watson: Building a Future for the City: An Interview with the Mayor of Ottawa",
]

# GPU/embedding controls tuned for laptop GPUs
USE_FP16 = True
BATCH_SIZE = 8
MODEL='Octen/Octen-Embedding-0.6B'

def load_taxonomy(path):
    """Load taxonomy from JSON file.
    
    Returns:
        list: List of taxonomy entries with category, subcategory, and description.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("Taxonomy must be a non-empty JSON array.")
    
    # Validate structure
    for entry in data:
        if not all(key in entry for key in ['category', 'subcategory', 'description']):
            raise ValueError("Each taxonomy entry must have 'category', 'subcategory', and 'description'.")
    
    return data

def _normalize_transcript_paragraphs(transcript_text):
    if isinstance(transcript_text, list):
        return [str(paragraph) for paragraph in transcript_text]
    if transcript_text is None:
        return []
    return [str(transcript_text)]

def compute_embeddings(model, texts, show_progress=False, batch_size=None):
    """Compute embeddings for a list of texts.
    
    Args:
        model: SentenceTransformer model
        texts: List of text strings to encode
        show_progress: Whether to show progress bar
        batch_size: Optional batch size for encoding
    
    Returns:
        Tensor of embeddings
    """
    encode_kwargs = {'convert_to_tensor': True, 'show_progress_bar': show_progress}
    if batch_size:
        encode_kwargs['batch_size'] = batch_size
    return model.encode(texts, **encode_kwargs)

def create_episode_text(title, summary):
    """Create combined text representation of an episode for semantic matching.
    
    Args:
        title: episode title
        summary: List of summary points from the episode (can be None or empty)
    
    Returns:
        Combined text string
    """
    parts = [title] if title else []
    
    # Add summary points if available
    if summary:
        if isinstance(summary, list):
            # Filter out empty or None values
            valid_points = [str(point) for point in summary if point]
            parts.extend(valid_points)
        else:
            point_str = str(summary).strip()
            if point_str:
                parts.append(point_str)
    
    return " ".join(parts) if parts else title or ""

def similarity_to_relevance(score):
    """Convert numerical similarity score to descriptive relevance level.
    
    Args:
        score: Cosine similarity score (0-1)
    
    Returns:
        String describing relevance level for RAG retrieval
    """

    if score >= SIMILARITY_VERY_HIGH:
        return "Very High"
    elif score >= SIMILARITY_HIGH:
        return "High"
    elif score >= SIMILARITY_AVERAGE:
        return "Average"
    elif score >= SIMILARITY_LOW:
        return "Low"
    else:
        return "Very Low"

def popularity_to_tier(value):
    """Convert audience popularity percentile to a tier label."""
    if value >= 80:
        return "Very High"
    elif value >= 60:
        return "High"
    elif value >= 40:
        return "Average"
    elif value >= 20:
        return "Low"
    else:
        return "Very Low"

def append_episode_markdown(target, episode, include_content=False):
    """Append episode markdown to target list.

    Args:
        target: list of markdown strings to append to
        episode: episode data dictionary
        include_content: whether to append episode content paragraphs
    """
    target.append(f"## {episode['title']}\n\n")
    target.append(f"Published date: {episode.get('published_date', 'N/A')}\n\n")
    target.append(f"Episode URL: {episode.get('url', 'N/A')}\n\n")

    # Extract episode number in the format "Ep. #"
    episode_title = episode.get('title', '')
    match = re.search(r'\bEpisode (\d+)', episode_title)
    citation = None
    for pattern, prefix in [(r'\bEpisode (\d+)', "Ep."), (r'\bUnderstanding Crypto (\d+)', "UC")]:
        match = re.search(pattern, episode_title)
        if match:
            citation = f"{prefix} {match.group(1)}"
            break

    if episode.get('popularity_percentile', -1) >= 0:
        target.append(
            f"Audience Popularity: popularity_to_tier({episode['popularity_percentile']})\n\n"
        )

    target.append(f"Category relevance: {episode.get('category_relevance', 'N/A')}\n\n")
    if episode.get('related_categories'):
        target.append("Related categories:\n\n")
        for related_cat in episode['related_categories']:
            target.append(f"- {related_cat}\n")
        target.append("\n")

    if episode.get('summary'):
        target.append("### Summary\n\n")
        for item in episode['summary']:
            target.append(f"- {item}\n")
        target.append("\n")

    if include_content:
        target.append("### Content\n\n")
        for paragraph in episode['content']:
            if len(paragraph.split()) > CITATION_LEN and citation:
                paragraph += f" ({citation})"
            target.append(f"{paragraph}\n\n")

    target.append("---\n\n")

def assign_category_by_similarity(episode_data, taxonomy, category_embeddings, model):
    """Assign a category to an episode based on semantic similarity.
    
    Args:
        episode_data: Dictionary containing episode information
        taxonomy: List of taxonomy entries
        category_embeddings: Pre-computed embeddings for category descriptions
        model: SentenceTransformer model
    
    Returns:
        Tuple of (category_label, similarity_score, description, related_categories)
    """
    # Create episode text representation
    title = episode_data.get('title', '')
    summary = episode_data.get('summary', [])

    title_text = (title or "").strip()
    summary_text = create_episode_text("", summary).strip()
    if not summary_text:
        summary_text = title_text

    # Encode title and summary separately
    title_embedding = model.encode(title_text, convert_to_tensor=True, batch_size=BATCH_SIZE)
    summary_embedding = model.encode(summary_text, convert_to_tensor=True, batch_size=BATCH_SIZE)

    # Compute cosine similarities and combine with equal weights
    title_similarities = util.cos_sim(title_embedding, category_embeddings)[0]
    summary_similarities = util.cos_sim(summary_embedding, category_embeddings)[0]
    similarities = WEIGHT_TITLE * title_similarities + WEIGHT_SUMMARY * summary_similarities
    
    # Find best match
    best_idx = similarities.argmax().item()
    best_score = similarities[best_idx].item()
    
    # Get category info
    best_entry = taxonomy[best_idx]
    category_label = f"{best_entry['category']} - {best_entry['subcategory']}"
    description = best_entry['description']

    # Get top 3 other categories with similarity > SIMILARITY_HIGH
    other_matches = []
    for idx, score in enumerate(similarities):
        if idx != best_idx and score.item() > SIMILARITY_HIGH:
            other_matches.append((idx, score.item()))

    other_matches.sort(key=lambda x: x[1], reverse=True)
    related_categories = [
        f"{taxonomy[idx]['category']} - {taxonomy[idx]['subcategory']}"
        for idx, _ in other_matches[:3]
    ]
    
    return category_label, best_score, description, related_categories

def _format_published_date(value):
    if not value:
        return ""
    if isinstance(value, str) and len(value) == 6 and value.isdigit():
        year = int(value[:2])
        month = value[2:4]
        day = value[4:6]
        year_full = 2000 + year if year < 70 else 1900 + year
        return f"{year_full:04d}-{month}-{day}"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return ""
    return parsed.date().isoformat()


def _build_episode_cache_key(data, episode_file):
    source_url = normalize_cache_url(data.get("url"))
    if source_url:
        return source_url
    raise ValueError(f"Episode missing valid URL for cache key: {episode_file}")

def _taxonomy_hash(taxonomy):
    serialized = json.dumps(taxonomy, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

def load_match_cache(path):
    if not path.exists():
        return {
            "taxonomy_hash": "",
            "matches": {},
        }

    with open(path, 'r', encoding='utf-8') as f:
        cache_data = json.load(f)

    if not isinstance(cache_data, dict):
        raise ValueError("Unified cache file must be a JSON object")

    taxonomy_hash = cache_data.get("_taxonomy_hash", "")
    if not isinstance(taxonomy_hash, str):
        taxonomy_hash = ""

    matches = {}

    # Flat shape: top-level URL-key entries with category fields in each URL object.
    for episode_key, entry in cache_data.items():
        if not isinstance(episode_key, str) or not isinstance(entry, dict):
            continue
        if not episode_key.startswith(("https://", "http://")):
            continue

        if "category" not in entry or "subcategory" not in entry:
            continue

        related_categories = entry.get("related_categories", [])
        if not isinstance(related_categories, list):
            related_categories = []

        matches[episode_key] = {
            "category": entry["category"],
            "subcategory": entry["subcategory"],
            "category_relevance": entry.get("category_relevance", "N/A"),
            "related_categories": related_categories,
        }

    return {
        "taxonomy_hash": taxonomy_hash,
        "matches": matches,
    }

def save_match_cache(path, taxonomy_hash, matches):
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        if not isinstance(cache_data, dict):
            cache_data = {}
    else:
        cache_data = {}

    cache_data["_taxonomy_hash"] = taxonomy_hash

    for cache_key, match_data in matches.items():
        normalized_key = normalize_cache_url(cache_key)
        if not normalized_key:
            continue
        existing_entry = cache_data.get(normalized_key)
        entry = dict(existing_entry) if isinstance(existing_entry, dict) else {}

        entry["category"] = match_data.get("category")
        entry["subcategory"] = match_data.get("subcategory")
        entry["category_relevance"] = match_data.get("category_relevance")
        related_categories = match_data.get("related_categories", [])
        entry["related_categories"] = related_categories if isinstance(related_categories, list) else []

        cache_data[normalized_key] = entry

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2)

def load_percentile_map(metrics_file):
    """Load video_id to percentile mapping from CSV file."""
    percentile_map = {}
    
    if not os.path.exists(metrics_file):
        print(f"Warning: Metrics file not found: {metrics_file}")
        return percentile_map
    
    with open(metrics_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_id = row.get('video_id')
            percentile_str = row.get('percentile', '')
            
            if video_id and percentile_str:
                try:
                    percentile_map[video_id] = float(percentile_str)
                except ValueError:
                    continue
    
    return percentile_map

def parse_args():
    parser = argparse.ArgumentParser(
        description="Categorize podcast transcripts filtered by popularity percentile"
    )
    
    # Create mutually exclusive group for source selection
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--rr-only",
        action="store_true",
        help="Only use content from The Rational Reminder podcast"
    )
    source_group.add_argument(
        "--source-dirs",
        nargs="+",
        help=f"Source directories containing JSON files (default: {' '.join(DEFAULT_SOURCE_DIRS)})"
    )
    
    parser.add_argument(
        "--min-percentile",
        type=float,
        default=-1,
        help="Minimum percentile threshold for including episodes (default: off)"
    )
    parser.add_argument(
        "--metrics-file",
        default=METRICS_FILE,
        help=f"YouTube metrics CSV file (default: {METRICS_FILE})"
    )
    parser.add_argument(
        "--recategorize",
        action="store_true",
        help="Force recategorization for all filtered episodes using semantic similarity"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    use_cache = not args.recategorize

    load_dotenv()

    # Handle mutually exclusive source options
    if args.rr_only:
        args.source_dirs = ["output/rational_reminder"]
    elif args.source_dirs is None:
        args.source_dirs = DEFAULT_SOURCE_DIRS 
    
    # Load taxonomy
    print("Loading taxonomy...")
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    print(f"✓ Loaded {len(taxonomy)} categories\n")
    
    # Load percentile mapping (needed for both cache and non-cache paths)
    print(f"Loading popularity metrics from {args.metrics_file}...")
    percentile_map = load_percentile_map(args.metrics_file)
    print(f"✓ Loaded percentiles for {len(percentile_map)} videos\n")

    # Create category label to description mapping
    category_to_description = {
        f"{entry['category']} - {entry['subcategory']}": entry['description']
        for entry in taxonomy
    }

    # Collect all JSON files from all source directories
    episode_files = []
    for source_dir in args.source_dirs:
        source_path = Path(source_dir)
        if source_path.exists():
            dir_files = sorted(source_path.glob("*.json"))
            episode_files.extend(dir_files)
            print(f"Found {len(dir_files)} files in {source_dir}")
        else:
            print(f"Warning: Directory not found: {source_dir}")

    print(f"\nTotal: {len(episode_files)} episode files")

    if args.min_percentile < 0:
        disp_min_perc = "None"
    else:
        disp_min_perc = f"{args.min_percentile}th percentile"

    # Pre-filter episodes by percentile threshold and title exclusions
    print(f"Filtering episodes by percentile threshold ({disp_min_perc}) and title exclusions...")
    filtered_episode_data = []
    episodes_skipped_percentile = 0
    episodes_excluded_by_title = 0

    for episode_file in tqdm(episode_files, desc="Filtering"):
        with open(episode_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        youtube_data = data.get('youtube', {}) or {}
        video_id = youtube_data.get('video_id')

        percentile = None
        if video_id and video_id in percentile_map:
            percentile = percentile_map[video_id]
            if percentile < args.min_percentile:
                episodes_skipped_percentile += 1
                continue

        title = data.get("title", episode_file.stem)
        if EXCLUDE_TITLES:
            title_lower = title.lower()
            if any(exclude.lower() in title_lower for exclude in EXCLUDE_TITLES):
                episodes_excluded_by_title += 1
                continue

        cache_key = _build_episode_cache_key(data, episode_file)
        filtered_episode_data.append({
            "file": episode_file,
            "data": data,
            "title": title,
            "percentile": percentile,
            "cache_key": cache_key,
        })

    print(f"✓ Episodes after filtering: {len(filtered_episode_data)}")
    if episodes_skipped_percentile > 0:
        print(f"  Skipped (percentile < {args.min_percentile}): {episodes_skipped_percentile}")
    if episodes_excluded_by_title > 0:
        print(f"  Excluded by title filter: {episodes_excluded_by_title}")

    current_taxonomy_hash = _taxonomy_hash(taxonomy)
    cache_matches = {}
    cached_taxonomy_hash = ""
    taxonomy_changed = False

    if use_cache:
        print(f"Loading category match cache from {CACHE_FILE}...")
        cache_blob = load_match_cache(CACHE_FILE)
        cache_matches = cache_blob["matches"]
        cached_taxonomy_hash = cache_blob["taxonomy_hash"]
        taxonomy_changed = cached_taxonomy_hash != current_taxonomy_hash

        if taxonomy_changed and cache_matches:
            print("Taxonomy changed since cache creation; rematching all filtered episodes.")
            cache_matches = {}

        print(f"✓ Loaded {len(cache_matches)} cached episode-category matches\n")
    else:
        print("Forced recategorization enabled; matching all filtered episodes.")

    categorized_episodes = defaultdict(list)
    episode_matches = []

    episodes_reused_from_cache = 0
    episodes_to_match = []
    newly_matched_topics = set()

    for item in filtered_episode_data:
        cache_key = item["cache_key"]
        cached_match = cache_matches.get(cache_key) if use_cache else None

        item["is_new_match"] = False

        if cached_match:
            matched_category = cached_match["category"]
            matched_subcategory = cached_match["subcategory"]
            category_label = f"{matched_category} - {matched_subcategory}"

            if category_label not in category_to_description:
                episodes_to_match.append(item)
                continue

            episodes_reused_from_cache += 1
            item["category_label"] = category_label
            item["matched_category"] = matched_category
            item["matched_subcategory"] = matched_subcategory
            item["category_relevance"] = cached_match.get("category_relevance", "Cached")
            item["related_categories"] = cached_match.get("related_categories", [])
        else:
            if use_cache:
                item["is_new_match"] = True
            episodes_to_match.append(item)

    model = None
    category_embeddings = None

    if episodes_to_match:
        print(f"Matching {len(episodes_to_match)} episodes with semantic similarity...")
        print("Initializing semantic similarity model...")

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        print(f"Using device: {device}")

        model = SentenceTransformer(MODEL, trust_remote_code=True, device=device)
        if device == "cuda" and USE_FP16:
            model = model.to(torch.float16)
        print(f"✓ Model loaded: {MODEL}")

        print("Computing category embeddings...")
        category_descriptions = [
            f"{entry['subcategory']}: {entry['description']}"
            for entry in taxonomy
        ]
        category_embeddings = compute_embeddings(
            model,
            category_descriptions,
            show_progress=True,
            batch_size=BATCH_SIZE
        )
        print(f"✓ Computed embeddings for {len(taxonomy)} categories\n")

        for item in tqdm(episodes_to_match, desc="Categorizing"):
            data = item["data"]
            title = item["title"]
            summary = data.get("summary", [])
            content = _normalize_transcript_paragraphs(data.get("content"))

            episode_data = {
                'title': title,
                'summary': summary,
                'content': content
            }
            category_label, similarity_score, _, related_categories = assign_category_by_similarity(
                episode_data,
                taxonomy,
                category_embeddings,
                model,
            )

            matched_category, matched_subcategory = category_label.split(" - ", 1)
            item["category_label"] = category_label
            item["matched_category"] = matched_category
            item["matched_subcategory"] = matched_subcategory
            item["category_relevance"] = similarity_to_relevance(similarity_score)
            item["related_categories"] = related_categories

            if item.get("is_new_match"):
                newly_matched_topics.add(category_label)

            cache_matches[item["cache_key"]] = {
                "category": matched_category,
                "subcategory": matched_subcategory,
                "category_relevance": item["category_relevance"],
                "related_categories": related_categories,
            }
    else:
        print("No semantic matching needed; all filtered episodes resolved from cache.\n")

    for item in filtered_episode_data:
        data = item["data"]
        youtube_data = data.get('youtube', {}) or {}
        title = item["title"]
        summary = data.get("summary", [])
        content = _normalize_transcript_paragraphs(data.get("content"))

        published_date = _format_published_date(data.get('pub_date'))
        episode_url = data.get("url")
        youtube_url = youtube_data.get('url')

        episode_record = {
            'title': title,
            'published_date': published_date,
            'summary': summary,
            'category_relevance': item.get("category_relevance", "N/A"),
            'related_categories': item.get("related_categories", []),
            'content': content,
            'url': episode_url,
            'percentile': item.get("percentile"),
            'youtube': youtube_url
        }

        category_label = item["category_label"]
        categorized_episodes[category_label].append(episode_record)

        episode_matches.append({
            "title": title,
            "category": item["matched_category"],
            "subcategory": item["matched_subcategory"]
        })

    print("\n✓ Categorization complete")
    print(f"  episodes after filtering: {len(filtered_episode_data)}")
    print(f"  reused from cache: {episodes_reused_from_cache}")
    print(f"  newly matched: {len(episodes_to_match)}")
    if use_cache and taxonomy_changed:
        print("  rematch reason: taxonomy changed")
    if use_cache:
        if newly_matched_topics:
            print("  topics with newly matched episodes:")
            for topic in sorted(newly_matched_topics):
                print(f"    - {topic}")
        else:
            print("  topics with newly matched episodes: none")

    print(f"Saving category match cache to {CACHE_FILE}...")
    save_match_cache(CACHE_FILE, current_taxonomy_hash, cache_matches)
    print("✓ Cache saved\n")

    OUTPUT_MATCHES_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MATCHES_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "category", "subcategory"])
        writer.writeheader()
        writer.writerows(episode_matches)
    
    # Generate markdown files for each category
    FULL_OUTPUT_DIR = Path("output/categorized")
    SUMMARY_OUTPUT_DIR = Path("output/summaries")   
    
    FULL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Track statistics for CSV output
    category_statistics = []
    
    for category, episodes in categorized_episodes.items():
        # Sort episodes by similarity score (highest first)
        episodes.sort(key=lambda ep: ep.get('category_relevance', ''), reverse=True)
        
        # Create human-readable filename from category
        base_filename = category.replace("/", "-").replace(":", " -")
        full_path = FULL_OUTPUT_DIR / f"{base_filename}.md"
        summary_path = SUMMARY_OUTPUT_DIR / f"Summary - {base_filename}.md"
        episode_batch = episodes
        
        markdown_summary = []
        markdown_full = []
        
        # Get description for this category
        description = category_to_description.get(category, "")

        # Add category header and description to full markdown
        markdown_full.append(f"# {category}\n\n")
        markdown_full.append(f"## Topic Description\n\n{description}\n\n")
        markdown_full.append(f"Pieces of Content: {len(episode_batch)}\n\n")
        
        if args.rr_only:
            markdown_full.append(
                "Source: [Rational Reminder Podcast](https://rationalreminder.ca/podcast/)\n\n"
            )
        else:
            markdown_full.append(
                "Source: [Rational Reminder Podcast](https://rationalreminder.ca/podcast/) and "
                "[Kitces](https://www.kitces.com/)\n\n"
            )

        # Header section for summary markdown is the same
        markdown_summary = markdown_full.copy()
        
        # Add episodes
        for episode in episode_batch:
            append_episode_markdown(markdown_summary, episode, include_content=False)
            append_episode_markdown(markdown_full, episode, include_content=True)

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.writelines(markdown_summary)

        with open(full_path, 'w', encoding='utf-8') as f:
            f.writelines(markdown_full)
        
        # Get file size
        file_size_mb = os.path.getsize(full_path) / (1024 * 1024)
        word_count = sum(len(line.split()) for line in markdown_full)
        if word_count > 200000:
            print(f"  ⚠ Warning: {category} exceeded word limit with {word_count:,} words")
        if file_size_mb > 200:
            print(f"  ⚠ Warning: {category} exceeded file size limit with {file_size_mb:.1f} MB")
        
        # Collect statistics for CSV
        cat_name, subcat_name = category.split(" - ", 1)
        
        category_statistics.append({
            'Category': cat_name,
            'Subcategory': subcat_name,
            '#': len(episode_batch),
            'WC': word_count,
            'Size': round(file_size_mb, 2),
        })
    
    # Print summary
    print("\n" + "="*95)
    print("Categorization Summary")
    print("="*95)
    
    # Sort by word count (descending)
    sorted_stats = sorted(category_statistics, key=lambda x: x['WC'], reverse=True)
    
    # Print header
    print(f"{'Topics':<22} {'Subtopics':<52} {'Content':>8} {'Words':>10}")
    print("-" * 95)
    
    # Print each category with consistent column widths
    for stat in sorted_stats:
        category_label = f"{stat['Category']}"
        subcategory_label = f"{stat['Subcategory']}"
        episode_count = stat['#']
        word_count = stat['WC']
        print(f"{category_label:<22} {subcategory_label:<52} {episode_count:>8} {word_count:>10,}")
    print("="*95)
    
    # Write category statistics to CSV
    STATS_CSV = Path("output/category_statistics.csv")
    print(f"\nWriting category statistics to {STATS_CSV}...")
    STATS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(STATS_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'Category',
            'Subcategory',
            '#',
            'WC',
            'Size',
        ])
        writer.writeheader()
        writer.writerows(category_statistics)
    print(f"✓ Wrote category statistics to {STATS_CSV}")

if __name__ == "__main__":
    main()
