#!/usr/bin/env python3
"""Compile sources into category-specific markdown files using semantic similarity"""

import argparse
import csv
import json
import os
import torch
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

DEFAULT_SOURCE_DIRS = ["output/rational_reminder", "output/kitces"]
METRICS_FILE = "output/youtube_metrics.csv"
TAXONOMY_PATH = Path("taxonomy.json")
OUTPUT_MATCHES_CSV = Path("output/episode_category_matches.csv")
CACHE_FILE = Path("output/episodes_cache.json")

# Hardcoded switch: Set to True to include full content in embeddings (requires larger model)
# Set to False to use only title and summary (works with smaller models)
INCLUDE_CONTENT_IN_EMBEDDINGS = False

# GPU/embedding controls tuned for laptop GPUs
USE_FP16 = True
BATCH_SIZE = 2
CHUNK_WORDS = 800
CHUNK_OVERLAP = 120

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

def _format_metadata_block(metadata_items, heading_level=2):
    """Format metadata as a simple markdown block (non-YAML)."""
    heading_prefix = "#" * heading_level
    lines = [f"{heading_prefix} Metadata\n"]

    for label, value in metadata_items:
        if value is None or value == "":
            continue
        if isinstance(value, list):
            if not value:
                continue
            lines.append(f"{label}:\n")
            for item in value:
                if item:
                    lines.append(f"- {item}\n")
        else:
            lines.append(f"{label}: {value}\n")

    lines.append("\n")
    return lines

def create_episode_text(title, summary, content=None):
    """Create combined text representation of an episode for semantic matching.
    
    Uses title and summary by default. Optionally includes full content if
    INCLUDE_CONTENT_IN_EMBEDDINGS is True and content is provided.
    
    Args:
        title: episode title
        summary: List of summary points from the episode (can be None or empty)
        content: Full episode content/transcript (can be None, string, or list of strings)
    
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
    
    # Add full content if the switch is enabled
    if INCLUDE_CONTENT_IN_EMBEDDINGS and content:
        if isinstance(content, list):
            # Join list of paragraphs
            valid_paragraphs = [str(para) for para in content if para]
            if valid_paragraphs:
                parts.append(" ".join(valid_paragraphs))
        else:
            content_str = str(content).strip()
            if content_str:
                parts.append(content_str)
    
    return " ".join(parts) if parts else title or ""

def _join_content(content):
    if not content:
        return ""
    if isinstance(content, list):
        return " ".join(str(para) for para in content if para)
    return str(content)

def _chunk_words(text, chunk_size, overlap):
    words = text.split()
    if not words:
        return []
    if chunk_size <= 0:
        return [text]
    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
    return chunks

def build_episode_chunks(title, summary, content):
    base_text = create_episode_text(title, summary, content=None).strip()
    content_text = _join_content(content).strip()

    if INCLUDE_CONTENT_IN_EMBEDDINGS and content_text:
        content_chunks = _chunk_words(content_text, CHUNK_WORDS, CHUNK_OVERLAP)
        if base_text:
            return [f"{base_text}\n{chunk}" for chunk in content_chunks] or [base_text]
        return content_chunks

    return [base_text] if base_text else []

def similarity_to_relevance(score):
    """Convert numerical similarity score to descriptive relevance level.
    
    Args:
        score: Cosine similarity score (0-1)
    
    Returns:
        String describing relevance level for RAG retrieval
    """
    if score >= 0.6:
        return "Very High"
    elif score >= 0.5:
        return "High"
    elif score >= 0.4:
        return "Average"
    elif score >= 0.3:
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

def assign_category_by_similarity(episode_data, taxonomy, category_embeddings, model):
    """Assign a category to an episode based on semantic similarity.
    
    Args:
        episode_data: Dictionary containing episode information
        taxonomy: List of taxonomy entries
        category_embeddings: Pre-computed embeddings for category descriptions
        model: SentenceTransformer model
    
    Returns:
        Tuple of (category_label, similarity_score, description)
    """
    # Create episode text representation
    title = episode_data.get('title', '')
    summary = episode_data.get('summary', [])
    content = episode_data.get('content', [])

    episode_chunks = build_episode_chunks(title, summary, content)
    if not episode_chunks:
        episode_chunks = [title] if title else [""]

    # Encode episode chunks and mean-pool to a single embedding
    chunk_embeddings = model.encode(episode_chunks, convert_to_tensor=True, batch_size=BATCH_SIZE)
    if chunk_embeddings.dim() == 1:
        episode_embedding = chunk_embeddings
    else:
        episode_embedding = chunk_embeddings.mean(dim=0)
    
    # Compute cosine similarities
    similarities = util.cos_sim(episode_embedding, category_embeddings)[0]
    
    # Find best match
    best_idx = similarities.argmax().item()
    best_score = similarities[best_idx].item()
    
    # Get category info
    best_entry = taxonomy[best_idx]
    category_label = f"{best_entry['category']} - {best_entry['subcategory']}"
    description = best_entry['description']
    
    return category_label, best_score, description

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
    parser.add_argument(
        "--source-dirs",
        nargs="+",
        default=DEFAULT_SOURCE_DIRS,
        help=f"Source directories containing JSON files (default: {' '.join(DEFAULT_SOURCE_DIRS)})"
    )
    parser.add_argument(
        "--min-percentile",
        type=float,
        default=-1.0,
        help="Minimum percentile threshold for including episodes (default: -1.0, includes all)"
    )
    parser.add_argument(
        "--metrics-file",
        default=METRICS_FILE,
        help=f"YouTube metrics CSV file (default: {METRICS_FILE})"
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Skip categorization and regenerate markdown from cached categorization (faster)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load taxonomy
    print("Loading taxonomy...")
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    print(f"✓ Loaded {len(taxonomy)} categories\n")
    
    # Load percentile mapping (needed for both cache and non-cache paths)
    print(f"Loading popularity metrics from {args.metrics_file}...")
    percentile_map = load_percentile_map(args.metrics_file)
    print(f"✓ Loaded percentiles for {len(percentile_map)} videos\n")
    
    # Check if using cache
    if args.use_cache:
        if not CACHE_FILE.exists():
            print(f"Error: Cache file not found: {CACHE_FILE}")
            print("Run without --use-cache to perform categorization first.")
            return
        print(f"Loading categorization from cache: {CACHE_FILE}\n")
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # Apply percentile filtering to cached episodes
        categorized_episodes = defaultdict(list)
        total_cached = 0
        episodes_filtered_from_cache = 0
        
        for category, episodes in cache_data.items():
            for episode in episodes:
                total_cached += 1
                
                # Apply min-percentile filter
                percentile = episode.get('percentile', -1.0)
                if percentile >= 0 and percentile < args.min_percentile:
                    episodes_filtered_from_cache += 1
                    continue
                
                categorized_episodes[category].append(episode)
        
        category_to_description = {
            f"{entry['category']} - {entry['subcategory']}": entry['description']
            for entry in taxonomy
        }
        
        episodes_kept = sum(len(eps) for eps in categorized_episodes.values())
        print(f"✓ Loaded {total_cached} episodes from cache")
        if args.min_percentile > -1.0:
            print(f"  Applied percentile filter: kept {episodes_kept}, filtered out {episodes_filtered_from_cache}")
        print()
    else:
        # Initialize semantic similarity model
        print("Initializing semantic similarity model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, device=device)
        if device == "cuda" and USE_FP16:
            model = model.to(torch.float16)
        print("✓ Model loaded: nomic-ai/nomic-embed-text-v1.5\n")
        
        # Pre-compute category embeddings
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
        print(f"Categorizing episodes using semantic similarity...\n")
        
        # Dictionary to store episodes by category
        categorized_episodes = defaultdict(list)
        episode_matches = []
        
        # Process each episode
        episodes_filtered = 0
        episodes_skipped = 0
        
        for episode_file in tqdm(episode_files):
            with open(episode_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check percentile filter (if YouTube data is available)
            youtube_data = data.get('youtube', {})
            video_id = youtube_data.get('video_id') if youtube_data else None
            published_at_raw = data.get('pub_date')
            published_date = _format_published_date(published_at_raw)
            
            # Build YouTube URL if available
            youtube_url = None
            if youtube_data:
                youtube_url = youtube_data.get('url')
                if not youtube_url and video_id:
                    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
            
            # Handle percentile filtering only if percentile data exists
            percentile = None
            if video_id and video_id in percentile_map:
                percentile = percentile_map[video_id]
                if percentile < args.min_percentile:
                    episodes_skipped += 1
                    continue
            # else: No percentile data - include by default
            
            episodes_filtered += 1
            
            title = data.get("title", episode_file.stem)
            # Handle both old field name (transcript) and new field name (content)
            content = _normalize_transcript_paragraphs(
                data.get("content") or data.get("transcript", [])
            )
            episode_url = data.get("url") or data.get("episode_url")
            summary = data.get("summary", [])

            # Assign category using semantic similarity
            episode_data = {
                'title': title,
                'summary': summary,
                'content': content
            }
            category_label, similarity_score, description = assign_category_by_similarity(
                episode_data, taxonomy, category_embeddings, model
            )
            
            # Convert similarity score to descriptive relevance level
            relevance_level = similarity_to_relevance(similarity_score)
            
            episode_record = {
                'title': title,
                'published_date': published_date,
                'summary': summary,
                'category_relevance': relevance_level,
                'content': content,
                'url': episode_url,
                'percentile': percentile if percentile is not None else -1.0,
                'youtube': youtube_url
            }

            categorized_episodes[category_label].append(episode_record)

            if " - " in category_label:
                matched_category, matched_subcategory = category_label.split(" - ", 1)
            else:
                matched_category, matched_subcategory = category_label, ""

            episode_matches.append({
                "title": title,
                "category": matched_category,
                "subcategory": matched_subcategory
            })
        
        print(f"\n✓ Categorization complete")
        print(f"  episodes included: {episodes_filtered}")
        if args.min_percentile > -1.0:
            print(f"  episodes skipped (percentile < {args.min_percentile}): {episodes_skipped}")
        print()
        
        # Save cache for future regeneration
        print(f"Saving categorization cache to {CACHE_FILE}...")
        cache_data = {category: episodes for category, episodes in categorized_episodes.items()}
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)
        print(f"✓ Cache saved\n")
        
        with open(OUTPUT_MATCHES_CSV, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["title", "category", "subcategory"])
            writer.writeheader()
            writer.writerows(episode_matches)
    
    # Generate markdown files for each categoryCACHE_FILE = Path("output/episodes_cache.json")
    OUTPUT_DIR = Path("output/categorized_transcripts")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Track statistics for CSV output
    category_statistics = []
    
    for category, episodes in categorized_episodes.items():
        # Sort episodes by percentile (highest first)
        episodes.sort(key=lambda ep: ep.get('percentile', -1.0), reverse=True)
        
        # Create human-readable filename from category
        base_filename = category.replace("/", "-").replace(":", " -")
        filepath = OUTPUT_DIR / f"{base_filename}.md"
        episode_batch = episodes
        
        markdown_content = []
        
        # Get description for this category
        description = category_to_description.get(category, "")
        
        # Add header first
        markdown_content.append(f"# {category}\n\n")
            
        # Add description as subheading
        if description:
            markdown_content.append(f"## Description\n{description}\n\n")
        
        # Create metadata block for category
        category_metadata = [
            ("Type", "category"),
            ("episodes in file", len(episode_batch)),
            ("Description", description),
            ("Source", "Rational Reminder Podcast"),
            ("Source URL", "https://rationalreminder.ca/podcast/")
        ]

        if args.min_percentile > -1.0:
            category_metadata.append(("Audience Popularity Filter", args.min_percentile))

        markdown_content.extend(_format_metadata_block(category_metadata, heading_level=2))
        
        # Add episodes
        for episode in episode_batch:
                # Episode title heading
                markdown_content.append(f"## {episode['title']}\n\n")
                
                # Create YAML frontmatter for episode
                published_date = episode.get('published_date') or "N/A"
                percentile_value = episode.get('percentile', -1.0)
                if percentile_value >= 0:
                    percentile_display = percentile_value
                else:
                    percentile_display = None
                
                episode_frontmatter = {
                    'type': 'episode',
                    'category': category,
                    'published_date': published_date,
                }
                
                if episode.get('episode_url'):
                    episode_frontmatter['episode_url'] = episode['episode_url']
                if episode.get('youtube_url'):
                    episode_frontmatter['youtube_url'] = episode['youtube_url']
                if percentile_display is not None:
                    episode_frontmatter['audience_popularity_tier'] = popularity_to_tier(percentile_display)
                if episode.get('category_relevance'):
                    episode_frontmatter['category_relevance'] = episode['category_relevance']
                
                # Add summary
                summary_data = episode.get('summary')
                if summary_data and (isinstance(summary_data, list) and len(summary_data) > 0):
                    episode_frontmatter['summary'] = summary_data
                
                # Write metadata block after heading
                # Format summary as bullet points if available
                summary_text = None
                summary_data = episode.get('summary')
                if summary_data and isinstance(summary_data, list) and len(summary_data) > 0:
                    summary_text = "\n".join([f"- {point}" for point in summary_data])

                episode_metadata = [
                    ("Type", episode_frontmatter.get("type")),
                    ("Category", episode_frontmatter.get("category")),
                    ("Published date", episode_frontmatter.get("published_date")),
                    ("URL", episode_frontmatter.get("episode_url")),
                    ("Category relevance", episode_frontmatter.get("category_relevance")),
                    ("Audience popularity", episode_frontmatter.get("audience_popularity_tier")),
                    ("Summary", summary_text),
                ]
                markdown_content.extend(_format_metadata_block(episode_metadata, heading_level=3))
                
                for paragraph in episode['content']:
                    markdown_content.append(f"{paragraph}\n\n")
            
                markdown_content.append("---\n\n")
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(markdown_content)
        
        # Get file size
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        word_count = sum(len(line.split()) for line in markdown_content)
        print(f"Generated {filepath} ({len(episode_batch)} episodes)...")
        if word_count > 200000:
            print(f"  ⚠ Warning: Exceeded word limit with {word_count:,} words")
        if file_size_mb > 200:
            print(f"  ⚠ Warning: Exceeded file size limit with {file_size_mb:.1f} MB")
        
        # Collect statistics for CSV
        if " - " in category:
            cat_name, subcat_name = category.split(" - ", 1)
        else:
            cat_name, subcat_name = category, ""
        
        episode_titles = [ep['title'] for ep in episode_batch]
        category_statistics.append({
            'Category': cat_name,
            'Subcategory': subcat_name,
            '#': len(episode_batch),
            'WC': word_count,
            'Size': round(file_size_mb, 2),
            'List': '; '.join(episode_titles)
        })
    
    # Print summary
    print("\n" + "="*70)
    print("Categorization Summary")
    print("="*70)
    for category in sorted(categorized_episodes.keys()):
        count = len(categorized_episodes[category])
        print(f"{category}: {count} episodes")
    print("="*70)
    
    if not args.use_cache:
        print(f"\n✓ Wrote episode match CSV to {OUTPUT_MATCHES_CSV}")
    
    # Write category statistics to CSV
    STATS_CSV = Path("output/category_statistics.csv")
    print(f"\nWriting category statistics to {STATS_CSV}...")
    with open(STATS_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'Category',
            'Subcategory',
            '#',
            'WC',
            'Size',
            'List'
        ])
        writer.writeheader()
        writer.writerows(category_statistics)
    print(f"✓ Wrote category statistics to {STATS_CSV}")

if __name__ == "__main__":
    main()
