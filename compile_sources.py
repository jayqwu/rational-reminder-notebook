#!/usr/bin/env python3
"""Compile sources into category-specific markdown files using semantic similarity"""

import argparse
import csv
import json
import os
from dotenv import load_dotenv
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
OUTPUT_SIMILARITIES_CSV = Path("output/episode_similarity_matrix.csv")
CACHE_FILE = Path("output/episodes_cache.json")

# Similarity thresholds for relevance levels (these can be tuned based on observed score distribution)
SIMILARITY_VERY_HIGH = 0.60
SIMILARITY_HIGH = 0.50
SIMILARITY_AVERAGE = 0.40
SIMILARITY_LOW = 0.30

# Exclude episodes whose titles contain any of these strings (case-insensitive)
EXCLUDE_TITLES = [
    "Year in Review",
    "Scott Galloway",
    "Comprehensive Overview",
]

# GPU/embedding controls tuned for laptop GPUs
USE_FP16 = True
BATCH_SIZE = 8
# MODEL='Octen/Octen-Embedding-8B'
MODEL='Octen/Octen-Embedding-4B'
# MODEL='Octen/Octen-Embedding-0.6B'
# MODEL='nomic-ai/nomic-embed-text-v1.5'

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

def assign_category_by_similarity(episode_data, taxonomy, category_embeddings, model, return_similarities=False):
    """Assign a category to an episode based on semantic similarity.
    
    Args:
        episode_data: Dictionary containing episode information
        taxonomy: List of taxonomy entries
        category_embeddings: Pre-computed embeddings for category descriptions
        model: SentenceTransformer model
    
    Returns:
        Tuple of (category_label, similarity_score, description, related_categories)
        If return_similarities is True, also returns a list of similarity scores
        aligned to the taxonomy order.
    """
    # Create episode text representation
    title = episode_data.get('title', '')
    summary = episode_data.get('summary', [])

    episode_text = create_episode_text(title, summary).strip()
    if not episode_text:
        episode_text = title or ""

    # Encode episode text to a single embedding
    episode_embedding = model.encode(episode_text, convert_to_tensor=True, batch_size=BATCH_SIZE)
    
    # Compute cosine similarities
    similarities = util.cos_sim(episode_embedding, category_embeddings)[0]
    
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
    
    if return_similarities:
        similarity_scores = [score.item() for score in similarities]
        return category_label, best_score, description, related_categories, similarity_scores

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
        default=50,
        help="Minimum percentile threshold for including episodes (default: 50)"
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
        model = SentenceTransformer(MODEL, trust_remote_code=True, device=device)
        if device == "cuda" and USE_FP16:
            model = model.to(torch.float16)
        print(f"✓ Model loaded: {MODEL}\n")
        
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
        
        # Pre-filter episodes by percentile threshold and title exclusions
        print(f"Filtering episodes by percentile threshold ({args.min_percentile}) and title exclusions...")
        filtered_episode_files = []
        episodes_skipped_percentile = 0
        episodes_excluded_by_title = 0
        
        for episode_file in tqdm(episode_files, desc="Filtering"):
            with open(episode_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check percentile filter (if YouTube data is available)
            youtube_data = data.get('youtube', {})
            video_id = youtube_data.get('video_id') if youtube_data else None
            
            # Handle percentile filtering only if percentile data exists
            if video_id and video_id in percentile_map:
                percentile = percentile_map[video_id]
                if percentile < args.min_percentile:
                    episodes_skipped_percentile += 1
                    continue
            # else: No percentile data - include by default
            
            # Check title exclusion filter
            title = data.get("title", episode_file.stem)
            if EXCLUDE_TITLES:
                title_lower = title.lower()
                if any(exclude.lower() in title_lower for exclude in EXCLUDE_TITLES):
                    episodes_excluded_by_title += 1
                    continue
            
            filtered_episode_files.append(episode_file)
        
        print(f"✓ Episodes after filtering: {len(filtered_episode_files)}")
        if episodes_skipped_percentile > 0:
            print(f"  Skipped (percentile < {args.min_percentile}): {episodes_skipped_percentile}")
        if episodes_excluded_by_title > 0:
            print(f"  Excluded by title filter: {episodes_excluded_by_title}")
        print(f"Categorizing episodes using semantic similarity...\n")
        
        # Dictionary to store episodes by category
        categorized_episodes = defaultdict(list)
        episode_matches = []
        similarity_rows = []
        all_similarity_scores = []  # Track all similarity scores for distribution analysis
        
        for episode_file in tqdm(filtered_episode_files):
            with open(episode_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract episode data
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
            
            # Get percentile (already filtered, so just retrieve for metadata)
            percentile = None
            if video_id and video_id in percentile_map:
                percentile = percentile_map[video_id]
            
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
            category_label, similarity_score, description, related_categories, similarity_scores = (
                assign_category_by_similarity(
                    episode_data,
                    taxonomy,
                    category_embeddings,
                    model,
                    return_similarities=True,
                )
            )
            
            # Convert similarity score to descriptive relevance level
            relevance_level = similarity_to_relevance(similarity_score)
            
            # Track all similarity scores for distribution analysis (not just best match)
            all_similarity_scores.extend(similarity_scores)
            
            episode_record = {
                'title': title,
                'published_date': published_date,
                'summary': summary,
                'category_relevance': relevance_level,
                'related_categories': related_categories,
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

            similarity_row = {
                "episode_title": title,
            }
            for entry, score in zip(taxonomy, similarity_scores):
                label = f"{entry['category']} - {entry['subcategory']}"
                similarity_row[label] = score
            similarity_rows.append(similarity_row)
        
        print(f"\n✓ Categorization complete")
        print(f"  episodes categorized: {len(filtered_episode_files)}")
        
        # Calculate and display similarity score distribution
        if all_similarity_scores:
            sorted_scores = sorted(all_similarity_scores)
            n = len(sorted_scores)
            
            def percentile(data, p):
                """Calculate percentile from sorted data"""
                k = (n - 1) * p / 100
                f = int(k)
                c = k - f
                if f + 1 < n:
                    return data[f] + c * (data[f + 1] - data[f])
                return data[f]
            
            p20 = percentile(sorted_scores, 20)
            p40 = percentile(sorted_scores, 40)
            p60 = percentile(sorted_scores, 60)
            p80 = percentile(sorted_scores, 80)
            
            print("\nSimilarity Score Distribution:")
            print(f"  20th percentile: {p20:.4f}")
            print(f"  40th percentile: {p40:.4f}")
            print(f"  60th percentile: {p60:.4f}")
            print(f"  80th percentile: {p80:.4f}")
            print(f"  Mean: {sum(all_similarity_scores) / len(all_similarity_scores):.4f}")
            print(f"  Min: {min(all_similarity_scores):.4f}")
            print(f"  Max: {max(all_similarity_scores):.4f}")
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

        print(f"\nWriting episode similarity matrix to {OUTPUT_SIMILARITIES_CSV}...")
        similarity_fieldnames = [
            "episode_title",
            *[f"{entry['category']} - {entry['subcategory']}" for entry in taxonomy],
        ]
        with open(OUTPUT_SIMILARITIES_CSV, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=similarity_fieldnames)
            writer.writeheader()
            writer.writerows(similarity_rows)
        print(f"✓ Wrote episode similarity matrix to {OUTPUT_SIMILARITIES_CSV}")
    
    # Generate markdown files for each category
    OUTPUT_DIR = Path("output/categorized")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Track statistics for CSV output
    category_statistics = []
    
    for category, episodes in categorized_episodes.items():
        # Sort episodes by similarity score (highest first)
        episodes.sort(key=lambda ep: ep.get('category_relevance', ''), reverse=True)
        
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
        markdown_content.append(f"## Topic Description\n\n{description}\n\n")
    
        # Add metadata block for the category
        markdown_content.append(f"Pieces of Content: {len(episode_batch)}\n\n")
        markdown_content.append(f"Source: [Rational Reminder Podcast](https://rationalreminder.ca/podcast/) and [Kitces](https://www.kitces.com/)\n\n")
        
        # Add episodes
        for episode in episode_batch:
            # Episode title heading
            markdown_content.append(f"## {episode['title']}\n\n")
            
            # Build episode metadata
            markdown_content.append(f"Published date: {episode.get('published_date', 'N/A')}\n\n")
            markdown_content.append(f"Episode URL: {episode.get('url', 'N/A')}\n\n")
            
            if episode.get('popularity_percentile', -1) >= 0:
                markdown_content.append(f"Audience Popularity: popularity_to_tier({episode['popularity_percentile']})\n\n")

            markdown_content.append(f"Category relevance: {episode.get('category_relevance', 'N/A')}\n\n")
            if episode.get('related_categories'):
                markdown_content.append("Related categories:\n\n")
                for related_cat in episode['related_categories']:
                    markdown_content.append(f"- {related_cat}\n")
                markdown_content.append("\n")
                        
            if episode.get('summary'):
                markdown_content.append("### Summary\n\n")
                for item in episode['summary']:
                    markdown_content.append(f"- {item}\n")
                markdown_content.append("\n")

            markdown_content.append("### Content\n\n")
            for paragraph in episode['content']:
                markdown_content.append(f"{paragraph}\n\n")
        
            markdown_content.append("---\n\n")
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(markdown_content)
        
        # Get file size
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        word_count = sum(len(line.split()) for line in markdown_content)
        # print(f"Generated {filepath} ({len(episode_batch)} episodes)...")
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
    print("\n" + "="*100)
    print("Categorization Summary")
    print("="*100)
    
    # Sort by word count (descending)
    sorted_stats = sorted(category_statistics, key=lambda x: x['WC'], reverse=True)
    
    # Print header
    print(f"{'Category':<78} {'Episodes':>10} {'Words':>10}")
    print("-" * 100)
    
    # Print each category with consistent column widths
    for stat in sorted_stats:
        category_label = f"{stat['Category']} - {stat['Subcategory']}"
        episode_count = stat['#']
        word_count = stat['WC']
        print(f"{category_label:<78} {episode_count:>10} {word_count:>10,}")
    print("="*100)
    
    if not args.use_cache:
        print(f"\n✓ Wrote episode match CSV to {OUTPUT_MATCHES_CSV}")
    elif args.use_cache:
        print(
            "\nNote: Similarity matrix CSV is only generated when categorizing; "
            "run without --use-cache to regenerate it."
        )
    
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
    load_dotenv()
    main()
