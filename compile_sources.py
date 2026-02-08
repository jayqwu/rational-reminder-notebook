#!/usr/bin/env python3
"""Compile sources into category-specific markdown files using semantic similarity"""

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

TRANSCRIPTS_DIR = "output/rational_reminder"
METRICS_FILE = "output/youtube_metrics.csv"
TAXONOMY_PATH = Path("taxonomy.json")
OUTPUT_MATCHES_CSV = Path("episode_category_matches.csv")

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

def compute_embeddings(model, texts, show_progress=False):
    """Compute embeddings for a list of texts.
    
    Args:
        model: SentenceTransformer model
        texts: List of text strings to encode
        show_progress: Whether to show progress bar
    
    Returns:
        Tensor of embeddings
    """
    return model.encode(texts, convert_to_tensor=True, show_progress_bar=show_progress)

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

def create_episode_text(title, summary, transcript=None, max_transcript_chars=5000):
    """Create combined text representation of an episode for semantic matching.
    
    Args:
        title: Episode title
        summary: List of key points from the episode (can be None or empty)
        transcript: Optional full transcript (will be truncated if too long)
        max_transcript_chars: Maximum characters to include from transcript
    
    Returns:
        Combined text string
    """
    parts = [title] if title else []
    
    # Add key points if available
    if summary:
        if isinstance(summary, list):
            # Filter out empty or None values
            valid_points = [str(point) for point in summary if point]
            parts.extend(valid_points)
        else:
            point_str = str(summary).strip()
            if point_str:
                parts.append(point_str)
    
    # Optionally add truncated transcript for more context
    if transcript:
        if isinstance(transcript, list):
            transcript_text = " ".join(str(p) for p in transcript if p)
        else:
            transcript_text = str(transcript)
        
        if transcript_text.strip():
            if len(transcript_text) > max_transcript_chars:
                transcript_text = transcript_text[:max_transcript_chars]
            parts.append(transcript_text)
    
    return " ".join(parts) if parts else title or ""

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
    transcript = episode_data.get('transcript', [])
    
    episode_text = create_episode_text(title, summary, transcript)
    
    # Encode episode text
    episode_embedding = model.encode(episode_text, convert_to_tensor=True)
    
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

def generate_summary(category, description, episodes):
    """Generate a structured summary for a category.
    
    Args:
        category: Category label (e.g., "Investing - Market Efficiency")
        description: Category description from taxonomy
        episodes: List of episodes in this category
    
    Returns:
        Formatted summary string
    """
    # Parse category and subcategory
    if " - " in category:
        main_cat, sub_cat = category.split(" - ", 1)
    else:
        main_cat = category
        sub_cat = None
    
    # High-level overview
    overview = f"This collection focuses on **{category}**"
    if sub_cat:
        overview += f", specifically covering {sub_cat.lower()}"
    # Only include episode count if more than one episode
    if len(episodes) > 1:
        overview += f". It contains {len(episodes)} podcast episodes from the Rational Reminder Podcast."
    
    return overview

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
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load taxonomy
    print("Loading taxonomy...")
    taxonomy = load_taxonomy(TAXONOMY_PATH)
    print(f"✓ Loaded {len(taxonomy)} categories\n")
    
    # Initialize semantic similarity model
    print("Initializing semantic similarity model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Model loaded\n")
    
    # Pre-compute category embeddings
    print("Computing category embeddings...")
    category_descriptions = [entry['description'] for entry in taxonomy]
    category_embeddings = compute_embeddings(model, category_descriptions, show_progress=True)
    print(f"✓ Computed embeddings for {len(taxonomy)} categories\n")
    
    # Create category label to description mapping
    category_to_description = {
        f"{entry['category']} - {entry['subcategory']}": entry['description']
        for entry in taxonomy
    }
    
    # Load percentile mapping
    print(f"Loading popularity metrics from {args.metrics_file}...")
    percentile_map = load_percentile_map(args.metrics_file)
    print(f"✓ Loaded percentiles for {len(percentile_map)} videos\n")
    
    # Collect all transcript files
    episode_files = sorted(Path(TRANSCRIPTS_DIR).glob("*.json"))
    
    print(f"Found {len(episode_files)} episode files")
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
        
        # Check percentile filter
        youtube_data = data.get('youtube', {})
        video_id = youtube_data.get('video_id')
        published_at_raw = data.get('pub_date')
        published_date = _format_published_date(published_at_raw)
        youtube_url = youtube_data.get('url')
        if not youtube_url and video_id:
            youtube_url = f"https://www.youtube.com/watch?v={video_id}"
        
        if video_id and video_id in percentile_map:
            percentile = percentile_map[video_id]
            if percentile < args.min_percentile:
                episodes_skipped += 1
                continue
        elif args.min_percentile > -1.0:
            # No percentile data and filtering is enabled, skip
            episodes_skipped += 1
            continue
        else:
            percentile = None
        
        episodes_filtered += 1
        
        title = data.get("title", episode_file.stem)
        transcript = _normalize_transcript_paragraphs(data.get("transcript", []))
        episode_url = data.get("episode_url")
        summary = data.get("summary", [])

        # Assign category using semantic similarity
        episode_data = {
            'title': title,
            'summary': summary,
            'transcript': transcript
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
            'transcript': transcript,
            'episode_url': episode_url,
            'percentile': percentile if percentile is not None else -1.0,
            'youtube_url': youtube_url
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
    print(f"  Episodes included: {episodes_filtered}")
    print(f"  Episodes skipped (percentile < {args.min_percentile}): {episodes_skipped}\n")
    
    # Generate markdown files for each category
    OUTPUT_DIR = Path("categorized_transcripts")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    for category, episodes in categorized_episodes.items():
        # Sort episodes by percentile (highest first)
        episodes.sort(key=lambda ep: ep.get('percentile', -1.0), reverse=True)
        
        # Create human-readable filename from category
        base_filename = category.replace("/", "-").replace(":", " -")
        filepath = OUTPUT_DIR / f"{base_filename}.md"
        episode_batch = episodes
        
        print(f"Generating {filepath} ({len(episode_batch)} episodes)...")
        
        markdown_content = []
        
        # Get description for this category
        description = category_to_description.get(category, "")
        
        # Add header first
        markdown_content.append(f"# {category}\n\n")
            
        # Add description as subheading
        if description:
            markdown_content.append(f"## Description\n{description}\n\n")
        
        # Add structured summary
        summary = generate_summary(category, description, episode_batch)
        markdown_content.append(f"## Summary\n{summary}\n\n")
        
        # Create metadata block for category
        category_metadata = [
            ("Type", "category"),
            ("Episodes in file", len(episode_batch)),
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
                    episode_frontmatter['audience_popularity'] = percentile_display
                    episode_frontmatter['audience_popularity_tier'] = popularity_to_tier(percentile_display)
                
                # Add summary (only if non-empty) and category relevance
                summary_data = episode.get('summary')
                if summary_data and (isinstance(summary_data, list) and len(summary_data) > 0):
                    episode_frontmatter['summary'] = summary_data
                if episode.get('category_relevance'):
                    episode_frontmatter['category_relevance'] = episode['category_relevance']
                
                # Write metadata block after heading
                episode_metadata = [
                    ("Type", episode_frontmatter.get("type")),
                    ("Category", episode_frontmatter.get("category")),
                    ("Published date", episode_frontmatter.get("published_date")),
                    ("Episode URL", episode_frontmatter.get("episode_url")),
                    ("YouTube URL", episode_frontmatter.get("youtube_url")),
                    ("Audience Popularity", episode_frontmatter.get("audience_popularity")),
                    ("Audience Popularity Tier", episode_frontmatter.get("audience_popularity_tier")),
                    ("Key points", episode_frontmatter.get("summary")),
                    ("Category relevance", episode_frontmatter.get("category_relevance"))
                ]
                markdown_content.extend(_format_metadata_block(episode_metadata, heading_level=3))
                
                for paragraph in episode['transcript']:
                    markdown_content.append(f"{paragraph}\n\n")
            
                markdown_content.append("---\n\n")
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(markdown_content)
        
        # Get file size
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  ✓ {filepath} ({file_size_mb:.2f} MB)\n")
    
    # Print summary
    print("\n" + "="*70)
    print("Categorization Summary")
    print("="*70)
    for category in sorted(categorized_episodes.keys()):
        count = len(categorized_episodes[category])
        print(f"{category}: {count} episodes")
    print("="*70)

    with open(OUTPUT_MATCHES_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "category", "subcategory"])
        writer.writeheader()
        writer.writerows(episode_matches)

    print(f"\n✓ Wrote episode match CSV to {OUTPUT_MATCHES_CSV}")

if __name__ == "__main__":
    main()
