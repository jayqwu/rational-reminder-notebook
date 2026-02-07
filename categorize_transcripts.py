#!/usr/bin/env python3
"""Categorize podcast transcripts and generate category-specific markdown files"""

import argparse
import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import yaml

TRANSCRIPTS_DIR = "transcripts"
METRICS_FILE = "youtube_metrics.csv"
TAXONOMY_PATH = Path("taxonomy.json")

UNCATEGORIZED_LABEL = "Uncategorized"
MIN_SCORE = 3
SECONDARY_MIN_SCORE = 4
MAX_LABELS = 2
SOFT_CAP_RATIO = 0.15

def load_categories(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError("Taxonomy must be a non-empty JSON object.")
    return data

def _build_subcategory_map(taxonomy):
    subcategories = {}
    for category, keywords in taxonomy.items():
        if isinstance(keywords, dict):
            for subcategory, sub_keywords in keywords.items():
                if not isinstance(sub_keywords, list):
                    raise ValueError(
                        f"Subcategory '{subcategory}' in '{category}' must be a list."
                    )
                label = f"{category} - {subcategory}"
                subcategories[label] = sub_keywords
        elif isinstance(keywords, list):
            subcategories[category] = keywords
        else:
            raise ValueError(f"Category '{category}' must be a list or object.")
    return subcategories

RAW_TAXONOMY = load_categories(TAXONOMY_PATH)
SUBCATEGORIES = _build_subcategory_map(RAW_TAXONOMY)

def _normalize_transcript_text(transcript_text):
    if isinstance(transcript_text, list):
        return " ".join(transcript_text)
    if transcript_text is None:
        return ""
    return str(transcript_text)

def _normalize_transcript_paragraphs(transcript_text):
    if isinstance(transcript_text, list):
        return [str(paragraph) for paragraph in transcript_text]
    if transcript_text is None:
        return []
    return [str(transcript_text)]

def score_transcript(title, transcript_text):
    """Score a transcript against each category based on keyword matches."""

    full_text = (title + " " + _normalize_transcript_text(transcript_text)).lower()
    scores = {subcategory: 0 for subcategory in SUBCATEGORIES}

    for subcategory, keywords in SUBCATEGORIES.items():
        for keyword in keywords:
            scores[subcategory] += full_text.count(keyword.lower())

    return scores

def assign_labels(scores, category_counts, soft_cap):
    candidates = [
        (category, score) for category, score in scores.items() if score >= MIN_SCORE
    ]

    if not candidates:
        return [UNCATEGORIZED_LABEL]

    candidates.sort(
        key=lambda item: (
            item[1] - (category_counts[item[0]] / max(soft_cap, 1)),
            item[1],
        ),
        reverse=True,
    )

    primary = None
    for category, score in candidates:
        if category_counts[category] < soft_cap:
            primary = (category, score)
            break

    if primary is None:
        primary = candidates[0]

    labels = [primary[0]]
    category_counts[primary[0]] += 1

    if MAX_LABELS > 1:
        for category, score in candidates:
            if category == primary[0]:
                continue
            if len(labels) >= MAX_LABELS:
                break
            if score < SECONDARY_MIN_SCORE:
                continue
            if category_counts[category] < soft_cap:
                labels.append(category)
                category_counts[category] += 1

    return labels

def generate_summary(category, keywords, episodes):
    """Generate a structured summary for a category."""
    
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
    
    # Topic areas from keywords
    topic_areas = []
    if keywords:
        # Group keywords into logical clusters (simple approach: first 5-7 keywords as main topics)
        main_topics = keywords[:min(7, len(keywords))]
        if len(main_topics) > 0:
            topic_areas.append("\n**Key topics covered:**")
            topic_areas.append("- " + ", ".join(main_topics))
    
    # Combine into structured summary
    summary_parts = [overview]
    if topic_areas:
        summary_parts.extend(topic_areas)
    
    return "\n".join(summary_parts)

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
    
    # Load percentile mapping
    print(f"Loading popularity metrics from {args.metrics_file}...")
    percentile_map = load_percentile_map(args.metrics_file)
    print(f"✓ Loaded percentiles for {len(percentile_map)} videos\n")
    # Collect all transcript files
    episode_files = sorted(Path(TRANSCRIPTS_DIR).glob("*.json"))
    
    print(f"Found {len(episode_files)} episode files")
    print(f"Categorizing episodes...\n")

    category_counts = defaultdict(int)
    num_subcategories = len(SUBCATEGORIES)
    target_size = math.ceil(len(episode_files) / num_subcategories)
    soft_cap = math.ceil(target_size * (1 + SOFT_CAP_RATIO))
    
    # Dictionary to store episodes by category
    categorized_episodes = defaultdict(list)
    
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

        scores = score_transcript(title, transcript)
        labels = assign_labels(scores, category_counts, soft_cap)
        episode_record = {
            'title': title,
            'transcript': transcript,
            'episode_url': episode_url,
            'percentile': percentile if percentile is not None else -1.0,
            'published_date': published_date,
            'youtube_url': youtube_url
        }

        for label in labels:
            categorized_episodes[label].append(episode_record)
    
    print(f"\n✓ Categorization complete")
    print(f"  Episodes included: {episodes_filtered}")
    print(f"  Episodes skipped (percentile < {args.min_percentile}): {episodes_skipped}\n")
    
    # Generate markdown files for each category
    EPISODES_PER_FILE = 1000
    OUTPUT_DIR = Path("categorized_transcripts")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    for category, episodes in categorized_episodes.items():
        # Sort episodes by percentile (highest first)
        episodes.sort(key=lambda ep: ep.get('percentile', -1.0), reverse=True)
        
        # Create human-readable filename from category
        base_filename = category.replace("/", "-").replace(":", " -")
        
        # Split episodes into chunks of 1000
        num_files = (len(episodes) + EPISODES_PER_FILE - 1) // EPISODES_PER_FILE
        
        for file_index in range(num_files):
            start_idx = file_index * EPISODES_PER_FILE
            end_idx = min(start_idx + EPISODES_PER_FILE, len(episodes))
            episode_batch = episodes[start_idx:end_idx]
            
            # Create filename with part number if multiple files needed
            if num_files > 1:
                filepath = OUTPUT_DIR / f"{base_filename}_part{file_index + 1}.md"
                part_info = f" (Part {file_index + 1} of {num_files})"
            else:
                filepath = OUTPUT_DIR / f"{base_filename}.md"
                part_info = ""
            
            print(f"Generating {filepath} ({len(episode_batch)} episodes)...")
            
            markdown_content = []
            
            # Get keywords for this category
            keywords = SUBCATEGORIES.get(category, [])
            
            # Add header first
            markdown_content.append(f"# {category}{part_info}\n\n")
            
            # Add structured summary
            if category in SUBCATEGORIES:
                summary = generate_summary(category, keywords, episode_batch)
                markdown_content.append(f"## Summary\n{summary}\n\n")
            
            # Create YAML frontmatter for category
            frontmatter = {
                'type': 'category',
                'episodes_in_file': len(episode_batch),
                'keywords': keywords,
                'source': 'Rational Reminder Podcast',
                'source_url': 'https://rationalreminder.ca/podcast/'
            }
            
            if num_files > 1:
                frontmatter['total_episodes'] = len(episodes)
                frontmatter['part'] = file_index + 1
                frontmatter['total_parts'] = num_files
            
            if args.min_percentile > -1.0:
                frontmatter['filter_percentile'] = args.min_percentile
            
            # Write YAML frontmatter after summary
            markdown_content.append("---\n")
            markdown_content.append(yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True))
            markdown_content.append("---\n\n")
            
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
                    episode_frontmatter['percentile'] = percentile_display
                
                # Add keywords
                if keywords:
                    episode_frontmatter['keywords'] = keywords
                
                # Write episode frontmatter after heading
                markdown_content.append("---\n")
                markdown_content.append(yaml.dump(episode_frontmatter, default_flow_style=False, allow_unicode=True))
                markdown_content.append("---\n\n")
                
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

if __name__ == "__main__":
    main()
