#!/usr/bin/env python3
"""
Compile key topics from all transcript files into a markdown document.
"""

import csv
import json
import os
from pathlib import Path


def load_youtube_metrics(metrics_file="output/youtube_metrics.csv"):
    """
    Load YouTube metrics from CSV and create a mapping of titles to percentiles.
    
    Args:
        metrics_file: Path to YouTube metrics CSV file
        
    Returns:
        Dictionary mapping video titles to percentile scores
    """
    metrics_by_title = {}
    
    if not os.path.exists(metrics_file):
        print(f"Warning: YouTube metrics file '{metrics_file}' not found")
        return metrics_by_title
    
    try:
        with open(metrics_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                title = row.get('title', '').strip()
                percentile_str = row.get('percentile', '')
                
                if title and percentile_str:
                    try:
                        percentile = float(percentile_str)
                        metrics_by_title[title] = percentile
                    except ValueError:
                        # Skip rows where percentile is not a valid number
                        pass
        
        print(f"Loaded YouTube metrics for {len(metrics_by_title)} episodes")
    except Exception as e:
        print(f"Error reading metrics file '{metrics_file}': {e}")
    
    return metrics_by_title


def compile_key_topics(source_dirs=None, output_file="output/summaries/Full Summary.md", min_percentile=None, metrics_file="output/youtube_metrics.csv"):
    """
    Read all transcript JSON files and compile their titles and summary
    into a markdown file, optionally filtering by YouTube percentile threshold.
    
    Args:
        source_dirs: List of directories containing transcript JSON files
        output_file: Path to output markdown file
        min_percentile: Minimum percentile threshold for YouTube videos (0-100).
                       Posts without YouTube data are always included.
        metrics_file: Path to YouTube metrics CSV file
    """
    if source_dirs is None:
        source_dirs = ["output/rational_reminder", "output/kitces"]
    
    # Ensure source_dirs is a list
    if isinstance(source_dirs, str):
        source_dirs = [source_dirs]
    
    # Collect all JSON files from all source directories
    json_files = []
    for source_dir in source_dirs:
        source_path = Path(source_dir)
        
        if not source_path.exists():
            print(f"Warning: Source directory '{source_dir}' not found, skipping")
            continue
        
        # Get all JSON files from this directory
        dir_json_files = sorted(source_path.glob("*.json"))
        
        if not dir_json_files:
            print(f"No JSON files found in '{source_dir}'")
        else:
            print(f"Found {len(dir_json_files)} transcript files in '{source_dir}'")
            json_files.extend(dir_json_files)
    
    if not json_files:
        print("No JSON files found in any source directories")
        return
    
    print(f"\nTotal: {len(json_files)} transcript files from {len(source_dirs)} directories")
    
    # Load YouTube metrics if percentile filtering is enabled
    metrics_by_title = {}
    if min_percentile is not None and min_percentile > 0:
        metrics_by_title = load_youtube_metrics(metrics_file)
        print(f"Using minimum percentile threshold: {min_percentile}")
    
    # Compile data
    posts = []
    posts_filtered_out = 0
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                title = data.get('title', 'Unknown Title')
                summary = data.get('summary', [])
                
                # Ensure summary is a list (handle case where it's a single string)
                if isinstance(summary, str):
                    summary = [summary] if summary else []
                
                # Check percentile threshold if filtering is enabled
                if min_percentile is not None and min_percentile > 0:
                    if title in metrics_by_title:
                        # Has YouTube data: check if it meets the threshold
                        percentile = metrics_by_title[title]
                        if percentile < min_percentile:
                            posts_filtered_out += 1
                            continue
                    # else: No YouTube data - include by default
                
                posts.append({
                    'title': title,
                    'summary': summary,
                    'filename': json_file.name
                })
        except Exception as e:
            print(f"Error reading {json_file.name}: {e}")
            continue
    
    # Write to markdown file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Summary from Sources\n\n")
        f.write(f"Compiled pieces of {len(posts)} content across {len(source_dirs)} sources\n\n")
        f.write("---\n\n")
        
        for episode in posts:
            f.write(f"## {episode['title']}\n\n")
            
            if episode['summary']:
                f.write("### Summary\n\n")
                for point in episode['summary']:
                    f.write(f"- {point}\n")
                f.write("\n")
            else:
                f.write("*No summary available*\n\n")
            
            f.write("---\n\n")
    
    # Report results
    print(f"Successfully compiled {len(posts)} pieces of content to '{output_file}'")
    if min_percentile is not None and min_percentile > 0:
        print(f"  (Filtered out {posts_filtered_out} pieces of content below {min_percentile}th percentile)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compile key topics from source files"
    )
    parser.add_argument(
        '--source-dir',
        action='append',
        dest='source_dirs',
        help='Directory containing source JSON files (can be specified multiple times; default: output/rational_reminder and output/kitces)'
    )
    parser.add_argument(
        '--output',
        default='output/summaries/Full Summary.md',
        help='Output markdown file (default: output/summaries/Full Summary.md)'
    )
    parser.add_argument(
        '--min-percentile',
        type=float,
        default=-1,
        help='Minimum YouTube percentile threshold (0-100 scale). Sources without YouTube metadata are always included (default: off)'
    )
    parser.add_argument(
        '--metrics-file',
        default='output/youtube_metrics.csv',
        help='Path to YouTube metrics CSV file (default: output/youtube_metrics.csv)'
    )
    
    args = parser.parse_args()
    
    compile_key_topics(args.source_dirs, args.output, args.min_percentile, args.metrics_file)
