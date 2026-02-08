#!/usr/bin/env python3
"""
Compile key topics from all transcript files into a markdown document.
"""

import json
import os
from pathlib import Path


def compile_key_topics(source_dirs=None, output_file="output/summary.md"):
    """
    Read all transcript JSON files and compile their titles and summary
    into a markdown file.
    
    Args:
        source_dirs: List of directories containing transcript JSON files
        output_file: Path to output markdown file
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
    
    # Compile data
    posts = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                title = data.get('title', 'Unknown Title')
                summary = data.get('summary', [])
                
                # Ensure summary is a list (handle case where it's a single string)
                if isinstance(summary, str):
                    summary = [summary] if summary else []
                
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
        f.write(f"Compiled from {len(posts)} posts across {len(source_dirs)} sources\n\n")
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
    
    print(f"Successfully compiled {len(posts)} posts to '{output_file}'")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compile key topics from transcript files"
    )
    parser.add_argument(
        '--source-dir',
        action='append',
        dest='source_dirs',
        help='Directory containing transcript JSON files (can be specified multiple times; default: output/rational_reminder and output/kitces)'
    )
    parser.add_argument(
        '--output',
        default='output/summary.md',
        help='Output markdown file (default: output/summary.md)'
    )
    
    args = parser.parse_args()
    
    compile_key_topics(args.source_dirs, args.output)
