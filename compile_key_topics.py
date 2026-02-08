#!/usr/bin/env python3
"""
Compile key topics from all transcript files into a markdown document.
"""

import json
import os
from pathlib import Path


def compile_key_topics(transcripts_dir="transcripts", output_file="key_topics.md"):
    """
    Read all transcript JSON files and compile their titles and key points
    into a markdown file.
    
    Args:
        transcripts_dir: Directory containing transcript JSON files
        output_file: Path to output markdown file
    """
    transcripts_path = Path(transcripts_dir)
    
    if not transcripts_path.exists():
        print(f"Error: Transcripts directory '{transcripts_dir}' not found")
        return
    
    # Get all JSON files
    json_files = sorted(transcripts_path.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in '{transcripts_dir}'")
        return
    
    print(f"Found {len(json_files)} transcript files")
    
    # Compile data
    episodes = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                title = data.get('title', 'Unknown Title')
                key_points = data.get('key_points', [])
                
                episodes.append({
                    'title': title,
                    'key_points': key_points,
                    'filename': json_file.name
                })
        except Exception as e:
            print(f"Error reading {json_file.name}: {e}")
            continue
    
    # Write to markdown file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Key Topics from Rational Reminder Podcast Transcripts\n\n")
        f.write(f"Compiled from {len(episodes)} episodes\n\n")
        f.write("---\n\n")
        
        for episode in episodes:
            f.write(f"## {episode['title']}\n\n")
            
            if episode['key_points']:
                f.write("### Key Points\n\n")
                for point in episode['key_points']:
                    f.write(f"- {point}\n")
                f.write("\n")
            else:
                f.write("*No key points available*\n\n")
            
            f.write("---\n\n")
    
    print(f"Successfully compiled {len(episodes)} episodes to '{output_file}'")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Compile key topics from transcript files"
    )
    parser.add_argument(
        '--transcripts-dir',
        default='transcripts',
        help='Directory containing transcript JSON files (default: transcripts)'
    )
    parser.add_argument(
        '--output',
        default='key_topics.md',
        help='Output markdown file (default: key_topics.md)'
    )
    
    args = parser.parse_args()
    
    compile_key_topics(args.transcripts_dir, args.output)
