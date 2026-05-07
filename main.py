#!/usr/bin/env python3
"""
Podcast Processing Pipeline
Orchestrates the complete workflow: scraping, metrics, and categorization.

Usage:
    python main.py                             # Run full pipeline with defaults
    python main.py --help                      # Show all options
    python main.py --force                     # Force re-scrape even if URLs are cached
    python main.py --scrape-retry-failed       # Retry failed episodes during scrape
    python main.py --finance-friday            # Include Kitces content in addition to Rational Reminder
    python main.py --min-percentile 50         # Use different percentile threshold
    python main.py --skip-categorize           # Skip categorization
    python main.py --skip-upload               # Skip upload to Google Docs
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Orchestrate podcast scraping, metrics, and categorization pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Scraping options
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-scrape even if URLs are already cached"
    )
    parser.add_argument(
        "--scrape-retry-failed",
        action="store_true",
        help="Only retry previously failed episodes (scraping step)"
    )
    parser.add_argument(
        "--scrape-url",
        type=str,
        help="Scrape a single specific episode URL"
    )
    
    # Metrics options
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Skip the metrics calculation step"
    )
    
    # Categorization options
    parser.add_argument(
        "--skip-categorize",
        action="store_true",
        help="Skip the categorization step"
    )
    parser.add_argument(
        "--finance-friday",
        action="store_true",
        help="Include Kitces content and enable Finance Friday mode"
    )
    parser.add_argument(
        "--min-percentile",
        type=float,
        default=-1,
        help="Minimum percentile threshold for including episodes (default: off)"
    )

    # Upload options
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip uploading updated sources to Google Docs"
    )
    parser.add_argument(
        "--upload-all-sources",
        action="store_true",
        help="Upload all markdown sources instead of only newly updated ones"
    )
    
    return parser.parse_args()


def find_most_recent_episode(source_dir):
    """Find the episode with the most recent publication date from filename."""
    source_path = Path(source_dir)
    if not source_path.exists():
        return None
    
    episodes = []
    for json_file in source_path.glob("*.json"):
        filename = json_file.stem
        if len(filename) >= 6 and filename[:6].isdigit():
            date_str = filename[:6]
            try:
                year = int(date_str[:2])
                month = int(date_str[2:4])
                day = int(date_str[4:6])
                year_full = 2000 + year if year < 70 else 1900 + year
                pub_date = datetime(year_full, month, day)
                
                # Get title from JSON
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    title = data.get('title', filename)
                except (json.JSONDecodeError, IOError):
                    title = filename
                
                episodes.append((pub_date, title))
            except ValueError:
                continue
    
    if not episodes:
        return None
    
    # Sort by date descending
    episodes.sort(key=lambda x: x[0], reverse=True)
    return episodes[0][1]


def run_command(description, command):
    """Run a command and report results."""
    print("\n" + "="*70)
    print(f"Step: {description}")
    print("="*70)
    print(f"Command: {' '.join(command)}\n")
    
    result = subprocess.run(command)
    
    if result.returncode != 0:
        print(f"\n✗ Failed at step: {description}")
        return False
    
    print(f"\n✓ Completed: {description}")
    return True


def main():
    args = parse_args()
    
    print("\n" + "="*70)
    print("PODCAST PROCESSING PIPELINE")
    print("="*70)
    
    steps_completed = []
    steps_skipped = []
    
    # Determine source directories
    source_dirs = ["output/rational_reminder"]
    
    # Step 1: Scraping
    scrape_steps = []
    if args.scrape_url:
        parsed = urlparse(args.scrape_url)
        host = (parsed.netloc or "").lower()
        if "kitces.com" in host:
            scrape_steps.append((
                "Kitces",
                ["python", "scrape_kitces.py", "--url", args.scrape_url]
            ))
        elif "rationalreminder.ca" in host:
            scrape_steps.append((
                "Rational Reminder",
                ["python", "scrape_rationalreminder.py", "--url", args.scrape_url]
            ))
        else:
            print("\n✗ Unknown scrape URL domain. Expected kitces.com or rationalreminder.ca")
            print("\nPipeline aborted.")
            return 1
    else:
        rr_cmd = ["python", "scrape_rationalreminder.py"]
        kitces_cmd = ["python", "scrape_kitces.py"]
        if args.scrape_retry_failed:
            rr_cmd.append("--retry-failed")
            kitces_cmd.append("--retry-failed")

        if args.finance_friday:
            scrape_steps = [
                ("Rational Reminder", rr_cmd),
                ("Kitces", kitces_cmd),
            ]
        else:
            scrape_steps = [("Rational Reminder", rr_cmd)]

    # Add --force flag to all scrape commands if specified
    if args.force:
        for i, (label, cmd) in enumerate(scrape_steps):
            scrape_steps[i] = (label, cmd + ["--force"])

    for source_label, scrape_cmd in scrape_steps:
        if run_command(f"Scraping {source_label} episodes", scrape_cmd):
            steps_completed.append(f"Scraping ({source_label})")
        else:
            print("\nPipeline aborted.")
            return 1
    
    # Step 2: Metrics
    needs_percentiles = args.min_percentile > 0
    if args.skip_metrics or not needs_percentiles:
        reason = "--skip-metrics" if args.skip_metrics else "min-percentile <= 0"
        print(f"\n⊘ Skipped: Metrics ({reason})")
        steps_skipped.append("Metrics")
    else:
        metrics_cmd = ["python", "fetch_youtube_metrics.py"]
        
        if run_command("Fetching YouTube metrics", metrics_cmd):
            steps_completed.append("Metrics")
        else:
            print("\nPipeline aborted.")
            return 1
    
    # Step 3: Categorization
    if not args.skip_categorize:
        categorize_cmd = ["python", "compile_sources.py"]
        if args.finance_friday:
            categorize_cmd.append("--finance-friday")
        if args.min_percentile >-1:
            categorize_cmd.extend(["--min-percentile", str(args.min_percentile)])
        
        if run_command("Categorizing episodes", categorize_cmd):
            steps_completed.append("Categorization")
        else:
            print("\nPipeline aborted.")
            return 1
    else:
        print("\n⊘ Skipped: Categorization (--skip-categorize)")
        steps_skipped.append("Categorization")

    # Step 4: Compile all summaries into a single markdown file for upload
    compile_summary_cmd = ["bash", "-c", "cat output/summaries/*.md > \"output/categorized/! Source Summary.md\""]
    if run_command("Compiling summary", compile_summary_cmd):
        steps_completed.append("Summary")
    else:
        print("\nPipeline aborted.")
        return 1

    # Step 5: Upload to Google Docs
    if args.finance_friday:
        copy_cmd = ["bash", "-c", "cp output/categorized/* ~/Finance-Friday"]
        if run_command("Copying categorized outputs to Finance Friday folder", copy_cmd):
            steps_completed.append("Finance Friday copy")
            steps_skipped.append("Upload")
        else:
            print("\nPipeline aborted.")
            return 1
    elif not args.skip_upload:
        upload_cmd = ["python", "upload_to_drive.py"]
        if args.upload_all_sources:
            upload_cmd.append("--all-sources")

        if run_command("Uploading updated sources to Google Docs", upload_cmd):
            steps_completed.append("Upload")
        else:
            print("\nPipeline aborted.")
            return 1
    else:
        print("\n⊘ Skipped: Upload (--skip-upload)")
        steps_skipped.append("Upload")
    
    # Summary
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    
    if steps_completed:
        print(f"\n✓ Completed steps ({len(steps_completed)}):")
        for step in steps_completed:
            print(f"  • {step}")
    
    if steps_skipped:
        print(f"\n⊘ Skipped steps ({len(steps_skipped)}):")
        for step in steps_skipped:
            print(f"  • {step}")
    
    print("\n✓ Pipeline execution complete!")
    print("="*70 + "\n")
    
    # Step 6: Generate NotebookLM prompts for the most recent episode
    if args.finance_friday:
        print("\n⊘ Skipped: NotebookLM prompts (--finance-friday)")
    else:
        latest_title = find_most_recent_episode(source_dirs[0])
        if latest_title:
            print("\n" + "="*70)
            print("NOTEBOOKLM PROMPTS FOR LATEST EPISODE")
            print("="*70)
            print(f"\nSummarize the discussion from \"{latest_title}\" into a concise executive summary with a neutral, high-density tone. DO NOT include information from other episodes. Use thematic groupings and highlight specific figures, percentages, and technical metrics.")
            print(f"\nConsult \"! Source Summary\" to identify which other episodes are most closely related to \"{latest_title}\". Then, using those transcripts, list three related episodes and provide a one sentence description on the specific connection to \"{latest_title}\". You MUST use information from other episodes to determine which are best suited for further exploration on this podcast discussion.")
            print("\n")
        else:
            print("\nNo episodes found to generate prompts.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
