#!/usr/bin/env python3
"""
Podcast Processing Pipeline
Orchestrates the complete workflow: scraping, metrics, and categorization.

Usage:
    python main.py                             # Run full pipeline with defaults
    python main.py --help                      # Show all options
    python main.py --force                     # Force re-scrape even if URLs are cached
    python main.py --scrape-retry-failed       # Retry failed episodes during scrape
    python main.py --min-percentile 50         # Use different percentile threshold
    python main.py --skip-categorize           # Skip categorization
"""

import argparse
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse


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
        "--min-percentile",
        type=float,
        default=-1,
        help="Minimum percentile threshold for including episodes (default: off)"
    )
    
    return parser.parse_args()


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
        scrape_steps = [
            ("Rational Reminder", rr_cmd),
            ("Kitces", kitces_cmd),
        ]

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
        if args.min_percentile != -1:
            categorize_cmd.extend(["--min-percentile", str(args.min_percentile)])
        
        if run_command("Categorizing episodes", categorize_cmd):
            steps_completed.append("Categorization")
        else:
            print("\nPipeline aborted.")
            return 1
    else:
        print("\n⊘ Skipped: Categorization (--skip-categorize)")
        steps_skipped.append("Categorization")

    # Step 4: Summary
    summary_cmd = ["python", "compile_summary.py"]
    if args.min_percentile != -1:
        summary_cmd.extend(["--min-percentile", str(args.min_percentile)])

    if run_command("Compiling summary", summary_cmd):
        steps_completed.append("Summary")
    else:
        print("\nPipeline aborted.")
        return 1
    
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
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
