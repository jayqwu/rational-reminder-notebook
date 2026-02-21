#!/usr/bin/env python3
"""Upload compiled markdown files to Google Docs with formatting"""

import argparse
import os
import json
import base64
import pickle
from pathlib import Path
from dotenv import load_dotenv, set_key
from tqdm import tqdm
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Google Drive API scopes
SCOPES = ['https://www.googleapis.com/auth/drive']

# Paths
CREDENTIALS_FILE = Path("credentials.json")
ENV_FILE = Path(".env")
MARKDOWN_DIR = Path("output/categorized")
CACHE_FILE = Path("output/cache.json")

def authenticate_google_drive():
    """Authenticate with Google Drive API using OAuth2."""
    load_dotenv()
    creds = None
    
    # Load existing token from .env if available
    token_str = os.getenv('GOOGLE_DRIVE_TOKEN')
    if token_str:
        try:
            token_bytes = base64.b64decode(token_str)
            creds = pickle.loads(token_bytes)
        except Exception as e:
            print(f"Warning: Could not decode token from .env: {e}")
            creds = None
    
    # If no valid credentials, get new ones
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_FILE.exists():
                raise FileNotFoundError(
                    f"Credentials file not found at {CREDENTIALS_FILE}\n"
                    "Please download OAuth2 credentials from Google Cloud Console:\n"
                    "1. Go to https://console.cloud.google.com/\n"
                    "2. Create a new project\n"
                    "3. Enable Google Drive API\n"
                    "4. Create OAuth2 credentials (Desktop application)\n"
                    "5. Download and save as credentials.json"
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE, SCOPES
            )
            creds = flow.run_local_server(port=0)
        
        # Save token to .env for future use
        token_bytes = pickle.dumps(creds)
        token_str = base64.b64encode(token_bytes).decode('utf-8')
        set_key(ENV_FILE, 'GOOGLE_DRIVE_TOKEN', token_str)
        print("✓ Saved authentication token to .env")
    
    return creds

def get_or_create_upload_folder(service):
    """Get or create a Google Drive folder for uploading documents."""
    load_dotenv()
    folder_id = os.getenv('GOOGLE_DRIVE_FOLDER_ID')
    
    if folder_id:
        print(f"✓ Using existing folder: {folder_id}")
        return folder_id
    
    # Create a new folder
    file_metadata = {
        'name': 'NotebookLM Sources',
        'mimeType': 'application/vnd.google-apps.folder'
    }
    folder = service.files().create(body=file_metadata, fields='id').execute()
    folder_id = folder.get('id')
    
    # Save folder ID to .env for future use
    set_key(ENV_FILE, 'GOOGLE_DRIVE_FOLDER_ID', folder_id)
    print(f"✓ Created new folder: {folder_id}")
    print(f"✓ Saved folder ID to .env")
    
    return folder_id


def upsert_markdown_file(service, file_path, parent_folder_id, doc_id=None):
    """Create or update a Google Doc from a markdown file and return file size in MB."""
    if not parent_folder_id and not doc_id:
        raise ValueError("Either parent_folder_id or doc_id must be provided")

    # Read markdown file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Create a temporary text file for upload
    temp_file = f"/tmp/{file_path.stem}_{os.getpid()}.txt"
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(content)

    try:
        media = MediaFileUpload(temp_file, mimetype='text/markdown', resumable=True)

        if doc_id:
            doc = service.files().update(
                fileId=doc_id,
                media_body=media,
                fields='id, webViewLink, size'
            ).execute()
        else:
            file_name = file_path.stem
            file_metadata = {
                'name': file_name,
                'mimeType': 'application/vnd.google-apps.document',
                'parents': [parent_folder_id]
            }
            doc = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, webViewLink, size'
            ).execute()

        file_size_mb = float(doc.get('size', 0)) / (1024 * 1024)
        return file_size_mb
    finally:
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)

def check_existing_docs(service, folder_id):
    """Check for existing documents in the folder."""
    query = f"'{folder_id}' in parents and trashed=false"
    results = service.files().list(
        q=query,
        spaces='drive',
        fields='files(id, name)',
        pageSize=1000
    ).execute()
    
    existing_docs = {item['name']: item['id'] for item in results.get('files', [])}
    return existing_docs


def load_cache_data(cache_path):
    if not cache_path.exists():
        return {}
    with open(cache_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def save_cache_data(cache_path, cache_data):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2)


def clear_needs_upload(cache_path, uploaded_topics, failed_topics):
    """Clear needs_upload entries that were uploaded successfully."""
    cache_data = load_cache_data(cache_path)
    needs_upload = cache_data.get('needs_upload', [])
    if not isinstance(needs_upload, list):
        needs_upload = []

    uploaded_set = set(uploaded_topics)
    failed_set = set(failed_topics)
    remaining = [
        topic for topic in needs_upload
        if topic in failed_set or topic not in uploaded_set
    ]

    cache_data['needs_upload'] = sorted(set(remaining))
    if '_newly_matched_topics' in cache_data:
        del cache_data['_newly_matched_topics']
    save_cache_data(cache_path, cache_data)


def load_updated_markdown_files(source_dir, cache_path):
    """Load only changed markdown files from cache metadata produced by compile_sources.py."""
    if not cache_path.exists():
        print(f"No cache file found at: {cache_path}")
        return [], []

    with open(cache_path, 'r', encoding='utf-8') as f:
        cache_data = json.load(f)

    if not isinstance(cache_data, dict):
        print(f"Invalid cache format in {cache_path}")
        return [], []

    updated_topics = cache_data.get('needs_upload', [])
    if not isinstance(updated_topics, list):
        print(f"Invalid needs_upload field in {cache_path}")
        return [], []

    markdown_files = []
    selected_topics = []
    for topic in updated_topics:
        if not isinstance(topic, str):
            continue
        name = topic.replace("/", "-").replace(":", " -") + ".md"
        file_path = source_dir / name
        if file_path.exists() and file_path.suffix.lower() == '.md':
            markdown_files.append(file_path)
            selected_topics.append(topic)
        else:
            print(f"Warning: Skipping missing/non-markdown file from topic: {topic}")

    return sorted(markdown_files), sorted(set(selected_topics))

def main():
    parser = argparse.ArgumentParser(
        description="Upload compiled markdown files to Google Docs"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=MARKDOWN_DIR,
        help=f"Directory containing markdown files (default: {MARKDOWN_DIR})"
    )
    parser.add_argument(
        "--credentials",
        type=Path,
        default=CREDENTIALS_FILE,
        help="Path to Google OAuth2 credentials JSON file"
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=CACHE_FILE,
        help=f"Path to cache file (default: {CACHE_FILE})"
    )
    parser.add_argument(
        "--all-sources",
        action="store_true",
        help="Upload all markdown files from source-dir instead of only updated ones"
    )
    
    args = parser.parse_args()
    
    load_dotenv()
    
    print("Authenticating with Google Drive...")
    creds = authenticate_google_drive()
    
    print("Building Google Drive service...")
    service = build('drive', 'v3', credentials=creds)
    
    print("Getting or creating upload folder...")
    folder_id = get_or_create_upload_folder(service)
    
    # Find all markdown files
    source_dir = args.source_dir
    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        return
    
    if args.all_sources:
        markdown_files = sorted(source_dir.glob("*.md"))
        selected_topics = []
        print("Using all markdown files from source directory")
    else:
        markdown_files, selected_topics = load_updated_markdown_files(source_dir, args.cache_file)
        print(f"Using updated topics from cache: {args.cache_file}")
    
    print(f"Found {len(markdown_files)} markdown files to upload\n")

    if not markdown_files:
        print("No files to upload. Exiting.")
        return
    
    # Check for existing documents
    print("Checking for existing documents...")
    existing_docs = check_existing_docs(service, folder_id)
    
    new_upload_count = 0
    failed_uploads = []
    updated_upload_count = 0
    total_processed_size_mb = 0.0
    failed_topics = set()
    successful_topics = set()

    topic_by_stem = {
        topic.replace("/", "-").replace(":", " -"): topic
        for topic in selected_topics
    }
    
    for markdown_file in tqdm(markdown_files, desc="Uploading to Google Docs"):
        doc_name = markdown_file.stem
        existing_doc_id = existing_docs.get(doc_name)
        
        try:
            file_size = upsert_markdown_file(
                service=service,
                file_path=markdown_file,
                parent_folder_id=folder_id,
                doc_id=existing_doc_id,
            )
            rounded_size_mb = round(file_size, 2)
            total_processed_size_mb += rounded_size_mb

            if existing_doc_id:
                updated_upload_count += 1
            else:
                new_upload_count += 1

            topic = topic_by_stem.get(doc_name)
            if topic:
                successful_topics.add(topic)
        except Exception as e:
            failed_uploads.append({
                'file': markdown_file.name,
                'error': str(e)
            })
            print(f"\n✗ Failed to upload {markdown_file.name}: {e}")
            topic = topic_by_stem.get(doc_name)
            if topic:
                failed_topics.add(topic)
    
    # Print summary
    print("\n" + "="*95)
    print("Upload Summary")
    print("="*95)
    print(f"Newly uploaded: {new_upload_count}")
    print(f"Updated existing: {updated_upload_count}")
    print(f"Failed uploads: {len(failed_uploads)}")
    
    if failed_uploads:
        print("\nFailed uploads:")
        for failed in failed_uploads:
            print(f"  - {failed['file']}: {failed['error']}")
    
    total_uploaded = new_upload_count + updated_upload_count
    if total_uploaded > 0:
        print(f"\nTotal processed: {total_processed_size_mb:.2f} MB")

    if not args.all_sources:
        clear_needs_upload(
            args.cache_file,
            uploaded_topics=successful_topics,
            failed_topics=failed_topics,
        )
        if failed_topics:
            print(f"Remaining topics in needs_upload (failed): {len(failed_topics)}")
        else:
            print("Cleared needs_upload after successful upload")
    
    print(f"\nGoogle Drive folder: https://drive.google.com/drive/folders/{folder_id}")

if __name__ == "__main__":
    main()
