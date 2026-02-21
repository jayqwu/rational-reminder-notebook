#!/usr/bin/env python3
"""Upload compiled markdown files to Google Docs with formatting"""

import argparse
import os
import json
import re
import base64
import pickle
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv, set_key
from tqdm import tqdm
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials as OAuth2Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Google Drive API scopes
SCOPES = ['https://www.googleapis.com/auth/drive']

# Paths
CREDENTIALS_FILE = Path("credentials.json")
ENV_FILE = Path(".env")
MARKDOWN_DIR = Path("output/categorized")

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

def markdown_to_plain_text(markdown_text):
    """
    Convert markdown to plain text while preserving structure and readability.
    
    This removes markdown syntax but preserves the content and adds visual separators
    for better readability in Google Docs.
    
    Args:
        markdown_text: The markdown content as string
    
    Returns:
        Plain text with preserved structure
    """
    text = markdown_text
    
    # Remove markdown link syntax [text](url) -> text (url)
    text = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'\1 (\2)', text)
    
    # Convert bold **text** -> text (keep content, remove markers)
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
    
    # Convert italics *text* -> text
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)
    
    # Convert headers # Text -> Text with extra line breaks for visual separation
    text = re.sub(r'^# ', '\n', text, flags=re.MULTILINE)
    text = re.sub(r'^## ', '\n', text, flags=re.MULTILINE)
    text = re.sub(r'^### ', '\n', text, flags=re.MULTILINE)
    text = re.sub(r'^#### ', '\n', text, flags=re.MULTILINE)
    
    # Convert bullet points - keep the dash with proper indentation
    # Already preserved in markdown, just ensure proper line breaks
    
    # Remove excessive blank lines (more than 2 consecutive)
    text = re.sub(r'\n\n\n+', '\n\n', text)
    
    # Trim leading/trailing whitespace
    text = text.strip()
    
    return text

def upload_markdown_file(service, file_path, parent_folder_id):
    """
    Upload a markdown file to Google Docs.
    
    Args:
        service: Google Drive API service
        file_path: Path to the markdown file
        parent_folder_id: ID of the parent folder in Google Drive
    
    Returns:
        Tuple of (document_id, document_url, file_size_mb)
    """
    # Read markdown file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create file metadata - document will be created as Google Doc
    file_name = file_path.stem  # Remove .md extension
    file_metadata = {
        'name': file_name,
        'mimeType': 'application/vnd.google-apps.document',
        'parents': [parent_folder_id]
    }
    
    # Convert markdown to plain text while preserving structure
    doc_content = markdown_to_plain_text(content)
    
    # Create a temporary text file for upload
    temp_file = f"/tmp/{file_path.stem}_{os.getpid()}.txt"
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(doc_content)
    
    try:
        # Upload with plain text mimetype
        # Google Drive will convert this to a Google Doc automatically
        media = MediaFileUpload(temp_file, mimetype='text/plain', resumable=True)
        
        doc = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, webViewLink, size'
        ).execute()
        
        file_size_mb = float(doc.get('size', 0)) / (1024 * 1024)
        
        return doc.get('id'), doc.get('webViewLink'), file_size_mb
    finally:
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)

def update_markdown_file(service, file_path, doc_id):
    """
    Update an existing markdown file in Google Docs with new content.
    Preserves the original File ID while replacing the content.
    
    Args:
        service: Google Drive API service
        file_path: Path to the markdown file
        doc_id: ID of the existing Google Doc to update
    
    Returns:
        Tuple of (document_id, document_url, file_size_mb)
    """
    # Read markdown file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Convert markdown to plain text while preserving structure
    doc_content = markdown_to_plain_text(content)
    
    # Create a temporary text file for upload
    temp_file = f"/tmp/{file_path.stem}_{os.getpid()}.txt"
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(doc_content)
    
    try:
        # Update the existing document with new content
        media = MediaFileUpload(temp_file, mimetype='text/plain', resumable=True)
        
        doc = service.files().update(
            fileId=doc_id,
            media_body=media,
            fields='id, webViewLink, size'
        ).execute()
        
        file_size_mb = float(doc.get('size', 0)) / (1024 * 1024)
        
        return doc.get('id'), doc.get('webViewLink'), file_size_mb
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
    
    markdown_files = sorted(source_dir.glob("*.md"))
    
    print(f"Found {len(markdown_files)} markdown files to upload\n")
    
    # Check for existing documents
    print("Checking for existing documents...")
    existing_docs = check_existing_docs(service, folder_id)
    
    upload_results = []
    failed_uploads = []
    updated_uploads = []
    
    for markdown_file in tqdm(markdown_files, desc="Uploading to Google Docs"):
        doc_name = markdown_file.stem
        
        try:
            # Check if document already exists
            if doc_name in existing_docs:
                # Update existing document
                doc_id, doc_url, file_size = update_markdown_file(service, markdown_file, existing_docs[doc_name])
                updated_uploads.append({
                    'file': markdown_file.name,
                    'doc_id': doc_id,
                    'url': doc_url,
                    'size_mb': round(file_size, 2)
                })
            else:
                # Create new document
                doc_id, doc_url, file_size = upload_markdown_file(service, markdown_file, folder_id)
                upload_results.append({
                    'file': markdown_file.name,
                    'doc_id': doc_id,
                    'url': doc_url,
                    'size_mb': round(file_size, 2)
                })
        except Exception as e:
            failed_uploads.append({
                'file': markdown_file.name,
                'error': str(e)
            })
            print(f"\n✗ Failed to upload {markdown_file.name}: {e}")
    
    # Save results
    results_file = Path("output/google_docs_uploads.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    results_data = {
        'folder_id': folder_id,
        'uploaded': upload_results,
        'updated': updated_uploads,
        'failed': failed_uploads
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2)
    
    # Print summary
    print("\n" + "="*95)
    print("Upload Summary")
    print("="*95)
    print(f"Newly uploaded: {len(upload_results)}")
    print(f"Updated existing: {len(updated_uploads)}")
    print(f"Failed uploads: {len(failed_uploads)}")
    
    if failed_uploads:
        print("\nFailed uploads:")
        for failed in failed_uploads:
            print(f"  - {failed['file']}: {failed['error']}")
    
    total_uploaded = len(upload_results) + len(updated_uploads)
    if total_uploaded > 0:
        total_size = sum(doc['size_mb'] for doc in upload_results + updated_uploads)
        print(f"\nTotal processed: {total_size:.2f} MB")
    
    print(f"\nGoogle Drive folder: https://drive.google.com/drive/folders/{folder_id}")
    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()
