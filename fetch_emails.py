"""
fetch_emails.py

Fetches emails from Gmail and saves them to a CSV for ML classification.
Columns: Sender, Subject, Body, Date, Label

Requirements:
    pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib beautifulsoup4 lxml

Usage:
    python3 fetch_emails.py

Notes:
    - credentials.json must be in the same directory (from Google Cloud Console)
    - token.json will be created automatically on first run (OAuth browser flow)
    - The 'Label' column is left blank for you to fill in manually, or you can
      use the auto-labeling logic at the bottom to get a rough starting point
"""

import os
import base64
import csv
import sys
from time import strftime, gmtime

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from bs4 import BeautifulSoup

# ------------------------------------------------------------
# CONFIG — adjust these to control what gets fetched
# ------------------------------------------------------------

# gmail.readonly is safest — we're only reading, not modifying anything
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# How many emails to fetch total (start with 500, increase later)
MAX_EMAILS = 2000

# Optional: filter to specific Gmail labels
# e.g. ["INBOX"] for inbox only, [] for everything including sent/spam
LABEL_IDS = ["INBOX"]

# Senders that should be auto-labeled as "junk" (rough starting point)
# You'll manually review and correct these in the CSV afterwards
JUNK_SENDERS = [
    "linkedin",
    "noreply",
    "no-reply",
    "newsletter",
    "promotions",
    "marketing",
    "notifications",
]

# ------------------------------------------------------------
# AUTH — reuses the same pattern as quickstart.py
# ------------------------------------------------------------

def get_gmail_service():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)

# ------------------------------------------------------------
# FETCH EMAIL LIST — handles pagination past the 500 limit
# ------------------------------------------------------------

def list_messages(service, label_ids=[], max_results=500):
    messages = []
    try:
        response = service.users().messages().list(
            userId="me",
            labelIds=label_ids,
            maxResults=min(max_results, 500)  # API max per page is 500
        ).execute()

        if "messages" in response:
            messages.extend(response["messages"])

        # keep paginating until we have enough or run out
        while "nextPageToken" in response and len(messages) < max_results:
            page_token = response["nextPageToken"]
            response = service.users().messages().list(
                userId="me",
                labelIds=label_ids,
                pageToken=page_token,
                maxResults=min(max_results - len(messages), 500)
            ).execute()
            if "messages" in response:
                messages.extend(response["messages"])
            print(f"... fetched {len(messages)} email IDs so far")
            sys.stdout.flush()

    except HttpError as e:
        print(f"Error listing messages: {e}")

    return messages[:max_results]

# ------------------------------------------------------------
# EXTRACT EMAIL DETAILS
# ------------------------------------------------------------

def get_header(headers, name):
    """Pull a specific header value (e.g. Subject, From, Date) from headers list."""
    for h in headers:
        if h["name"].lower() == name.lower():
            return h["value"]
    return ""


def decode_body(data):
    """Decode base64-encoded email body to plain text."""
    try:
        decoded = base64.urlsafe_b64decode(data.encode("UTF-8")).decode("UTF-8", errors="replace")
        # strip HTML tags if present
        soup = BeautifulSoup(decoded, "lxml")
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        return ""


def extract_body(payload):
    """
    Recursively find the email body.
    Emails can be simple (body directly in payload) or multipart (body nested in parts).
    """
    # simple email — body is directly in payload
    if "body" in payload and payload["body"].get("data"):
        return decode_body(payload["body"]["data"])

    # multipart email — body is in parts
    if "parts" in payload:
        for part in payload["parts"]:
            mime_type = part.get("mimeType", "")
            # prefer plain text; fall back to html
            if mime_type == "text/plain" and part["body"].get("data"):
                return decode_body(part["body"]["data"])
        # if no plain text found, try html
        for part in payload["parts"]:
            if part.get("mimeType") == "text/html" and part["body"].get("data"):
                return decode_body(part["body"]["data"])
        # recurse into nested multipart
        for part in payload["parts"]:
            result = extract_body(part)
            if result:
                return result

    return ""


def read_email(service, msg_id):
    """Fetch a single email and return a dict with Sender, Subject, Body, Date."""
    try:
        message = service.users().messages().get(userId="me", id=msg_id).execute()
        payload = message["payload"]
        headers = payload.get("headers", [])

        sender  = get_header(headers, "From")
        subject = get_header(headers, "Subject")
        date    = get_header(headers, "Date")
        body    = extract_body(payload)

        return {
            "Sender":  sender,
            "Subject": subject,
            "Body":    body[:2000],  # cap body length to keep CSV manageable
            "Date":    date,
            "Label":   auto_label(sender),  # rough starting label — review manually
        }

    except Exception as e:
        print(f"  Error reading message {msg_id}: {e}")
        return None

# ------------------------------------------------------------
# AUTO-LABELING — rough heuristic to give you a starting point
# Edit or remove this once you start labeling manually
# ------------------------------------------------------------

def auto_label(sender):
    """
    Very rough label based on sender. Returns 'junk' or blank.
    You should review and correct these in the CSV — this is just
    to give you a starting point so not every row is blank.
    """
    sender_lower = sender.lower()
    for keyword in JUNK_SENDERS:
        if keyword in sender_lower:
            return "junk"
    return ""  # leave blank for you to label as 'important' or 'junk'

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":
    print("\n--- Gmail Email Fetcher ---")

    print("\n[1/3] Authenticating...")
    service = get_gmail_service()
    print("     Done.")

    print(f"\n[2/3] Listing up to {MAX_EMAILS} emails...")
    email_list = list_messages(service, label_ids=LABEL_IDS, max_results=MAX_EMAILS)
    print(f"     Found {len(email_list)} emails.")

    output_file = f"emails_{strftime('%Y_%m_%d_%H%M%S', gmtime())}.csv"
    print(f"\n[3/3] Fetching email content and saving to {output_file}...")

    rows_written = 0
    rows_failed  = 0

    with open(output_file, "w", encoding="utf-8", newline="") as csvfile:
        fieldnames = ["Sender", "Subject", "Body", "Date", "Label"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, email in enumerate(email_list):
            email_data = read_email(service, email["id"])

            if email_data:
                writer.writerow(email_data)
                rows_written += 1
            else:
                rows_failed += 1

            if (i + 1) % 50 == 0:
                print(f"     ... {i + 1}/{len(email_list)} processed")
                sys.stdout.flush()

    print(f"\n--- Done ---")
    print(f"Saved:  {rows_written} emails → {output_file}")
    print(f"Failed: {rows_failed} emails (errors printed above)")
    print(f"\nNext step: open {output_file} and fill in the 'Label' column")
    print("  Use 'junk' for promotional/noise emails")
    print("  Use 'important' for emails you'd want to keep")
