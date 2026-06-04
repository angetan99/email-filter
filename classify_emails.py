"""
classify_emails.py

Fetches unread emails from Gmail and classifies them using the trained model.
Applies 'Junk' or 'Important' label based on model prediction.

Requirements:
    pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib beautifulsoup4 lxml joblib scikit-learn
"""

import os
import base64
import joblib
import numpy as np
from bs4 import BeautifulSoup
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import sqlite3
from time import strftime, gmtime

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

# need modify scope now since we're applying labels
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

# your chosen threshold from Phase 3
THRESHOLD = 0.75

# how many unread emails to classify at a time
# MAX_EMAILS = 50

# ------------------------------------------------------------
# SQLite logging setup
# ------------------------------------------------------------

# Local repo base directory
BASE_DIR = os.path.dirname(__file__)

# SQLite database file for logging
DB_PATH = os.path.join(BASE_DIR, "email_logs.db")

def setup_db(conn):
    # conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS runs (
        timestamp TEXT,
        emails_processed INTEGER,
        junk INTEGER,
        important INTEGER,
        junk_pct REAL
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS emails (
        timestamp TEXT,
        sender TEXT,
        subject TEXT,
        probability REAL,
        label TEXT
    )''')
    conn.commit()
    # conn.close()

def log_run(conn, total, junk, important):
    # conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    timestamp = strftime('%Y-%m-%d %H:%M:%S', gmtime())
    junk_pct = round((junk / total * 100), 1) if total > 0 else 0
    row = (timestamp, total, junk, important, junk_pct)
    c.execute('INSERT INTO runs VALUES (?, ?, ?, ?, ?)', row)
    conn.commit()
    # conn.close()

def log_email(conn, sender, subject, probability, label):
    # conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    timestamp = strftime('%Y-%m-%d %H:%M:%S', gmtime())
    row = (timestamp, sender, subject[:200], probability, label)
    c.execute('INSERT INTO emails VALUES (?, ?, ?, ?, ?)', row)
    conn.commit()
    # conn.close()
# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

print("Loading model and vectorizer...")
lr = joblib.load(os.path.join(BASE_DIR, "email_classifier.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))
print("Done.")

# ------------------------------------------------------------
# AUTH — same as fetch_emails.py
# ------------------------------------------------------------

CREDENTIALS_PATH = os.path.join(BASE_DIR, "credentials.json")
TOKEN_PATH = os.path.join(BASE_DIR, "token.json")

def get_gmail_service():
    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, "w") as token:
            token.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)

# ------------------------------------------------------------
# HELPERS — reused from fetch_emails.py
# ------------------------------------------------------------

def get_header(headers, name):
    for h in headers:
        if h["name"].lower() == name.lower():
            return h["value"]
    return ""

def decode_body(data):
    try:
        decoded = base64.urlsafe_b64decode(data.encode("UTF-8")).decode("UTF-8", errors="replace")
        soup = BeautifulSoup(decoded, "lxml")
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        return ""

def extract_body(payload):
    if "body" in payload and payload["body"].get("data"):
        return decode_body(payload["body"]["data"])
    if "parts" in payload:
        for part in payload["parts"]:
            if part.get("mimeType") == "text/plain" and part["body"].get("data"):
                return decode_body(part["body"]["data"])
        for part in payload["parts"]:
            if part.get("mimeType") == "text/html" and part["body"].get("data"):
                return decode_body(part["body"]["data"])
        for part in payload["parts"]:
            result = extract_body(part)
            if result:
                return result
    return ""

# ------------------------------------------------------------
# CLASSIFY + GET UNREAD
# ------------------------------------------------------------

def classify_email(sender, subject, body):
    combined = sender + ' ' + subject + ' ' + body[:2000]
    vectorized = vectorizer.transform([combined])
    prob = lr.predict_proba(vectorized)[0][1]
    return prob

def get_unread_messages(service):
    messages = []
    response = service.users().messages().list(
        userId="me",
        labelIds=["INBOX", "UNREAD"]
    ).execute()
    messages.extend(response.get("messages", []))
    while "nextPageToken" in response:
        response = service.users().messages().list(
            userId="me",
            labelIds=["INBOX", "UNREAD"],
            pageToken=response["nextPageToken"]
        ).execute()
        messages.extend(response.get("messages", []))
    return messages

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":

    print("\n--- Email Classifier ---")

    print("\nAuthenticating...")
    service = get_gmail_service()
    print("     Done.")

    # print("Setting up database...")
    # setup_db()

    service = get_gmail_service()
    results = service.users().labels().list(userId='me').execute()
    for label in results['labels']:
        if 'CATEGORY' in label['id']:
            print(label['id'], '|', label['name']) 

    print("Setting up database...")
    conn = sqlite3.connect(DB_PATH, timeout=10)
    setup_db(conn)

    messages = get_unread_messages(service)
    EXCLUDE_LABELS = {"CATEGORY_PROMOTIONS", "CATEGORY_SOCIAL", "CATEGORY_FORUMS"}

    filtered = []
    for msg in messages:
        full = service.users().messages().get(userId="me", id=msg["id"], format="metadata").execute()
        msg_labels = set(full.get("labelIds", []))
        if not msg_labels & EXCLUDE_LABELS:
            filtered.append(msg)

    messages = filtered

    if not messages:
        print("No unread emails. Exiting.")
        conn.close()
        exit()

    print(f"Found {len(messages)} Primary unread emails.")

    junk_count = 0
    important_count = 0

    for i, msg in enumerate(messages):
        message = service.users().messages().get(userId="me", id=msg["id"]).execute()
        payload = message["payload"]
        headers = payload.get("headers", [])

        sender = get_header(headers, "From")
        subject = get_header(headers, "Subject")
        body = extract_body(payload)

        prob = classify_email(sender, subject, body)

        if prob >= THRESHOLD:
            service.users().messages().modify(
            userId="me",
            id=msg["id"],
            body={
                "addLabelIds": ["CATEGORY_PROMOTIONS"],
                "removeLabelIds": ["INBOX"]
            }
            ).execute()
            label = "Promotions"
            junk_count += 1
        else:
            label = "Important"
            important_count += 1
        
        log_email(conn, sender, subject, prob, label)
        print(f"  [{i+1}/{len(messages)}] {label} ({prob:.3f}) — {subject[:60]}")

    print(f"\n--- Done ---")
    print(f"Junk:      {junk_count}")
    print(f"Important: {important_count}")
    log_run(conn, len(messages), junk_count, important_count)
    conn.close()
    print(f"Logged to database.")