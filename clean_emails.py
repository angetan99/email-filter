"""
clean_emails.py

Cleans raw fetched email CSV and prepares it for ML classification.

Steps:
    1. Fixes garbled unicode characters (encoding issues)
    2. Combines Sender + Subject + Body into a single 'Text' field
    3. Drops rows with no usable signal at all
    4. Saves cleaned version ready for labeling and training

Requirements:
    pip install pandas ftfy

Usage:
    python3 clean_emails.py

Input:  your raw emails CSV (set INPUT_FILE below)
Output: emails_cleaned.csv
"""

import pandas as pd
import ftfy

# ------------------------------------------------------------
# CONFIG — set this to your actual CSV filename
# ------------------------------------------------------------

INPUT_FILE  = "emails_2026_05_27_001910.csv"  # replace with your actual filename
OUTPUT_FILE = "emails_cleaned.csv"



# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

print(f"\n[1/4] Loading {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)
original_count = len(df)
print(f"      {original_count} rows loaded.")

# ------------------------------------------------------------
# FIX ENCODING
# ftfy (fixes text for you) corrects garbled unicode like ‚Äá Õè ‚Äå
# which happens when emails encoded in windows-1252 or latin-1
# get decoded as UTF-8
# ------------------------------------------------------------

print("\n[2/4] Fixing garbled unicode characters...")

for col in ["Sender", "Subject", "Body"]:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: ftfy.fix_text(str(x)) if pd.notna(x) else "")

print("      Done.")

# ------------------------------------------------------------
# COMBINE FIELDS INTO SINGLE TEXT COLUMN
# The model trains on this single field rather than separate columns.
# This way emails with missing body still contribute via sender/subject.
# ------------------------------------------------------------

print("\n[3/4] Combining Sender + Subject + Body into Text field...")

def combine_fields(row):
    parts = []

    # combine subject and body only — sender stays as its own column
    if str(row["Subject"]).strip():
        parts.append(str(row["Subject"]).strip())

    # include body regardless of length — even short bodies like
    # "Hi Angela," carry some signal and shouldn't be discarded
    body = str(row["Body"]).strip()
    if body:
        parts.append(body)

    return " ".join(parts)

df["Text"] = df.apply(combine_fields, axis=1)

# Drop only rows where both subject and body are empty
# — meaning Text is completely blank. Sender alone isn't
# enough signal to train on, so these rows aren't useful.
before_drop = len(df)
df = df[df["Text"].str.strip() != ""]
dropped = before_drop - len(df)

print(f"      Dropped {dropped} rows where both subject and body were empty.")
print(f"      {len(df)} rows remaining.")

# ------------------------------------------------------------
# REORDER COLUMNS
# Sender stays as its own column alongside Text
# ------------------------------------------------------------

cols = ["Sender", "Subject", "Body", "Date", "Label", "Text"]
cols = [c for c in cols if c in df.columns]  # only keep cols that exist
df = df[cols]

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

print(f"\n[4/4] Saving cleaned data to {OUTPUT_FILE}...")
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
print(f"      Done. {len(df)} rows saved.")

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print(f"\n--- Summary ---")
print(f"Original rows:  {original_count}")
print(f"Dropped rows:   {original_count - len(df)}")
print(f"Remaining rows: {len(df)}")

if "Label" in df.columns:
    label_counts = df["Label"].value_counts(dropna=False)
    print(f"\nLabel breakdown:")
    for label, count in label_counts.items():
        print(f"  '{label}': {count} rows")

print(f"\nNext step: open {OUTPUT_FILE} and fill in the blank 'Label' column")
print("  Use 'junk'      for promotional/noise emails")
print("  Use 'important' for emails you want to keep")
print("\nTip: sort by 'Label' column to review auto-labeled rows first,")
print("     then work through the blank ones.")
