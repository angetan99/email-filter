# Email Filter

Python scripts for fetching Gmail messages, training an email classifier, and automatically classifying/filtering unread emails.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add your Gmail OAuth client secrets as `credentials.json` in the repo root.

## Workflow

### 1. Fetch Emails
Export existing emails from Gmail for initial labeling and training:

```bash
python3 fetch_emails.py
```

This outputs a CSV file (e.g., `emails_2026_05_27_001910.csv`) with email metadata and bodies.

### 2. Clean Emails
Clean the exported CSV, removing HTML, extra whitespace, and filtering for training data:

```bash
python3 clean_emails.py
```

Update `INPUT_FILE` in `clean_emails.py` to point to your export if needed. Outputs `emails_cleaned.csv`.

### 3. Train Classifier
Manually label the cleaned CSV (`emails_cleaned.csv`), then train the model:

```bash
python3 train_classifier.py
```

This generates `email_classifier.pkl` (logistic regression model) and `vectorizer.pkl` (TF-IDF vectorizer), along with evaluation plots in `train results/`.

### 4. Classify & Filter Unread Emails (Production)
Automatically classify unread Gmail messages and apply labels:

```bash
python3 classify_emails.py
```

This script:
- Loads the trained model and vectorizer
- Fetches unread emails from your Gmail inbox
- Classifies each as "Junk" (moved to Promotions folder) or "Important"
- Logs results to SQLite database (`email_logs.db`)

**Can be run manually or scheduled as a cron job.**

## Configuration

Edit constants in `classify_emails.py`:
- `THRESHOLD`: Classification probability threshold (default: 0.75)
- `MAX_EMAILS`: Max emails to classify per run (default: 50)
- `SCOPES`: Gmail API scopes (modify if adding features)

## Cron Job Example

To classify emails hourly (adjust frequency as needed):

```bash
0 * * * * /path/to/venv/bin/python3 /path/to/email-filter/classify_emails.py >> /tmp/email_filter.log 2>&1
```

## Data & Credentials

- `credentials.json`: Gmail OAuth 2.0 client secret (local only, ignored by git)
- `token.json`: Refresh token created on first run (local only, ignored by git)
- `email_classifier.pkl`, `vectorizer.pkl`: Trained model artifacts (local only, ignored by git)
- `email_logs.db`: SQLite database for classification history (local only, ignored by git)

All sensitive files and generated data are ignored by `.gitignore`.
