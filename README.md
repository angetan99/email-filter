# Email Filter

Python scripts for fetching Gmail messages, cleaning exported email data, and training a simple email classifier.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Workflow

1. Add your Gmail OAuth client secrets as `credentials.json`.
2. Fetch emails:

   ```bash
   python3 fetch_emails.py
   ```

3. Update `INPUT_FILE` in `clean_emails.py`, then clean the export:

   ```bash
   python3 clean_emails.py
   ```

4. Label the cleaned CSV and train the classifier:

   ```bash
   python3 train_classifier.py
   ```

Local credentials, OAuth tokens, email CSVs, virtual environments, and generated plots are intentionally ignored by git.
