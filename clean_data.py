"""Data cleaning (Step 7.4, updated in Step 8.5). Safe to run more than once.

Keeps only the ORIGINAL data-collection rounds:
1. Random-computer rounds only. Adaptive-AI rounds were played by the author while testing,
   knowing how the AI works, so they are not a fair test of the AI.
2. Rounds from the data-collection day (27 Sep 2026) only. Rounds played later (tests,
   screenshots, the "Jhon" test player) were played by someone who already knew the patterns.
3. The two sessions are labelled honestly: "Nipun" and "Kasun" were not two different people;
   both sessions were played by the author under different names.
"""

import shutil

import pandas as pd

from stats import DATA_FILE

BACKUP = DATA_FILE.with_name("rounds_backup_before_cleaning.csv")
DATA_COLLECTION_DAY = "20260927"  # game IDs start with the date: YYYYMMDD
RENAME = {"Nipun": "Session 1", "Kasun": "Session 2"}

if __name__ == "__main__":
    if not BACKUP.exists():
        shutil.copy(DATA_FILE, BACKUP)
        print(f"Backup saved as {BACKUP.name}")

    data = pd.read_csv(DATA_FILE)
    before = len(data)

    data = data[data["ai_mode"] == "random"]
    data = data[data["game_id"].astype(str).str.startswith(DATA_COLLECTION_DAY)]
    data["player"] = data["player"].replace(RENAME)
    data = data[data["player"].isin(RENAME.values())]
    data.to_csv(DATA_FILE, index=False)

    print(f"Rows before: {before}, rows after: {len(data)}, removed: {before - len(data)}")
    print("Rounds per session:")
    print(data["player"].value_counts().to_string())
