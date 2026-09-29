"""One-time data cleaning (Step 7.4). Saves a backup first, then:

1. Removes test rounds: every adaptive-AI round (played by the author while testing, knowing
   how the AI works) and the 7-round test player "Jhon".
2. Relabels the two real data-collection sessions honestly. "Nipun" and "Kasun" were not two
   different people: both sessions were played by the author under different names.
"""

import shutil

import pandas as pd

from stats import DATA_FILE

BACKUP = DATA_FILE.with_name("rounds_backup_before_cleaning.csv")
RENAME = {"Nipun": "Session 1", "Kasun": "Session 2"}

if __name__ == "__main__":
    if BACKUP.exists():
        print(f"{BACKUP.name} already exists, so the cleaning has probably been done already.")
        print("Nothing was changed. (Delete the backup file only if you really want to run it again.)")
    else:
        shutil.copy(DATA_FILE, BACKUP)
        data = pd.read_csv(DATA_FILE)
        before = len(data)

        data = data[data["ai_mode"] == "random"]    # remove adaptive-AI test rounds
        data = data[data["player"].isin(RENAME)]    # remove the "Jhon" test rounds
        data["player"] = data["player"].replace(RENAME)
        data.to_csv(DATA_FILE, index=False)

        print(f"Backup saved as {BACKUP.name}")
        print(f"Rows before: {before}, rows after: {len(data)}, removed: {before - len(data)}")
        print("Rounds per session:")
        print(data["player"].value_counts().to_string())
