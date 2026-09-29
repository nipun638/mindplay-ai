# MindPlay AI — Project Log

> A running record of how this project was built: what was done at each step, why, what went wrong, and what I learned.
> This file is updated after every completed step.

---

## 1. Project Overview

**Project name:** MindPlay AI — Adaptive Rock-Paper-Scissors

**Research question:**
> Can statistical pattern analysis and machine learning be used to predict a player's next move in Rock-Paper-Scissors?

**Goal:** Build a small, complete, and explainable undergraduate portfolio project that demonstrates a basic data science workflow:

Problem → Data collection → Data exploration → Statistical analysis → Feature creation → ML model → Evaluation → Interpretation

**Priority order:** Correctness → Statistics → ML → Evaluation → Clean UI → Documentation

---

## 2. Technology Stack

| Tool | Purpose |
|---|---|
| Python 3.14 | Programming language |
| Streamlit | Interactive web app (game + dashboard) |
| pandas | Storing and analysing the round-by-round dataset |
| NumPy | Numerical calculations |
| scikit-learn | ML models and evaluation metrics |
| matplotlib | Visualisations |

**Development environment:** Windows, VS Code, PowerShell terminal, Python virtual environment (`.venv`)

**Repository:** https://github.com/nipun638/mindplay-ai

**Project location:** `D:\My_Own_Projects\mindplay-ai`

---

## 3. Progress Tracker

| Stage | Description | Status |
|---|---|---|
| 0 | Setup (Python, folder, virtual environment, libraries) | ✅ Done |
| 1 | Basic game | ✅ Done |
| 2 | Data collection | ✅ Done |
| 3 | Statistics | ✅ Done |
| 4 | Machine learning | ✅ Done |
| 5 | Model evaluation | ✅ Done |
| 6 | Visualisation | ✅ Done |
| 7 | Final Streamlit interface | ✅ Done |
| 8 | GitHub repository + README + LinkedIn post | ✅ Done |

Legend: ✅ Done · 🟡 In progress · ⬜ Not started

---

## 4. Development Log

### Step 0.1 — Check Python installation ✅

**What:** Checked that Python and pip are installed and working.

**Why:** Windows can have multiple Python installations or a fake Microsoft Store shortcut. Confirming the setup first prevents confusing errors later.

**Commands used:**
```powershell
python --version
py --list
python -m pip --version
```

**Result:** Python 3.14 and pip are installed and working.

**Learned:**
- `python -m pip` runs pip using that exact Python, which avoids mix-ups when multiple Pythons are installed.
- `py --list` (Windows Python Launcher) shows all installed Python versions.

---

### Step 0.2 — Create project folder and virtual environment ✅

**What:** Created the project folder and a virtual environment (`.venv`) inside it, then activated it.

**Why:** A virtual environment is a private copy of Python for this project only. Libraries installed here do not affect other projects, and the exact dependencies can later be listed in `requirements.txt` so others can reproduce the project.

**Commands used:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
```

**Result:** Prompt shows `(.venv)`, and Python runs from `D:\My_Own_Projects\mindplay-ai\.venv\Scripts\python.exe`.

**Problems encountered and fixes:**

| Problem | Cause | Fix |
|---|---|---|
| Terminal was "JavaSE-26" instead of PowerShell | A Java extension had set its own terminal profile | Opened a PowerShell terminal from the `+ ⌄` menu |
| `Activate.ps1 cannot be loaded because running scripts is disabled` | Windows blocks PowerShell scripts by default | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| Typing `y` alone gave "term 'y' is not recognized" | `Y` is only an answer to a prompt, not a command | Ran the full `Set-ExecutionPolicy` command instead |
| Folder path contained spaces (`My Own Projects`) | Spaces can cause issues in commands | Renamed to `My_Own_Projects` |

**Learned:**
- Always check for `(.venv)` at the start of the prompt before working. If missing, run `.venv\Scripts\Activate.ps1`.
- Never put my own code inside the `.venv` folder.
- Avoid spaces and OneDrive folders for project paths.

---

### Step 0.3 — Install libraries and connect VS Code ✅

**What:** Installed streamlit, pandas, numpy, scikit-learn, and matplotlib into the virtual environment, and selected the `.venv` interpreter in VS Code.

**Why:** Each library has a specific role (see Section 2). Selecting the `.venv` interpreter makes VS Code run files with the project's Python and recognise the installed libraries.

**Commands used:**
```powershell
python -m pip install --upgrade pip
python -m pip install streamlit pandas numpy scikit-learn matplotlib
python -c "import streamlit, pandas, numpy, sklearn, matplotlib; print('All libraries imported OK')"
```
VS Code: `Ctrl + Shift + P` → **Python: Select Interpreter** → `.venv`

**Result:** `All libraries imported OK`

**Problems encountered and fixes:**

| Problem | Cause | Fix |
|---|---|---|
| `ImportError: DLL load failed while importing _uarray: An Application Control policy has blocked this file.` | Windows 11 **Smart App Control** blocked unsigned compiled files (`.pyd`) inside SciPy, which scikit-learn depends on | Turned off Smart App Control (personal laptop). Microsoft Defender Antivirus remains active. |

**Alternatives considered:** Using WSL (Linux inside Windows) or GitHub Codespaces (cloud VS Code), both of which keep Smart App Control on. Rejected for now because they add setup complexity.

**Learned:**
- Installing a library successfully does not guarantee it can be imported. Always test with an import check.
- Reading an error from the **last line** upward is usually the fastest way to find the real cause.
- scikit-learn depends on SciPy, so a problem in SciPy shows up when importing sklearn.

---

### Step 0.4 — Final setup verification ✅

**What:** Checked the Python version, the environment location, and all library versions in one command, then ran Streamlit's built-in demo app.

**Why:** To confirm the whole setup works end-to-end before writing any project code.

**Commands used:**
```powershell
python -c "import sys, streamlit, pandas, numpy, sklearn, matplotlib; print(...versions...)"
streamlit hello
```

**Result — installed versions:**

| Package | Version |
|---|---|
| Python | 3.14.7 |
| streamlit | 1.64.0 |
| pandas | 3.0.6 |
| numpy | 2.5.3 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |

Streamlit server started successfully at `http://localhost:8501`.

**Problems encountered and fixes:**

| Problem | Cause | Fix |
|---|---|---|
| Errors like `numpy : The term 'numpy' is not recognized` | Pasted the *expected output* into the terminal along with the command | Ran only the command; expected output is for comparison, not typing |

**Learned:**
- A Streamlit app runs as a small local web server; `localhost:8501` means "this computer, port 8501".
- Stop a running Streamlit app with `Ctrl + C` in the terminal.

---

### Step 1.1 — Write the game rules (`game_logic.py`) ✅

**What:** Created `game_logic.py` containing the three moves, a random computer move, and a function that decides Win / Loss / Draw.

**Why:** The rules are kept in a separate file so the terminal game, the Streamlit app, and the ML code can all reuse the same logic ("separating logic from interface"). Testing the rules first ensures **correctness**, the project's top priority.

**Key code ideas:**
- `MOVES` — a list of the three moves.
- `BEATS` — a dictionary encoding the rules as `winner: loser` (e.g., `"Rock": "Scissors"`).
- `get_random_move()` — computer picks each move with probability 1/3.
- `determine_result(player_move, computer_move)` — returns the result from the player's point of view.
- `if __name__ == "__main__":` — self-test runs only when the file is run directly, not when imported.

**Test:** Ran `python game_logic.py`, which checks all 3 × 3 = 9 move combinations.

**Result:** All 9 combinations gave the correct result.

**Learned:**
- Test **every** possible case when the number of cases is small (exhaustive testing).
- A dictionary can replace many `if` statements when encoding rules.
- The terminal must be PowerShell with `(.venv)` active; the JavaSE-26 terminal runs a different Python.

---

### Step 1.2 — Playable terminal game (`play_terminal.py`) ✅

**What:** Built a terminal game loop: the player types r / p / s, the computer plays randomly, and the game shows the result, round number, and running score. `n` starts a new game, `q` quits and shows the win rate.

**Why:** Completes the Stage 1 features (moves, result, score, rounds, reset) as a quick prototype before building the Streamlit version. Also practises importing code from another file.

**Key code ideas:**
- `from game_logic import get_random_move, determine_result` — reuses Step 1.1 code.
- `while True:` loop with `break` (quit) and `continue` (skip invalid input / reset).
- `input().strip().lower()` — cleans user input.
- `scores[result] += 1` — dictionary keys match the result strings, so no extra `if` statements.
- Win rate = wins ÷ rounds, guarded by `if round_number > 0` to avoid division by zero.

**Tests performed:**

| Test | Expected behaviour | Passed? |
|---|---|---|
| Play rounds with `r`, `p`, `s` | Correct result each round; score updates | ✅ (Round 1: Rock vs Scissors → Win) |
| Wins + Losses + Draws | Always equals the round number | ☐ |
| Invalid input `x` | Warning shown; not counted as a round | ☐ |
| New game `n` | Scores and rounds reset to 0 | ☐ |
| Quit `q` | Shows total rounds and win rate | ☐ |

**Result:** Game plays correctly. (Tick the remaining boxes after running the full test.)

**Problems encountered and fixes:**

| Problem | Cause | Fix |
|---|---|---|
| `can't open file ... play_terminal.py: No such file or directory` | Ran the command before creating the file | Created `play_terminal.py` first, then ran it |
| Red `KeyboardInterrupt` traceback | Pressed `Ctrl + C` while the game was waiting for input, which force-stops Python | Not a bug. Quit the game with `q` instead |

**Learned:**
- Python can only run files that exist in the current folder (or a given path).
- `Ctrl + C` force-stops a running Python program and raises `KeyboardInterrupt`. Use the program's own quit option (`q`) for a clean exit.
- `__pycache__` is created automatically when one file imports another; it stores pre-compiled code. Don't edit it; exclude it from GitHub later.
- VS Code's bottom-right corner shows the active interpreter: `.venv (3.14.7)` confirms the project environment is used.
- First statistic of the project: **win rate**.

---

### Step 1.3 — Streamlit version of the game (`app.py`) ✅

**What:** Built the game as a Streamlit web app: Rock / Paper / Scissors buttons, the result of the latest round, a scoreboard (Rounds, Wins, Losses, Draws), and a New game button.

**Why:** The final project is a Streamlit app, and clicking buttons makes it practical to play hundreds of rounds for data collection in Stage 2. `game_logic.py` was reused unchanged, which shows the benefit of separating logic from interface.

**Key code ideas:**
- Streamlit **re-runs the whole script on every click**, so normal variables reset.
- `st.session_state` — memory that survives re-runs; initialised once with `if "scores" not in st.session_state:`.
- `st.button(..., on_click=play_round, args=(move,))` — the callback updates state before the page re-runs.
- Layout: `st.columns`, `st.metric`, `st.success` / `st.error` / `st.warning`.

**How to run:** `streamlit run app.py` → opens `http://localhost:8501`. Stop with `Ctrl + C`.

**Result:** Buttons play rounds correctly, the scoreboard updates, and New game resets everything.

**Learned:**
- Web apps need explicit "memory" (session state) because the script re-runs.
- A callback is a function passed to a button, to be run when it is clicked.

**✅ Stage 1 complete.**

---

### Step 2.1 — Record every round as a table ✅

**What:** The Streamlit app now records every round (not just the score) in `st.session_state.history` and shows it as a pandas DataFrame under the game.

**Why:** This turns the game into a data science project. Each round becomes one **row** of the dataset. The score only says *how many* wins; the round history keeps *what the player did and in what order*, which is where patterns can be found.

**Variables collected (raw data only):**

| Column | Example | Reason |
|---|---|---|
| `game_id` | `20260926-213512` | Identifies the game session, so patterns from separate games are not mixed |
| `round` | `7` | Keeps the order of moves |
| `player_move` | `Rock` | Main variable to analyse and predict |
| `computer_move` | `Paper` | Needed for the result; the player may react to it |
| `result` | `Loss` | Win/loss rates; behaviour may change after a win or a loss |

**Design decision:** Derived features such as `previous_player_move` are **not** stored. They will be created later with pandas (**feature engineering**), keeping the raw dataset clean and free of duplicate information.

**Key code ideas:**
- `reset_game()` now initialises everything (scores, history, new `game_id`) and is also used on the first run, so starting values live in one place.
- `history` is a list of dictionaries; `.append({...})` adds one round.
- `datetime.now().strftime("%Y%m%d-%H%M%S")` creates a simple unique game ID.
- `pd.DataFrame(history)` converts the list of dictionaries into a table (keys → columns, dictionaries → rows).
- `st.dataframe(df, hide_index=True)` displays the table.

**Result:** One row per round with the correct values; New game clears the table and creates a new `game_id`.

**Limitation (fixed in Step 2.2):** Data is lost when the app is closed or a new game starts.

**Learned:**
- A dataset is simply rows (observations) and columns (variables).
- Store raw data; derive features later.

---

### Step 2.2 — Save every round to a CSV file ✅

**What:** Each round is appended to `data/rounds.csv` as soon as it is played. The app also shows a summary of the saved dataset (total rounds, number of games).

**Why:** Session state is lost when the app closes. Statistics and ML need hundreds of rounds collected over several sessions, so the data must be stored permanently. CSV is a simple, universal format readable by pandas, Excel, and R.

**Key code ideas:**
- `DATA_FILE = Path(__file__).parent / "data" / "rounds.csv"` — the file is always saved next to `app.py`, wherever the terminal is.
- `mkdir(exist_ok=True)` — creates the `data` folder only if it is missing.
- `to_csv(mode="a", header=not file_exists, index=False)` — append mode; the header row is written only once.
- `pd.read_csv()` and `nunique()` — read the saved data back and count distinct games.

**Tests performed:**

| Test | Expected behaviour | Passed? |
|---|---|---|
| Play rounds | Rows appear in `data/rounds.csv` | ✅ |
| Header row | Written only once, at the top | ✅ |
| New game | New `game_id`; `round` restarts at 1 | ✅ (2 games: 12 + 5 rounds) |
| Result correctness | Every saved result matches the rules | ✅ (all 17 rows checked) |
| Restart the app | Saved total stays at 17 rounds, 2 games | ✅ |

**Result:** Data is stored permanently and accumulates across games and app restarts.

**Observation (not a conclusion):** 7 of the first 17 rounds were draws. With such a small sample this is very likely chance; conclusions will only come from formal tests on a larger dataset in Stage 3.

**Learned:**
- Append mode (`"a"`) adds to a file; write mode (`"w"`) would overwrite it.
- Small samples can show striking-looking patterns purely by chance.

---

### Step 2.3 — Add a player name ✅

**What:** Added a "Player name" box. Every saved round now includes a `player` column, and the saved-dataset summary shows the number of players and rounds per player.

**Why:** Different people have different patterns. Mixing players would hide individual patterns, and the ML model would learn an "average" player instead of a specific one. The `player` column lets each player's data be analysed separately. (Answers the open question "Should data from different players be kept separate?" → **Yes**.)

**Data dictionary (final Stage 2 schema):**

| Column | Type | Description |
|---|---|---|
| `player` | text | Cleaned player name (e.g., `Nipun`) |
| `game_id` | text | Game session ID from date and time (`YYYYMMDD-HHMMSS`) |
| `round` | integer | Round number within the game, starting at 1 |
| `player_move` | text | Rock / Paper / Scissors |
| `computer_move` | text | Rock / Paper / Scissors (random during data collection) |
| `result` | text | Win / Loss / Draw, from the player's point of view |

**Key code ideas:**
- `st.text_input(..., key="player_name", on_change=reset_game)` — the name is stored in session state; changing it starts a new game so one `game_id` never contains two players.
- `clean_name()` — `strip().title()` so `prabod`, `Prabod ` and `PRABOD` are saved identically (simple data cleaning).
- `st.stop()` — no move buttons until a name is entered, so no rows are saved without a player.
- `value_counts()` — rounds per player.

**Problems encountered and fixes:**

| Problem | Cause | Fix |
|---|---|---|
| `KeyError: 'player'` when the app loaded | The old `rounds.csv` (5 columns, no `player`) was still there, but the new code expects 6 columns (**schema change**) | Stopped the app, deleted the old test file, restarted; a new file with the correct header was created |

**Result:** 53 rounds, 4 games, 3 player names saved correctly.

**Data-quality observation:** "Tharaka" and "Tharaka2" are stored as two different players. `clean_name()` fixes spacing and capitals, but not different spellings, so each player must type **exactly the same name** every session.

**Learned:**
- `KeyError` means the requested key (here, a column name) does not exist.
- Changing a dataset's columns (schema) breaks old files; plan the schema before collecting real data.
- Data cleaning can fix format differences, but not inconsistent labels.

---

### Step 2.4 — Real data collection ✅

**What:** Deleted the test data, then collected real gameplay data against the **random** computer.

**Collection rules followed:**
- Play naturally (not trying to be random or to create patterns).
- The computer plays uniformly at random during collection (controlled condition: any pattern comes from the player, not from reacting to an adaptive AI).
- Same exact player name every session; no manual edits to the CSV.

> ⚠️ **Correction (Step 7.4):** Both "players" below were the author playing under two names. Read "Nipun" as **Session 1** and "Kasun" as **Session 2** of the same player.

**Dataset collected (27 Sep 2026):**

| Player | Rounds |
|---|---|
| Nipun | 251 |
| Kasun | 109 |
| **Total** | **360 rounds, 2 games** |

**Target check:** Nipun ≥ 150 rounds ✅ (enough for a 3 × 3 transition matrix with ~28 observations per row). Kasun 109 rounds, usable for comparison; more rounds would make his results more reliable.

**Limitation noted:** Each player's rounds were played in one long session. Fatigue or boredom could change behaviour over time. This can be checked in Stage 3 (e.g., compare the first and second half of the rounds).

**✅ Stage 2 complete.**

---

### Step 3.1 — Load, validate, and describe the data (`stats.py`) ✅

**What:** Created `stats.py` with reusable functions: `load_data()`, `check_data()` (missing values, invalid moves, and re-checking every result against the game rules), `move_summary()` (count and proportion of each move), and `outcome_rates()` (win/loss/draw proportions). Results are computed separately for each player with `groupby("player")`.

**Why:** Data must be validated before analysis (correctness first). Keeping statistics in their own module lets the Streamlit app reuse them later.

**Results — move distribution (random = 0.333 each):**

| Player | Rounds | Rock | Paper | Scissors |
|---|---|---|---|---|
| Nipun | 251 | 75 (0.299) | **107 (0.426)** | 69 (0.275) |
| Kasun | 109 | 39 (0.358) | **44 (0.404)** | 26 (0.239) |

**Results — outcome rates:**

| Player | Win | Loss | Draw |
|---|---|---|---|
| Nipun | 0.339 | 0.367 | 0.295 |
| Kasun | 0.257 | 0.330 | 0.413 |

**Interpretation:**
- Both players chose **Paper** most often (about 40–43%). Whether this is a real preference or chance is tested formally in Step 3.2.
- Outcome rates are close to 1/3 **by design**: against a uniformly random opponent, every player strategy has P(win) = P(loss) = P(draw) = 1/3. So outcome rates say nothing about player skill; they mainly confirm the computer is random. Kasun's higher draw rate (0.413) is within the range expected by chance for 109 rounds (about 1.8 standard deviations from 1/3).

**Learned:**
- `isin`, `~` (NOT), `value_counts(normalize=True)`, `reindex`, `groupby`.
- Validate data before analysing it.
- Understand what a statistic *can* tell you before interpreting it (outcome rates vs a random opponent).

---

### Step 3.2 — Chi-square goodness-of-fit test ✅

**Question:** Does the player choose Rock, Paper, and Scissors equally often?

**Method:** Chi-square goodness-of-fit test (`scipy.stats.chisquare`), computed per player.
- H₀: P(Rock) = P(Paper) = P(Scissors) = 1/3
- H₁: at least one move is chosen more or less often than 1/3
- Statistic: χ² = Σ (Observed − Expected)² / Expected, with Expected = n / 3
- Degrees of freedom = 3 − 1 = 2; significance level α = 0.05
- Assumption: expected count ≥ 5 in every category (Nipun 83.7, Kasun 36.3 ✅)
- The statistic was also calculated manually; it matched SciPy exactly, confirming the formula.

**Results:**

| Player | n | χ² | df | p-value | Decision (α = 0.05) |
|---|---|---|---|---|---|
| Nipun | 251 | 9.976 | 2 | **0.0068** | **Reject H₀**: moves are not equally likely |
| Kasun | 109 | 4.752 | 2 | 0.0929 | Do not reject H₀ |

Results were independently re-computed from the counts and matched.

**Interpretation:**
- **Nipun** shows statistically significant evidence of a move preference. Paper contributes most of the χ² (6.51 of 9.98): he played Paper 42.6% of the time vs 33.3% expected.
- **Kasun** has a very similar Paper proportion (40.4%), but the result is not significant. The main reason is **sample size**: with only 109 rounds, the same size of deviation provides less evidence. "Do not reject H₀" does **not** prove Kasun is random; it means there is not enough evidence yet.
- **Practical meaning (in-sample only):** A strategy of always playing Scissors against Nipun would have won 42.6% and lost 29.9% of these rounds. Whether such a pattern predicts *future* moves must be tested on unseen data (Stages 4–5).

**Learned:**
- Null / alternative hypotheses, test statistic, degrees of freedom, p-value, α.
- Statistical significance depends on both the size of the effect **and** the sample size.
- "Not significant" ≠ "no effect".

---

### Step 3.3 — Transition matrix and chi-square test of independence ✅

**Question:** Does the player's next move depend on their previous move?

**Feature engineering:** `add_previous_moves()` creates `prev_player_move`, `prev_computer_move`, and `prev_result` with `groupby(["player", "game_id"]).shift(1)`, so the previous move never crosses from one game into another. The first round of each game has no previous move and is dropped.

**Method:**
- Transition matrix with `pd.crosstab` (rows = previous move, columns = next move); dividing by row totals gives P(next | previous).
- Repeat rate = diagonal ÷ total (random = 1/3).
- Chi-square test of independence (`scipy.stats.chi2_contingency`). H₀: the next move is independent of the previous move. df = (3 − 1)(3 − 1) = 4. Assumption: every expected count ≥ 5.

**Results — Nipun (250 transitions):**

| Previous ↓ / Next → | Rock | Paper | Scissors |
|---|---|---|---|
| Rock (75) | 0.267 | **0.547** | 0.187 |
| Paper (106) | **0.406** | 0.321 | 0.274 |
| Scissors (69) | 0.159 | **0.464** | 0.377 |

- Repeat rate: 0.320 (random 0.333)
- χ² = 19.107, df = 4, **p = 0.0007**, smallest expected count = 19.0 ✅ → **Reject H₀**
- Independently re-computed from the counts and matched.

**Results — Kasun (108 transitions):**

| Previous ↓ / Next → | Rock | Paper | Scissors |
|---|---|---|---|
| Rock (39) | 0.231 | **0.615** | 0.154 |
| Paper (43) | **0.581** | 0.140 | 0.279 |
| Scissors (26) | 0.154 | **0.538** | 0.308 |

- Repeat rate: **0.213** (random 0.333): Kasun avoids repeating moves
- χ² = 25.741, df = 4, **p < 0.0001** (p = 0.000036), smallest expected count = 6.3 ✅ → **Reject H₀**
- Independently re-computed from the counts and matched.

**Interpretation (Kasun):**
- Kasun's move *frequencies* were not significantly different from random (Step 3.2), but his *sequence* is very strongly non-random. **A player can look random by frequency and still be highly predictable by sequence.**
- Same Rock ↔ Paper alternation as Nipun, but stronger: Rock → Paper 61.5%, Paper → Rock 58.1%, and he rarely repeats Paper (14.0%).
- In-sample, guessing the most likely next move given the previous move would have been correct **58.3%** of the time (vs 40.7% for "always Paper"). This is optimistic and must be verified on unseen data.

**Interpretation (Nipun):**
- Strong evidence that Nipun's next move depends on his previous move. This is the strongest finding so far.
- Main pattern: after **Rock → Paper** (54.7%), after **Paper → Rock** (40.6%), and after Scissors he rarely plays Rock (15.9%). Standardised residuals were largest for "Paper → Rock" (+2.08, more than expected) and "Scissors → Rock" (−2.09, less than expected).
- The repeat rate is close to random, so the predictable part is **how he switches**, not whether he repeats.
- **In-sample preview for ML:** predicting the most likely next move given the previous move would have been correct 46.4% of the time, vs 42.8% for "always Paper" and 33.3% for random guessing. These are in-sample (optimistic) figures; real performance must be measured on unseen data in Stage 5.

**Learned:**
- `shift()` for creating lag features; why the shift must be done within each game.
- A pattern can exist in the *sequence* of moves even when overall frequencies look balanced.
- In-sample accuracy is optimistic; evaluate on unseen data.

---

### Step 3.4 — Win-stay / lose-shift analysis ✅

**Question:** Does the previous result (Win / Loss / Draw) change whether the player repeats ("stays") or changes ("shifts") their move?

**Method:** Each round is labelled Stay (same move as the previous round) or Shift. A 3 × 2 table (previous result × Stay/Shift) gives the stay rate after each result (random = 1/3). Chi-square test of independence, H₀: the chance of staying does not depend on the previous result; df = (3 − 1)(2 − 1) = 2.

**Results:**

| Player | Stay after Win | Stay after Loss | Stay after Draw | χ² | p-value | Decision |
|---|---|---|---|---|---|---|
| Nipun (n = 250) | 0.306 | 0.326 | 0.329 | 0.119 | 0.9421 | Do not reject H₀ |
| Kasun (n = 108) | 0.286 | 0.200 | 0.178 | 1.252 | 0.5348 | Do not reject H₀ |

Smallest expected counts: Nipun 23.4, Kasun 6.0 ✅. Independently re-computed and matched.

**Interpretation:**
- **No evidence of win-stay / lose-shift** for either player. Neither player's stay rate after a win is above 1/3.
- This differs from the pattern reported for real players by Wang, Xu & Zhou (2014). Possible reasons: small samples; a different setting (a random computer opponent, one-click buttons, no stakes); and in our data the **previous move** matters much more than the previous result.
- Kasun's stay rates are all below 1/3, which matches his low overall repeat rate (0.213), not a reaction to results.
- A non-significant result is still a valid finding and is reported honestly.

**Fix in this step:** Kasun's independence-test p-value (0.000036) was printed as `0.0` because of rounding to 4 decimals. Added `format_p()` so very small p-values are shown as `< 0.0001`. A p-value is never exactly 0.

**Learned:**
- Research findings don't always replicate in a different setting or sample.
- Rounding can make a result look misleading; report small p-values as "< 0.0001".

---

### Step 3.5 — Report small p-values correctly ✅

**What:** Added `format_p()`; p-values are stored unrounded and only formatted when printed. Values below 0.0001 are shown as `< 0.0001`.

**Why:** Kasun's independence-test p-value (0.000036) was printed as `0.0`. A p-value is never exactly 0, so this was misleading.

**Result:** Kasun's line now reads `chi2 = 25.741, df = 4, p-value = < 0.0001`; all other results unchanged.

**✅ Stage 3 complete.**

---

### Stage 3 — Summary of findings

> ⚠️ **Correction (Step 7.4):** "Nipun" = Session 1 and "Kasun" = Session 2 of **the same player** (the author). Findings describe one person's play in two sessions, not two people.

**Central question:** Do these players play Rock-Paper-Scissors randomly?

| Test | Nipun (251 rounds) | Kasun (109 rounds) |
|---|---|---|
| Equal move frequencies (goodness-of-fit) | **Not random**: prefers Paper (p = 0.0068) | Not significant (p = 0.0929) |
| Next move depends on previous move (independence) | **Yes** (p = 0.0007) | **Yes, strongly** (p < 0.0001) |
| Win-stay / lose-shift | No evidence (p = 0.94) | No evidence (p = 0.53) |

**Answer:** Neither player behaves like a random player. The strongest and most consistent pattern is **sequential**: both alternate between Rock and Paper (Rock → Paper, Paper → Rock). The previous **result** does not appear to influence their choices.

**Implication for Stage 4 (ML):** The most useful feature is expected to be `prev_player_move`. `prev_result` is expected to add little. This will be checked with the models.

**Limitations:**
- Only two players, so results describe these individuals, not people in general.
- Each player's data comes from one long session.
- Several tests were run on the same data; with α = 0.05, some "significant" results can occur by chance. All three significant results (p = 0.0068, 0.0007, and < 0.0001) remain significant even under a strict correction (Bonferroni for 6 tests: 0.05 / 6 ≈ 0.0083).
- All pattern strengths so far are in-sample.

---

### Step 4.1 — Build the ML dataset (`model.py`) ✅

**What:** Created `model.py`. For each player, `build_dataset()` turns the rounds into a feature table **X** and a target **y**, and `time_split()` divides them into training and test sets.

**Features and target:**

| Role | Column | Reason |
|---|---|---|
| Feature | `prev_player_move` | Strongest pattern in Stage 3 (p = 0.0007 and p < 0.0001) |
| Feature | `prev_computer_move` | Lets the model check whether the player reacts to the computer's move |
| Feature | `prev_result` | No effect found in Stage 3; kept so the model can confirm this |
| Target | `player_move` | The player's next move |

**Data leakage check:** All features come from the **previous** round, so they are known before the player chooses. Using the current round's `computer_move` or `result` would leak information the AI would not have in a real game.

**Note:** `prev_result` is fully determined by the two previous moves, so it adds no new information; it is a readable summary.

**Encoding:** One-hot encoding (`pd.get_dummies`) → 9 binary columns (3 features × 3 categories). Numbers like Rock = 1, Paper = 2, Scissors = 3 were avoided because they would impose a false order. `pd.Categorical` guarantees all 9 columns exist.

**Split:** Time-based, not shuffled: the first 80% of rounds for training, the last 20% for testing. This simulates a real game, where the AI can only learn from the past to predict the future.

| Player | Examples | Train | Test |
|---|---|---|---|
| Nipun | 250 | 200 | 50 |
| Kasun | 108 | 86 | 22 |

**Target distribution:**

| Move | Nipun train | Nipun test | Kasun train | Kasun test |
|---|---|---|---|---|
| Rock | 0.300 | 0.280 | 0.360 | 0.318 |
| Paper | 0.415 | 0.480 | 0.395 | 0.455 |
| Scissors | 0.285 | 0.240 | 0.244 | 0.227 |

**Checks:** Rows match `rounds.csv` (e.g., Nipun's first example: previous Rock vs Scissors → Win, next move Rock). Train + test counts add up to each player's move totals minus the first round.

**Observation:** Paper is even more common in both test sets than in training. With only 50 and 22 test rounds, this is likely normal random variation, but it means the "always Paper" baseline will score relatively high on the test set, which makes it a tough baseline to beat.

**Limitation:** Small test sets make test accuracy noisy (for Kasun, one prediction changes accuracy by 4.5 percentage points).

**Learned:**
- Features (X) vs target (y); data leakage; one-hot encoding; time-based split.

---

### Step 4.2 — Decision Tree vs baselines ✅

**What:** Trained a `DecisionTreeClassifier(max_depth=3, random_state=42)` on the training rounds and compared it on the test rounds with two baselines: random guessing (1/3) and a majority baseline (`DummyClassifier(strategy="most_frequent")`, which always predicts Paper).

**Why a Decision Tree:** It learns readable if-then rules, handles categorical features, and its rules can be compared with the Stage 3 transition matrix. `max_depth=3` limits complexity to reduce **overfitting** (memorising noise in a small training set). `random_state=42` makes results reproducible.

**Results:**

| Player | Model | Train | Test | Correct (test) |
|---|---|---|---|---|
| Nipun | Random guess | 0.333 | 0.333 | ≈17/50 |
| Nipun | Always Paper | 0.415 | **0.480** | **24/50** |
| Nipun | Decision Tree | 0.470 | 0.440 | 22/50 |
| Kasun | Random guess | 0.333 | 0.333 | ≈7/22 |
| Kasun | Always Paper | 0.395 | 0.455 | 10/22 |
| Kasun | Decision Tree | 0.605 | **0.500** | **11/22** |

(The majority baseline's accuracy equals the Paper proportion in each set, because "always Paper" is correct exactly when the player plays Paper.)

**The rules the trees learned:** Both trees contain several splits, but every leaf on the same side of the first split predicts the same move. The trees therefore reduce to **one identical rule for both players**:

> If the previous move was **Paper** → predict **Rock**; otherwise → predict **Paper**.

This is the Rock ↔ Paper alternation found in the Stage 3 transition matrices: **the ML model rediscovered the statistical pattern**.

**Feature importance:** `prev_player_move_Paper` was the most important feature for both players (Nipun 0.586, Kasun 0.772), as predicted in Stage 3. `prev_result_Loss` (Nipun 0.292) and `prev_result_Draw` (Kasun 0.118) received some importance but **did not change any prediction**. Feature importance measures how much a split made the groups "purer", not whether it changed the predicted move.

**Interpretation (honest):**
- **Nipun:** The tree scored **below** the majority baseline on the test set (22 vs 24 correct). No evidence that the model beats the simple rule.
- **Kasun:** The tree beat the baseline by **one** prediction (11 vs 10 of 22). This is far too small a difference to call an improvement.
- Test sets are small, so test accuracy is noisy: the standard error is about ±0.07 for 50 rounds and ±0.11 for 22 rounds. The differences observed are much smaller than this noise.
- The train–test gap is small for Nipun (0.47 → 0.44) and larger for Kasun (0.605 → 0.50), which may reflect mild overfitting on his smaller training set.
- **Conclusion so far:** The sequential pattern exists (Stage 3), but with this single train/test split we **cannot** show that the model predicts future moves better than "always Paper". A more reliable evaluation is planned for Stage 5.

**Learned:**
- `fit()` / `predict()`, `accuracy_score`, `export_text`, `feature_importances_`.
- Always compare with a majority baseline, not only with random guessing.
- A high-looking accuracy can come from class imbalance alone.
- Feature importance ≠ effect on predictions.

---

### Step 4.3 — Naive Bayes and feature-set comparison ✅

**What:** Added `CategoricalNB` (Naive Bayes) and compared two feature sets for both models: **"prev move only"** (`prev_player_move`) and **"all 3 features"**.

**Why Naive Bayes:** It is built on the same conditional probabilities as Stage 3, combined with Bayes' rule: P(next | features) ∝ P(next) × Π P(featureᵢ | next). It is "naive" because it assumes the features are independent given the next move. That assumption is **false** here, because `prev_result` is determined by the two previous moves. It uses Laplace smoothing (α = 1) so unseen combinations never get probability 0. Features are encoded as integer codes; `CategoricalNB` treats them as labels, not as ordered numbers.

**Results (test set, same time-based split):**

| Model | Features | Nipun (test = 50) | Kasun (test = 22) |
|---|---|---|---|
| Always Paper | none | **24/50** (0.480) | 10/22 (0.455) |
| Decision Tree | prev move only | 22/50 (0.440) | **11/22** (0.500) |
| Naive Bayes | prev move only | 22/50 (0.440) | **11/22** (0.500) |
| Decision Tree | all 3 features | 22/50 (0.440) | **11/22** (0.500) |
| Naive Bayes | all 3 features | 19/50 (0.380) | 9/22 (0.409) |

**Naive Bayes (prev move only) — learned probabilities (training rounds):**

| Previous move | Nipun: prediction (P) | Kasun: prediction (P) |
|---|---|---|
| Rock | Paper (0.504) | Paper (0.577) |
| Paper | Rock (0.422) | Rock (0.601) |
| Scissors | Paper (0.487) | Paper (0.574) |

**Interpretation:**
- Three of the four models make **identical predictions** and reduce to the same rule: *after Paper → Rock, otherwise → Paper*. With a single feature, both models simply choose the most likely next move for each previous move.
- **Adding `prev_result` and `prev_computer_move` never helped.** The tree ignored them for its predictions, and Naive Bayes got **worse** with them (Nipun 22 → 19, Kasun 11 → 9). The redundant `prev_result` breaks the independence assumption, so Naive Bayes "double-counts" the same evidence. This matches Stage 3 (no win-stay / lose-shift effect).
- With one feature, Naive Bayes is essentially the **Stage 3 transition matrix** used as a predictor. Its probabilities are close to the Stage 3 table but not identical, because they come from the training rounds only (80%) and include smoothing.
- The single-split comparison with the majority baseline is still inconclusive (Nipun −2, Kasun +1).

**Model chosen for the rest of the project:** **Naive Bayes, prev move only.** It is the simplest model with the best result, it outputs probabilities (useful for showing the AI's confidence in the app), and it is directly explainable as "P(next move | previous move)".

**✅ Stage 4 complete.**

**Learned:**
- Simpler models (fewer features) can generalise as well or better.
- Model assumptions matter: correlated features hurt Naive Bayes.
- Different algorithms can learn the same rule when the signal is simple.

---

### Step 5.1 — Walk-forward evaluation and significance tests (`evaluate.py`) ✅

**Why:** The single 80/20 split gave only 50 (Nipun) and 22 (Kasun) test predictions, which is too few to tell models apart.

**Method — walk-forward evaluation:** For every example after a warm-up of 20, the model is trained on **all earlier** rounds and predicts the **current** round. This gives 230 (Nipun) and 88 (Kasun) out-of-sample predictions, never uses future data, and is exactly how the adaptive AI will work in the game. The majority baseline is also walk-forward ("most frequent move so far").

**Tests:**
- **Test 1, vs random:** one-sided exact binomial test. H₀: accuracy = 1/3; H₁: accuracy > 1/3.
- **Test 2, vs majority baseline:** exact McNemar test (paired, because both models predict the same rounds). Only rounds where exactly one model is correct are informative; under H₀ they split 50/50, so the count for Naive Bayes ~ Binomial(b + c, 0.5).
- **Practical metric:** AI outcome if it had played the move that beats each prediction.

**Results:**

| | Nipun (230 predictions) | Kasun (88 predictions) |
|---|---|---|
| Random guess (expected) | 0.333 (≈77) | 0.333 (≈29) |
| Majority so far | 0.409 (94) | 0.398 (35) |
| **Naive Bayes (prev move only)** | **0.448 (103)** | **0.568 (50)** |
| 95% CI for NB accuracy (exact) | 0.382 – 0.515 | 0.458 – 0.673 |
| Test 1: NB vs random | **p = 0.0002** | **p < 0.0001** |
| Test 2: only NB right / only majority right | 43 / 34 | 20 / 5 |
| Test 2: NB vs majority (McNemar) | p = 0.1810 (not significant) | **p = 0.0020** |
| AI win / loss / draw (counter-move) | 0.448 / 0.239 / 0.313 | 0.568 / 0.239 / 0.193 |

All p-values, counts and intervals were independently re-computed and matched.

**Interpretation:**
- **Kasun:** The model is significantly better than random **and** than the majority baseline (p = 0.002, still significant after a Bonferroni correction for 4 tests: 0.05 / 4 = 0.0125). It predicted 57% of his moves; as a counter-strategy the AI would have won 56.8% and lost only 23.9%. **Clear evidence that the ML model learned a real, useful pattern.**
- **Nipun:** The model is significantly better than random, but **not** significantly better than simply predicting his most frequent move (p = 0.18). Most of his predictability comes from his Paper preference; the sequential information adds a small gain (+9 correct) that is not proven.
- **Prediction check:** Kasun benefited most, despite having less data, because his sequential pattern is much stronger (Stage 3: χ² = 25.7 vs 19.1, repeat rate 0.213).
- **Lesson on evaluation design:** The single split showed Kasun +1 and Nipun −2 vs the baseline. With 4× more test predictions, a clear effect appeared for Kasun. **Small test sets can hide real effects** (and can also create fake ones).
- Note: the AI win rate equals prediction accuracy, because the counter-move wins exactly when the prediction is correct.

**Limitations:**
- This is a **simulation** on recorded games against a random computer. A real adaptive AI could change how players behave (they might notice and adjust), so live performance may differ. This will be checked in Stage 7 by labelling adaptive games separately.
- Only two players; each played one long session.

**Learned:**
- Walk-forward (expanding-window) evaluation for sequential data.
- Binomial test for accuracy vs chance; McNemar test for paired model comparison.
- Confidence intervals show how uncertain an accuracy estimate is.

---

### Step 5.2 — Confusion matrix and classification report ✅

**What:** Added a confusion matrix (`sklearn.metrics.confusion_matrix`) and a classification report (precision, recall, F1, support) for the walk-forward Naive Bayes predictions.

**Results — confusion matrices (rows = actual, columns = predicted):**

| Kasun | Rock | Paper | Scissors |
|---|---|---|---|
| **Rock** | **20** | 12 | 0 |
| **Paper** | 5 | **30** | 0 |
| **Scissors** | 9 | 12 | 0 |

| Nipun | Rock | Paper | Scissors |
|---|---|---|---|
| **Rock** | **39** | 25 | 0 |
| **Paper** | 37 | **64** | 0 |
| **Scissors** | 30 | 35 | 0 |

**Per-move metrics:**

| | Kasun precision | Kasun recall | Nipun precision | Nipun recall |
|---|---|---|---|---|
| Rock | 0.588 | 0.625 | 0.368 | 0.609 |
| Paper | 0.556 | 0.857 | 0.516 | 0.634 |
| Scissors | 0.000 | 0.000 | 0.000 | 0.000 |

Checks: diagonals sum to 50 (Kasun) and 103 (Nipun), matching Step 5.1; totals 88 and 230.

**Interpretation:**
- **The model never predicts Scissors.** After every previous move, Scissors was never the most likely next move, so a classifier that picks the most probable class never chooses it. Scissors recall is therefore 0, and every Scissors move (Kasun 24%, Nipun 28%) is missed. This is expected behaviour, not a bug.
- **Kasun:** Rock and Paper are predicted well (Paper recall 0.857, Rock precision 0.588), which reflects his Rock ↔ Paper alternation.
- **Nipun:** Rock predictions are weak (precision 0.368): when the model says "Rock", he often plays Paper or Scissors instead.
- **Where the AI's losses come from:** The counter to "Rock" is Paper, which loses to Scissors; the counter to "Paper" is Scissors, which loses to Rock. For Kasun, 9 + 12 = 21 losses (0.239), matching Step 5.1 exactly.
- **Macro vs weighted average:** The macro average treats all three moves equally, so it is pulled down by Scissors (0.000). The weighted average weights each move by how often it occurs.

**Possible improvement (not implemented):** Choosing the AI move that maximises *expected* score from the predicted probabilities, instead of simply countering the single most likely move, could reduce losses when a move is only slightly more likely. Kept out to keep the project simple.

**✅ Stage 5 complete.**

**Learned:**
- Confusion matrix, precision, recall, F1, support; macro vs weighted averages.
- Accuracy alone hides which classes a model ignores.

---

### Step 6.1 — Statistics charts (`visualize.py`) ✅

**What:** Created `visualize.py`, which saves PNG charts into an `images/` folder:
- `move_distribution.png` — grouped bar chart of each player's move proportions, with a dashed "random (1/3)" reference line.
- `transition_matrix.png` — one heatmap per player of P(next move | previous move).

**Design decisions:**
- Player colours blue (`#2a78d6`) and orange (`#eb6834`), checked with a colour-blindness validator (all checks passed). The legend and value labels mean colour is never the only way to identify a player.
- Heatmaps use a single-hue blue scale (light = low, dark = high) instead of a rainbow, and the **same scale (0–70%)** for both players so they can be compared fairly.
- Cell text switches between white and black based on the cell's brightness, so every number is readable.
- Titles are phrased as the question each chart answers; grid and axes are kept light so the data stands out.
- `matplotlib.use("Agg")` saves charts straight to files without opening windows.

**Result:** Both charts generated in `images/`.

**Learned:**
- `plt.subplots`, `ax.bar`, `ax.axhline`, `ax.imshow`, `ax.text`, `fig.savefig`.
- Good charts answer one question; colour should encode one thing consistently.

---

### Step 6.2 — Model charts ✅

**What:** Added two charts to `visualize.py` (walk-forward predictions are computed once per player and reused):
- `model_accuracy.png` — majority baseline (grey) vs Naive Bayes (violet) walk-forward accuracy per player, with **exact 95% confidence intervals** (Clopper–Pearson, `binomtest(...).proportion_ci()`), a random-guess line, and the one-sided McNemar p-value above each player (bold when p < 0.05).
- `confusion_matrix.png` — one heatmap per player; each cell shows the count and the share of that actual move, coloured by the share so players with different numbers of predictions can be compared.

**Design decisions:**
- Models get their own colours (violet / grey) so blue and orange keep meaning "players" across all charts.
- Accuracy labels sit at the base of each bar, so they never collide with error bars or the random line (layout was stress-tested with a very tall bar).
- A footnote states what the error bars and p-values are.

**Result:** All four charts are generated in `images/`:

| File | Shows |
|---|---|
| `move_distribution.png` | Move proportions vs 1/3 |
| `transition_matrix.png` | P(next move \| previous move) |
| `model_accuracy.png` | Model vs baseline, with 95% CIs and McNemar p-values |
| `confusion_matrix.png` | Where the model is right and wrong |

**✅ Stage 6 complete.**

**Learned:**
- Error bars show uncertainty; a chart without them can overstate a difference.
- Dictionary comprehensions; reusing computed results for several charts.

---

### Step 7.1 — Adaptive AI opponent (`ai.py`, `app.py`) ✅

**What:** Added an **Opponent** choice to the app: *Random computer* or *Adaptive AI*. The adaptive AI:
1. Loads all of this player's saved rounds.
2. Trains the Stage 5 model (Naive Bayes, previous move only).
3. Predicts the player's next move from their previous move in the current game.
4. Plays the move that **beats** the prediction (`COUNTER`).

It plays randomly when there is no previous move (round 1) or fewer than 20 training examples (the same `WARMUP` as the walk-forward evaluation), which handles the **cold-start** problem for new players. After each round the app explains the AI's decision, e.g. *"The AI predicted you would play Paper (58% likely), so it played Scissors."*

**Fairness:** The AI decides using only past rounds; it never looks at the button the player has just clicked.

**Schema change and migration:** Two new columns: `ai_mode` (`random` / `adaptive`) and `ai_prediction` (empty in random mode). Instead of deleting data (as in Step 2.3), `upgrade_data_file()` runs once at start-up and adds `ai_mode = "random"` to all existing rows, because they were all played against the random computer. The CSV was backed up first.

**Keeping the analysis valid:** Players may behave differently against an adaptive opponent, so `load_data()` now returns **only random-mode rounds by default**. The Stage 3–6 results are therefore unchanged by adaptive games. `check_data()` ignores the empty `ai_prediction` column.

**Tests:** Verified with Streamlit's `AppTest` on a copy of the project (schema upgrade, round 1 random, predictions from round 2, no exceptions), then tested by hand.

**Result (user test):** The app works: 382 rounds saved, Kasun 109 random + 11 adaptive, Nipun 251 random + 4 adaptive, and the AI's explanation is shown each round.

**Problem found and fixed:** A new player (Jhon) played 7 rounds in random mode. Those rounds entered the analysis dataset, which made `evaluate.py` and `visualize.py` crash and would have run chi-square tests on far too little data. Added `MIN_ROUNDS = 50` and `keep_players_with_enough_data()` in `stats.py`; `stats.py`, `model.py`, `evaluate.py` and `visualize.py` now skip players below the minimum and print which players were skipped. (50 because the transition test needs about 45+ transitions for expected counts ≥ 5 in 9 cells, and walk-forward needs 20 to start.)

**Learned:**
- Cold-start problem; schema migration instead of deleting data; backing up before changing data files.
- Separating data by experimental condition (random vs adaptive opponent).
- Code that works for today's data can break when new data arrives; guard against small groups.

---

### Step 7.2 — Player Statistics tab ✅

**What:** Split the app into tabs (`st.tabs`) and added a **📊 Player statistics** tab. For a chosen player (random-mode rounds only) it shows:
1. Move proportions with "± pts vs random" and a bar chart in Rock–Paper–Scissors order.
2. Win / loss / draw rates (with a note that against a random opponent these are ≈ 1/3 by design) and the goodness-of-fit test.
3. The transition matrix as a coloured table, the independence test, and the repeat rate.
4. Stay rates after a win / loss / draw and the win-stay test.

Every test result is written as a plain-language sentence with its p-value (e.g., "✅ Nipun does not choose the moves equally often (p = 0.0068)"). Players with fewer than 50 rounds see descriptive statistics only, plus a warning.

**Code changes:**
- The page is split into functions (`render_game`, `render_statistics`, `render_saved_data`).
- `st.stop()` was replaced with `return`: `st.stop()` would have stopped the **whole page**, leaving the statistics tab empty when no name was entered.
- All numbers come from the `stats.py` functions, so the app always matches `results.txt`.
- The AI message in the game became a visible blue info box (it was easy to miss as small grey text).
- Checked in a real browser: fixed a chart whose labels were sideways and sorted alphabetically, and text contrast in the coloured table.

**Learned:**
- `st.tabs`, `st.selectbox`, `st.metric` deltas, pandas `Styler.background_gradient`.
- `st.stop()` vs `return` in a multi-tab page.
- Always look at the rendered result; code that runs can still look wrong.

---

### Step 7.3 — AI / ML tab ✅

**What:** Added a third tab, **🤖 AI / ML**, for a chosen player:
1. **How the AI thinks:** the model in two sentences and its decision table (previous move → predicted move, probability → AI's move), using the same `predict_next_move()` as the game.
2. **Does it predict better than guessing?** Walk-forward accuracy vs random and "most frequent" baselines, 95% CI, binomial and McNemar tests in plain language, simulated AI win/loss/draw rates.
3. **Where is it right and wrong?** Confusion matrix as a coloured table, with a note when a move is never predicted.
4. **How did it do in real games?** Prediction accuracy and AI win/loss rate in real rounds against the adaptive AI, with a warning when there are fewer than 30.

**Code:** `@st.cache_data` caches the walk-forward evaluation (0.4–0.7 s per player), so the game does not slow down on every click; it recomputes only when the data changes. A guard handles players whose rounds are split across games and have fewer than 20 training pairs.

**Result (Kasun):**

| Item | Value |
|---|---|
| AI decision table | Rock → Paper (60%), Paper → Rock (57%), Scissors → Paper (53%) |
| Walk-forward accuracy | 57% (50/88) vs 40% most frequent, 95% CI 46–67% |
| Tests | Binomial p < 0.0001; McNemar p = 0.0020 |
| Real adaptive games | 21 predictions, 48% correct, AI win 48% / loss 14% |

All values were independently re-computed and matched (the decision-table probabilities include Laplace smoothing, so they are slightly closer to 1/3 than the raw Stage 3 proportions).

**Interpretation of the real games:** 48% is below the walk-forward 57%, but with only 21 predictions the 95% CI is roughly 26–70%, so the difference is not meaningful yet.

**Learned:**
- `@st.cache_data` for expensive computations; separate widget keys for similar widgets.
- Simulation (walk-forward) vs live performance; why both are reported.

---

### Step 7.4 — Data provenance correction and cleaning (`clean_data.py`) ✅

**Correction:** "Nipun" and "Kasun" were **not two different people**. Both data-collection sessions (and all later test rounds under other names) were played by the author. Earlier entries that describe "two players" should be read as **one player, two sessions**.

**What was done:** `clean_data.py` (run once; it refuses to run again if the backup exists):
1. Saved a backup: `data/rounds_backup_before_cleaning.csv`.
2. Removed all adaptive-AI rounds (played by the author while testing, knowing how the AI works, so they are not a fair test of the AI) and the 7 "Jhon" test rounds.
3. Relabelled the sessions honestly: Nipun → **Session 1** (251 rounds), Kasun → **Session 2** (109 rounds).

Result: 360 rows remain, all random-mode. All Stage 3–6 numbers are unchanged; only the labels changed. Chart titles no longer say "each player".

**What the results can and cannot claim now:**
- ✅ Valid: this player's moves were not random; a sequential Rock ↔ Paper pattern appeared in **both sessions** (a consistency finding); the model beat the majority baseline in Session 2 (p = 0.002).
- ❌ Not valid: any claim about people in general or about differences between players; the Step 7.3 "real games" figures (the author knew how the AI worked).

**Possible confound (to report as a limitation):** The game IDs suggest Session 1's 251 rounds were played within about 3 minutes (≈ 0.7 s per round) *(not confirmed)*. At that speed the choice may be driven by mouse movement: 🪨 Rock and 📄 Paper are neighbouring buttons and ✂️ Scissors is at the far end. The Rock ↔ Paper alternation and the low Scissors rate could therefore partly reflect the **button layout** rather than a mental strategy. Future work: randomise the button order to test this.

**Why a script instead of editing the CSV by hand:** The cleaning is documented, reproducible, and reversible (backup).

**✅ Stage 7 complete.**

**Learned:**
- Data provenance: always record who generated the data and under what conditions.
- Honest reporting means correcting the record, even late.
- Look for alternative explanations (confounds) before interpreting a pattern.

---

### Step 8.1 — `requirements.txt` and `.gitignore` ✅

**What:**
- `requirements.txt` pins the exact versions used (streamlit 1.64.0, pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1, matplotlib 3.11.2, scipy ≥ 1.11) so others can reproduce the results with `python -m pip install -r requirements.txt`. SciPy is listed explicitly because `scipy.stats` is used directly.
- `.gitignore` keeps `.venv/`, `__pycache__/`, generated text outputs, data backups and personal notes out of the repository.

**Test:** `pip install -r requirements.txt` → every line "Requirement already satisfied".

**Problem and fix:** The downloaded file was not in the project folder (`No such file or directory`), so it was created directly in VS Code.

---

### Step 8.2 — README ✅

**What:** Wrote `README.md`, the project's front page: research question, key findings table, the four charts, an app screenshot, how it works, limitations, how to run it, project structure, and future work.

**Honesty checks while writing:**
- Findings are described as one player in two sessions.
- The multiple-testing statement was checked: with all 10 tests (Bonferroni α = 0.005), the sequential pattern and all Session 2 model results stay significant, but Session 1's Paper preference (p = 0.0068) does not; the README says so.
- The fast-clicking confound is written as "appear to", because it is inferred from timestamps.

**Screenshots:** Only one app screenshot is used (the adaptive AI explaining a prediction); the charts already show the statistics and model results.

---

### Step 8.3 — Upload to GitHub ✅

**What:** Created the public repository **github.com/nipun638/mindplay-ai** and pushed the project:
```powershell
git init
git add .
git status          # checked: 19 files, no .venv
git commit -m "MindPlay AI: first complete version"
git branch -M main
git remote add origin https://github.com/nipun638/mindplay-ai.git
git push -u origin main
```

**Problems and fixes:**

| Problem | Cause | Fix |
|---|---|---|
| Placeholder email set in `git config` | Pasted the example command | Re-ran the command with the real email (it overwrites) |
| `git init.` not a git command | Typo (extra dot) | Ran `git init` |
| `images/app_game.png.png` | Windows hides extensions and added `.png` again | `git mv "images/app_game.png.png" "images/app_game.png"` before committing |
| `LF will be replaced by CRLF` warnings | Windows vs Linux line endings | Harmless; no action needed |

**Learned:**
- Git basics: init, add, status, commit, branch, remote, push.
- Always check `git status` before committing.

---

### Step 8.4 — LinkedIn post ✅

Wrote a post that uses only real numbers and states the limitations: 360 rounds of the author's own play in two sessions, chi-square tests, Naive Bayes, walk-forward accuracy 57% vs 40% for the best simple baseline (McNemar p = 0.002), and the one-player / fast-clicking limitations.

---

### Step 8.5 — Data consistency fix ✅

**Problem:** While taking the README screenshots (29 Sep), 24 extra rounds were saved under "Session 2" (3 random + 21 adaptive). Those 3 extra random rounds would have made Session 2 112 rounds instead of the 109 reported in the README and this log.

**Fix:** `clean_data.py` was rewritten so it can be run any number of times. It keeps only **random-mode rounds from the data-collection day (27 Sep 2026)**, drops everything else, and relabels the sessions. Then `visualize.py` was re-run and the new `clean_data.py` was pushed to GitHub (29 Sep). `git status` showed `data/rounds.csv` unchanged, confirming the dataset on GitHub was already the clean 360-round version.

**Result:** 360 rows (251 + 109); every number again matches the README (verified by re-running `stats.py` and `evaluate.py` on the repository data).

**Learned:**
- Using the app for demos can silently add rows to the analysis dataset.
- Cleaning rules should be reproducible (based on date and mode), not one-off manual edits.

**✅ Stage 8 complete. Project finished.**

---

## 5. Key Decisions

| Decision | Reason |
|---|---|
| Use a virtual environment | Isolates project dependencies; standard professional practice; enables `requirements.txt` |
| Start with simple ML models (Decision Tree, Naive Bayes) | Easy to understand and explain; the dataset will be small, so complex models would likely overfit |
| No deep learning | Not justified for a small dataset and a 3-class problem |
| Always compare the model with a baseline | Accuracy alone does not prove the model learned anything; random guessing gives about 33% |
| Keep Rock-Paper-Scissors (instead of a penalty-shootout game or a dataset-only project) | Simple deterministic rules, a clear 33.3% baseline, interactive, and combines statistics + ML + interactivity. A penalty-shootout version is noted as future work. |
| Strengthen the statistics with formal hypothesis tests | Turns the project into a statistical investigation of whether humans play randomly, with ML as a second test |
| Store only raw round data; derive features (e.g., previous move) later | Keeps the dataset clean and avoids duplicate information; feature creation is a separate workflow step |
| Add a `player` column and analyse players separately | Different players have different patterns; mixing them would hide individual behaviour |
| Collect data against a random computer | Controlled condition: patterns come from the player, not from reacting to an adaptive AI |
| Delete test data before real collection | The dataset should contain only natural play |
| Time-based train/test split (no shuffling) | Simulates a real game: learn from the past, predict the future |
| Final model: Naive Bayes with `prev_player_move` only | Simplest model with the best result; gives probabilities; explainable as P(next \| previous) |
| Label every round with `ai_mode`; analyse random-mode rounds only | Players may behave differently against an adaptive opponent; mixing conditions would bias the Stage 3–5 results |
| Migrate old data instead of deleting it | The 360 real rounds are valuable; all were played in random mode |
| Minimum 50 rounds per player for analysis | Chi-square expected counts and walk-forward evaluation need enough data |
| Relabel "Nipun"/"Kasun" as Session 1/2 and remove test rounds | Both were the author; honest labels prevent over-claiming |
| Finish with the existing data rather than recruit more players now | Time; the limitation (one player) is stated clearly instead |

---

## 5a. Planned Statistical Analysis (Stage 3)

**Central question:** Do humans play Rock-Paper-Scissors randomly? If not, the patterns can be used to predict their next move.

| Analysis | Question | Method |
|---|---|---|
| Move distribution | Does the player choose each move equally often (1/3 each)? | Proportions + **chi-square goodness-of-fit test** |
| Transition patterns | Does the next move depend on the previous move? | Transition matrix, conditional probabilities P(next \| previous), **chi-square test of independence** |
| Win-stay / lose-shift | Does the player repeat after a win and switch after a loss? | Conditional proportions of "stay" vs "switch" by previous result |
| Outcome rates | How often does the player win, lose, or draw? | Win / loss / draw rates |

**Background:** Research on real players (Wang, Xu & Zhou, 2014) found a tendency to repeat winning moves and switch after losing. This project tests whether the same pattern appears in our data.

**Future work idea:** Apply the same approach to penalty-kick prediction, a classic application of game theory in sports analytics (Chiappori, Groseclose & Levitt, 2002; Palacios-Huerta, 2003).

---

## 6. Concepts Learned

- **Virtual environment:** A private, isolated Python installation for one project.
- **pip:** Python's package installer.
- **Execution policy:** A Windows security setting that controls whether PowerShell scripts can run.
- **Dependency:** A library that another library needs in order to work (e.g., scikit-learn depends on SciPy and NumPy).
- **Smart App Control:** A Windows 11 feature that blocks unsigned programs and files; it can conflict with programming tools.
- **List / dictionary / function / f-string:** Core Python building blocks used in `game_logic.py`.
- **`if __name__ == "__main__":`** Code under this line runs only when the file is executed directly.
- **Session state:** Streamlit's memory that survives script re-runs.
- **DataFrame:** pandas' table structure; rows = observations, columns = variables.
- **Feature engineering:** Creating new input variables (e.g., previous move) from raw data.
- **CSV:** A plain-text table format; one line per row, values separated by commas.
- **Append mode:** Writing to the end of a file without deleting what is already there.
- **Schema:** The structure of a dataset (its columns and their types).
- **Data dictionary:** A table describing every column in a dataset.
- **Data validation:** Checking data for missing, invalid, or inconsistent values before analysis.
- **Proportion:** count ÷ total; the sample estimate of a probability.
- **p-value:** The probability of seeing data at least this extreme if H₀ were true.
- **Chi-square goodness-of-fit test:** Compares observed category counts with the counts expected under a hypothesised distribution.
- **Transition matrix:** A table of how often each state (move) is followed by each other state; rows normalised give conditional probabilities.
- **Chi-square test of independence:** Tests whether two categorical variables (here, previous move and next move) are related.
- **Lag feature:** A value from an earlier time step (e.g., previous move) used as a variable.
- **In-sample vs out-of-sample:** Performance on the data used to find a pattern vs on new data.
- **Win-stay / lose-shift:** A behavioural strategy of repeating a winning move and changing a losing one.
- **Multiple testing:** Running many tests increases the chance of false positives; a Bonferroni correction divides α by the number of tests.
- **Features / target:** The model's inputs (X) and the value it predicts (y).
- **Data leakage:** Giving the model information it would not have at prediction time, which inflates accuracy.
- **One-hot encoding:** Representing a category as several 0/1 columns, one per possible value.
- **Train/test split:** Learning on one part of the data and measuring performance on a separate, unseen part.
- **Majority (most-frequent) baseline:** Always predicting the most common class; the minimum a model must beat to show it learned something.
- **Overfitting:** A model memorises noise in the training data, so it performs well on training data but worse on new data.
- **Feature importance:** How much each feature reduced impurity in a tree's splits.
- **Naive Bayes:** A probabilistic classifier using Bayes' rule and assuming features are independent given the class.
- **Laplace smoothing:** Adding 1 to every count so no probability is exactly 0.
- **Walk-forward evaluation:** Repeatedly train on all past data and predict the next observation.
- **Binomial test:** Tests whether a success proportion (e.g., accuracy) differs from a hypothesised value.
- **McNemar test:** Compares two classifiers on the same cases, using only the cases where they disagree.
- **Confidence interval:** A range of plausible values for a population quantity (here, true accuracy).
- **Confusion matrix:** A table of actual vs predicted classes; the diagonal holds correct predictions.
- **Precision / recall / F1:** Precision = correct among predicted; recall = correct among actual; F1 = their harmonic mean.
- **Cold start:** A model cannot predict for a new user until it has seen some of their data.
- **Schema migration:** Updating existing data to a new structure without losing it.
- **Data provenance:** The record of where data came from and how it was produced.
- **Confound:** An alternative factor (e.g., button layout) that could explain an observed pattern.

---

## 7. Open Questions and Ideas for Later

- ~~How many rounds of data are needed before the ML model is meaningful?~~ → Depends on pattern strength: with a strong pattern (Kasun), ~100 rounds and walk-forward evaluation were enough to show a significant gain; with a weaker one (Nipun), 250 rounds were not.
- Would players change their behaviour against an adaptive AI? (to test in Stage 7)
- ~~Should data from different players be kept separate?~~ → Yes; a `player` column was added in Step 2.3.
