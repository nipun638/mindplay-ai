"""Statistical analysis of MindPlay AI gameplay data (Stage 3)."""

from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency, chisquare

from game_logic import MOVES, determine_result

DATA_FILE = Path(__file__).parent / "data" / "rounds.csv"
RESULTS = ["Win", "Loss", "Draw"]
# Columns every round must have (ai_prediction is empty in random mode, so it is not checked)
CORE_COLUMNS = ["player", "game_id", "round", "player_move", "computer_move", "result"]
# Players with fewer rounds are left out of the analysis: the transition test needs about
# 45+ transitions (expected count >= 5 in each of 9 cells), and walk-forward needs 20+ to start.
MIN_ROUNDS = 50


def format_p(p_value):
    """Show very small p-values as '< 0.0001' instead of rounding them to 0."""
    return "< 0.0001" if p_value < 0.0001 else f"{p_value:.4f}"


def load_data(path=DATA_FILE, ai_mode="random"):
    """Read the saved rounds into a DataFrame.

    By default only rounds played against the RANDOM computer are returned, because the
    Stage 3-6 analysis is about how people play when the opponent does not adapt.
    Use ai_mode=None to get every round.
    """
    df = pd.read_csv(path)
    if ai_mode is not None and "ai_mode" in df.columns:
        df = df[df["ai_mode"] == ai_mode]
    return df


def keep_players_with_enough_data(df, min_rounds=MIN_ROUNDS):
    """Return (data for players with at least min_rounds rounds, counts of skipped players)."""
    counts = df["player"].value_counts()
    keep = counts[counts >= min_rounds].index
    skipped = counts[counts < min_rounds]
    return df[df["player"].isin(keep)], skipped


def print_skipped(skipped):
    """Tell the user which players were left out and why."""
    if len(skipped) > 0:
        names = ", ".join(f"{name} ({count} rounds)" for name, count in skipped.items())
        print(f"Skipped (fewer than {MIN_ROUNDS} rounds): {names}\n")


def check_data(df):
    """Return a list of data problems. An empty list means the data is clean."""
    problems = []

    missing = int(df[CORE_COLUMNS].isna().sum().sum())
    if missing > 0:
        problems.append(f"{missing} missing values")

    for column in ["player_move", "computer_move"]:
        invalid = int((~df[column].isin(MOVES)).sum())
        if invalid > 0:
            problems.append(f"{invalid} invalid values in '{column}'")

    # Only re-check results if all moves are valid (otherwise the rules can't be applied)
    if not problems:
        expected = [
            determine_result(p, c)
            for p, c in zip(df["player_move"], df["computer_move"])
        ]
        wrong = int((df["result"] != pd.Series(expected, index=df.index)).sum())
        if wrong > 0:
            problems.append(f"{wrong} rows with an incorrect result")

    return problems


def move_summary(df):
    """Count and proportion of each player move."""
    counts = df["player_move"].value_counts().reindex(MOVES, fill_value=0)
    return pd.DataFrame({
        "count": counts,
        "proportion": (counts / counts.sum()).round(3),
    })


def outcome_rates(df):
    """Proportion of wins, losses, and draws."""
    return (
        df["result"]
        .value_counts(normalize=True)
        .reindex(RESULTS, fill_value=0)
        .round(3)
    )


def uniformity_test(df):
    """Chi-square goodness-of-fit test. H0: Rock, Paper, Scissors are equally likely."""
    observed = move_summary(df)["count"]
    n = observed.sum()
    expected = pd.Series(n / 3, index=MOVES)

    # Calculate the statistic by hand, to understand (and double-check) the formula
    chi2_manual = (((observed - expected) ** 2) / expected).sum()

    # SciPy calculates the same statistic plus the p-value
    result = chisquare(observed)

    return {
        "n": int(n),
        "expected_each": round(n / 3, 1),
        "chi2_manual": round(float(chi2_manual), 3),
        "chi2": round(float(result.statistic), 3),
        "p_value": float(result.pvalue),
    }


def add_previous_moves(df):
    """Feature engineering: add the previous round's moves and result (within the same game)."""
    df = df.sort_values(["player", "game_id", "round"]).copy()
    groups = df.groupby(["player", "game_id"])
    df["prev_player_move"] = groups["player_move"].shift(1)
    df["prev_computer_move"] = groups["computer_move"].shift(1)
    df["prev_result"] = groups["result"].shift(1)
    return df


def transition_counts(df):
    """3 x 3 table: rows = previous move, columns = next move."""
    df = add_previous_moves(df).dropna(subset=["prev_player_move"])
    table = pd.crosstab(
        df["prev_player_move"], df["player_move"],
        rownames=["previous"], colnames=["next"],
    )
    return table.reindex(index=MOVES, columns=MOVES, fill_value=0)


def transition_probabilities(df):
    """Conditional probabilities P(next move | previous move). Each row adds up to 1."""
    counts = transition_counts(df)
    return counts.div(counts.sum(axis=1), axis=0).round(3)


def repeat_rate(df):
    """Proportion of rounds where the player repeated their previous move (random = 1/3)."""
    counts = transition_counts(df)
    return round(float(counts.values.diagonal().sum() / counts.values.sum()), 3)


def independence_test(df):
    """Chi-square test of independence. H0: the next move does not depend on the previous move."""
    counts = transition_counts(df)
    result = chi2_contingency(counts)
    return {
        "n": int(counts.values.sum()),
        "chi2": round(float(result.statistic), 3),
        "dof": int(result.dof),
        "p_value": float(result.pvalue),
        "min_expected": round(float(result.expected_freq.min()), 1),
    }


def stay_shift_table(df):
    """Counts of Stay (repeat previous move) vs Shift (change move), by previous result."""
    df = add_previous_moves(df).dropna(subset=["prev_player_move"])
    choice = (df["player_move"] == df["prev_player_move"]).map({True: "Stay", False: "Shift"})
    table = pd.crosstab(
        df["prev_result"], choice,
        rownames=["previous result"], colnames=["choice"],
    )
    table = table.reindex(index=RESULTS, columns=["Stay", "Shift"], fill_value=0)
    table["stay_rate"] = (table["Stay"] / (table["Stay"] + table["Shift"])).round(3)
    return table


def stay_shift_test(df):
    """Chi-square test. H0: the chance of staying does not depend on the previous result."""
    counts = stay_shift_table(df)[["Stay", "Shift"]]
    result = chi2_contingency(counts)
    return {
        "n": int(counts.values.sum()),
        "chi2": round(float(result.statistic), 3),
        "dof": int(result.dof),
        "p_value": float(result.pvalue),
        "min_expected": round(float(result.expected_freq.min()), 1),
    }


if __name__ == "__main__":
    df = load_data()

    print("=== Dataset overview ===")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(df.head(), "\n")

    problems = check_data(df)
    print("Data check:", "OK, no problems found" if not problems else problems, "\n")

    print("=== Rounds per player ===")
    print(df["player"].value_counts(), "\n")

    df, skipped = keep_players_with_enough_data(df)
    print_skipped(skipped)

    for player, player_df in df.groupby("player"):
        print(f"========== {player} ({len(player_df)} rounds) ==========")
        print("Move distribution (random would be 0.333 each):")
        print(move_summary(player_df), "\n")
        print("Outcome rates:")
        print(outcome_rates(player_df), "\n")

        test = uniformity_test(player_df)
        print("Chi-square goodness-of-fit test (H0: all moves equally likely):")
        print(f"  n = {test['n']}, expected per move = {test['expected_each']}")
        print(f"  chi2 = {test['chi2']} (manual: {test['chi2_manual']}), df = 2, p-value = {format_p(test['p_value'])}")
        if test["p_value"] < 0.05:
            print("  Conclusion (alpha = 0.05): Reject H0. The moves are NOT equally likely.")
        else:
            print("  Conclusion (alpha = 0.05): Do not reject H0. No evidence of a move preference.")
        print()

        print("Transition counts (rows = previous move, columns = next move):")
        print(transition_counts(player_df), "\n")
        print("Transition probabilities P(next | previous) (random would be 0.333 each):")
        print(transition_probabilities(player_df), "\n")
        print(f"Repeat rate (same move as previous round): {repeat_rate(player_df)} (random: 0.333)\n")

        ind = independence_test(player_df)
        print("Chi-square test of independence (H0: next move does not depend on previous move):")
        print(f"  n = {ind['n']} transitions, chi2 = {ind['chi2']}, df = {ind['dof']}, p-value = {format_p(ind['p_value'])}")
        print(f"  Smallest expected count = {ind['min_expected']} (should be at least 5)")
        if ind["p_value"] < 0.05:
            print("  Conclusion (alpha = 0.05): Reject H0. The next move DEPENDS on the previous move.")
        else:
            print("  Conclusion (alpha = 0.05): Do not reject H0. No evidence that the next move depends on the previous move.")
        print()

        print("Win-stay / lose-shift (Stay = repeat previous move; random stay_rate would be 0.333):")
        print(stay_shift_table(player_df), "\n")

        ws = stay_shift_test(player_df)
        print("Chi-square test (H0: staying does not depend on the previous result):")
        print(f"  n = {ws['n']}, chi2 = {ws['chi2']}, df = {ws['dof']}, p-value = {format_p(ws['p_value'])}")
        print(f"  Smallest expected count = {ws['min_expected']} (should be at least 5)")
        if ws["p_value"] < 0.05:
            print("  Conclusion (alpha = 0.05): Reject H0. Staying/shifting DEPENDS on the previous result.")
        else:
            print("  Conclusion (alpha = 0.05): Do not reject H0. No evidence that the previous result affects staying.")
        print()
