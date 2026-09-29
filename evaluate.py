"""Model evaluation for MindPlay AI (Stage 5): walk-forward evaluation."""

import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import classification_report, confusion_matrix

from game_logic import BEATS, MOVES, determine_result
from model import build_dataset, train_naive_bayes
from stats import format_p, keep_players_with_enough_data, load_data, print_skipped

WARMUP = 20  # the model needs some past rounds before its first prediction
FEATURES_USED = ["prev_player_move"]

# The move that beats each move (e.g. COUNTER["Rock"] = "Paper")
COUNTER = {loser: winner for winner, loser in BEATS.items()}


def walk_forward(player_df, warmup=WARMUP):
    """For every round after the warm-up: train on ALL earlier rounds, then predict this round."""
    X, y = build_dataset(player_df, FEATURES_USED, encoding="codes")
    rows = []
    for t in range(warmup, len(X)):
        X_past, y_past = X.iloc[:t], y.iloc[:t]

        model = train_naive_bayes(X_past, y_past, FEATURES_USED)
        nb_prediction = model.predict(X.iloc[[t]])[0]
        majority_prediction = y_past.value_counts().idxmax()

        rows.append({
            "actual": y.iloc[t],
            "naive_bayes": nb_prediction,
            "majority": majority_prediction,
        })
    return pd.DataFrame(rows)


def confusion_table(actual, predicted):
    """Rows = the player's real move, columns = the model's prediction."""
    matrix = confusion_matrix(actual, predicted, labels=MOVES)
    return pd.DataFrame(
        matrix,
        index=pd.Index(MOVES, name="actual"),
        columns=pd.Index(MOVES, name="predicted"),
    )


def ai_outcomes(actual, predicted):
    """If the AI played the move that BEATS each prediction, how would the AI have done?"""
    results = [
        determine_result(COUNTER[p], a)  # from the AI's point of view
        for p, a in zip(predicted, actual)
    ]
    return pd.Series(results).value_counts(normalize=True).reindex(
        ["Win", "Loss", "Draw"], fill_value=0
    )


if __name__ == "__main__":
    df, skipped = keep_players_with_enough_data(load_data())
    print_skipped(skipped)

    for player, player_df in df.groupby("player"):
        preds = walk_forward(player_df)
        n = len(preds)
        nb_correct = int((preds["naive_bayes"] == preds["actual"]).sum())
        maj_correct = int((preds["majority"] == preds["actual"]).sum())

        print(f"========== {player} ==========")
        print(f"Walk-forward predictions: {n} (the first {WARMUP} examples are only used for learning)\n")

        print(f"  {'Model':<34}{'Accuracy':>9}{'Correct':>11}")
        print(f"  {'Random guess (expected)':<34}{1/3:>9.3f}{round(n / 3):>7}/{n}")
        print(f"  {'Majority so far':<34}{maj_correct / n:>9.3f}{maj_correct:>7}/{n}")
        print(f"  {'Naive Bayes (prev move only)':<34}{nb_correct / n:>9.3f}{nb_correct:>7}/{n}\n")

        # Test 1: is Naive Bayes better than random guessing?
        vs_random = binomtest(nb_correct, n, p=1/3, alternative="greater")
        print("Test 1: Naive Bayes vs random guessing (one-sided binomial test)")
        print(f"  H0: accuracy = 1/3.  p-value = {format_p(vs_random.pvalue)}")

        # Test 2: is Naive Bayes better than the majority baseline? (exact McNemar test)
        nb_only = int(((preds["naive_bayes"] == preds["actual"]) & (preds["majority"] != preds["actual"])).sum())
        maj_only = int(((preds["majority"] == preds["actual"]) & (preds["naive_bayes"] != preds["actual"])).sum())
        print("Test 2: Naive Bayes vs majority baseline (exact McNemar test)")
        print(f"  Rounds only Naive Bayes got right: {nb_only}")
        print(f"  Rounds only the majority baseline got right: {maj_only}")
        if nb_only + maj_only > 0:
            vs_majority = binomtest(nb_only, nb_only + maj_only, p=0.5, alternative="greater")
            print(f"  H0: both are equally good.  p-value = {format_p(vs_majority.pvalue)}")
        else:
            print("  The two models made identical predictions; no test possible.")
        print()

        print("If the AI had played the move that beats each Naive Bayes prediction:")
        outcomes = ai_outcomes(preds["actual"], preds["naive_bayes"]).round(3)
        print(f"  AI win {outcomes['Win']}, AI loss {outcomes['Loss']}, draw {outcomes['Draw']}"
              "   (a random AI: 0.333 each)\n")

        print("Confusion matrix (Naive Bayes, walk-forward):")
        print(confusion_table(preds["actual"], preds["naive_bayes"]), "\n")
        print("Classification report (Naive Bayes, walk-forward):")
        print(classification_report(
            preds["actual"], preds["naive_bayes"], labels=MOVES, digits=3, zero_division=0
        ))
