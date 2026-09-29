"""Machine learning for MindPlay AI: predict the player's next move (Stage 4)."""

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import CategoricalNB
from sklearn.tree import DecisionTreeClassifier, export_text

from stats import (MOVES, RESULTS, add_previous_moves, keep_players_with_enough_data, load_data,
                   print_skipped)

# Inputs the model can see (all known BEFORE the player chooses the next move)
FEATURES = ["prev_player_move", "prev_computer_move", "prev_result"]
# What the model predicts
TARGET = "player_move"

CATEGORIES = {
    "prev_player_move": MOVES,
    "prev_computer_move": MOVES,
    "prev_result": RESULTS,
}

# Two feature sets to compare
FEATURE_SETS = {
    "prev move only": ["prev_player_move"],
    "all 3 features": FEATURES,
}


def build_dataset(player_df, features=FEATURES, encoding="onehot"):
    """Turn one player's rounds into a feature table X and a target y.

    encoding="onehot": one 0/1 column per category (for the Decision Tree)
    encoding="codes":  one integer column per feature, 0/1/2 (for Naive Bayes)
    """
    df = add_previous_moves(player_df).dropna(subset=FEATURES)

    X_raw = df[features].copy()
    for column in features:
        X_raw[column] = pd.Categorical(X_raw[column], categories=CATEGORIES[column])

    if encoding == "onehot":
        X = pd.get_dummies(X_raw, dtype=int)
    else:
        X = X_raw.apply(lambda column: column.cat.codes)
    y = df[TARGET]
    return X, y


def time_split(X, y, test_fraction=0.2):
    """Train on the EARLIER rounds, test on the LATER rounds (no shuffling)."""
    n_test = round(len(X) * test_fraction)
    X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
    y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]
    return X_train, X_test, y_train, y_test


def train_majority_baseline(X_train, y_train):
    """Baseline: always predict the move the player chose most often in training."""
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train, max_depth=3):
    """Decision Tree: learns simple if-then rules. max_depth limits how complex the rules can get."""
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_naive_bayes(X_train, y_train, features):
    """Naive Bayes: uses conditional probabilities P(feature | next move) and Bayes' rule."""
    model = CategoricalNB(min_categories=[len(CATEGORIES[f]) for f in features])
    model.fit(X_train, y_train)
    return model


def accuracies(model, X_train, X_test, y_train, y_test):
    """Accuracy on the training rounds and on the unseen test rounds."""
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    return round(train_acc, 3), round(test_acc, 3)


def print_row(name, features_label, train_acc, test_acc, n_test):
    correct = round(test_acc * n_test)
    print(f"  {name:<20}{features_label:<18}{train_acc:>7.3f}{test_acc:>7.3f}{correct:>5}/{n_test}")


if __name__ == "__main__":
    df, skipped = keep_players_with_enough_data(load_data())
    print_skipped(skipped)

    for player, player_df in df.groupby("player"):
        print(f"========== {player} ==========")

        header_done = False
        for label, features in FEATURE_SETS.items():
            X_hot, y = build_dataset(player_df, features, encoding="onehot")
            X_codes, _ = build_dataset(player_df, features, encoding="codes")
            Xh_train, Xh_test, y_train, y_test = time_split(X_hot, y)
            Xc_train, Xc_test, _, _ = time_split(X_codes, y)
            n_test = len(y_test)

            if not header_done:
                print(f"Examples: {len(y)} (train {len(y_train)}, test {n_test})\n")
                print(f"  {'Model':<20}{'Features':<18}{'Train':>7}{'Test':>7}{'Correct':>9}")
                baseline = train_majority_baseline(Xh_train, y_train)
                majority_move = baseline.predict(Xh_test.iloc[:1])[0]
                print_row(f"Always {majority_move}", "none",
                          *accuracies(baseline, Xh_train, Xh_test, y_train, y_test), n_test)
                header_done = True

            tree = train_decision_tree(Xh_train, y_train)
            print_row("Decision Tree", label,
                      *accuracies(tree, Xh_train, Xh_test, y_train, y_test), n_test)

            nb = train_naive_bayes(Xc_train, y_train, features)
            print_row("Naive Bayes", label,
                      *accuracies(nb, Xc_train, Xc_test, y_train, y_test), n_test)

        # What does Naive Bayes predict after each previous move? (prev move only)
        X_codes, y = build_dataset(player_df, ["prev_player_move"], encoding="codes")
        Xc_train, _, y_train, _ = time_split(X_codes, y)
        nb = train_naive_bayes(Xc_train, y_train, ["prev_player_move"])
        grid = pd.DataFrame({"prev_player_move": range(len(MOVES))})
        probs = pd.DataFrame(nb.predict_proba(grid), index=MOVES, columns=nb.classes_)
        probs = probs[MOVES].round(3)
        probs["prediction"] = nb.predict(grid)
        probs.index.name = "previous move"
        print("\nNaive Bayes (prev move only): P(next move | previous move) and prediction")
        print(probs, "\n")
