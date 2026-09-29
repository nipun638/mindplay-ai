"""The adaptive AI opponent for MindPlay AI (Stage 7)."""

import random

import pandas as pd

from evaluate import COUNTER, WARMUP
from game_logic import MOVES
from model import build_dataset, train_naive_bayes

FEATURES_USED = ["prev_player_move"]


def predict_next_move(player_history, last_move):
    """Predict the player's next move from their past rounds.

    Returns (predicted_move, probability), or (None, None) when there is not enough data yet
    or no previous move in the current game.
    """
    if last_move is None or player_history.empty:
        return None, None

    X, y = build_dataset(player_history, FEATURES_USED, encoding="codes")
    if len(X) < WARMUP:
        return None, None

    model = train_naive_bayes(X, y, FEATURES_USED)
    x_now = pd.DataFrame({"prev_player_move": [MOVES.index(last_move)]})
    probabilities = model.predict_proba(x_now)[0]
    best = probabilities.argmax()
    return str(model.classes_[best]), float(probabilities[best])


def choose_ai_move(player_history, last_move):
    """Play the move that beats the predicted move; play randomly if there is no prediction.

    Note: this only uses PAST rounds. It never looks at the move the player is choosing now.
    """
    prediction, probability = predict_next_move(player_history, last_move)
    if prediction is None:
        return random.choice(MOVES), None, None
    return COUNTER[prediction], prediction, probability
