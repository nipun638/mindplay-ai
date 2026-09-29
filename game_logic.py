"""Core Rock-Paper-Scissors game rules for MindPlay AI."""

import random

# The three possible moves
MOVES = ["Rock", "Paper", "Scissors"]

# Each key BEATS its value: Rock beats Scissors, Paper beats Rock, Scissors beats Paper
BEATS = {
    "Rock": "Scissors",
    "Paper": "Rock",
    "Scissors": "Paper",
}


def get_random_move():
    """Return a random move for the computer."""
    return random.choice(MOVES)


def determine_result(player_move, computer_move):
    """Return 'Win', 'Loss', or 'Draw' from the player's point of view."""
    if player_move == computer_move:
        return "Draw"
    if BEATS[player_move] == computer_move:
        return "Win"
    return "Loss"


if __name__ == "__main__":
    # Quick self-test: check every possible combination of moves
    for player in MOVES:
        for computer in MOVES:
            result = determine_result(player, computer)
            print(f"{player:8} vs {computer:8} -> {result}")