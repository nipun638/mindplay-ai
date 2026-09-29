"""A simple terminal version of Rock-Paper-Scissors (Stage 1 prototype)."""

from game_logic import get_random_move, determine_result

# Shortcut keys, so the player can type one letter
SHORTCUTS = {"r": "Rock", "p": "Paper", "s": "Scissors"}


def play():
    """Run the game loop until the player quits."""
    scores = {"Win": 0, "Loss": 0, "Draw": 0}
    round_number = 0

    print("=== MindPlay AI: Rock, Paper, Scissors ===")
    print("Type r = Rock, p = Paper, s = Scissors, n = new game, q = quit\n")

    while True:
        choice = input("Your move: ").strip().lower()

        if choice == "q":
            break

        if choice == "n":
            scores = {"Win": 0, "Loss": 0, "Draw": 0}
            round_number = 0
            print("--- New game started. Scores reset. ---\n")
            continue

        if choice not in SHORTCUTS:
            print("Please type r, p, s, n, or q.\n")
            continue

        round_number += 1
        player_move = SHORTCUTS[choice]
        computer_move = get_random_move()
        result = determine_result(player_move, computer_move)
        scores[result] += 1

        print(f"Round {round_number}: You played {player_move}, "
              f"computer played {computer_move} -> {result}")
        print(f"Score -> Wins: {scores['Win']}, Losses: {scores['Loss']}, "
              f"Draws: {scores['Draw']}\n")

    # Summary when the player quits
    print(f"\nGame over after {round_number} rounds.")
    if round_number > 0:
        win_rate = scores["Win"] / round_number
        print(f"Your win rate: {win_rate:.1%}")


if __name__ == "__main__":
    play()