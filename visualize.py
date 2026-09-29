"""Charts for MindPlay AI (Stage 6). Saves PNG images into the 'images' folder."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw straight to files, no pop-up windows
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

from scipy.stats import binomtest

from evaluate import confusion_table, walk_forward
from stats import (MOVES, format_p, keep_players_with_enough_data, load_data, move_summary,
                   print_skipped, transition_probabilities)

IMAGES = Path(__file__).parent / "images"

# Colours (validated colour-blind-safe pair for the two players)
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#8a8984"
PLAYER_COLORS = ["#2a78d6", "#eb6834"]  # player 1 blue, player 2 orange
MODEL_COLOR = "#4a3aa7"      # violet: the Naive Bayes model
BASELINE_COLOR = "#b8b7b1"   # neutral grey: the majority baseline
# One-hue blue ramp (light = low, dark = high) for heatmaps
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


def text_color_for(cell_rgba):
    """White text on dark cells, dark text on light cells (based on brightness)."""
    r, g, b = cell_rgba[:3]
    brightness = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if brightness < 0.5 else TEXT


def style_axes(ax):
    """Quiet axes: light grid, no top/right border, muted tick labels."""
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED, labelsize=10)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_move_distribution(df, path):
    """Grouped bar chart: proportion of each move per player, with the random line at 1/3."""
    players = sorted(df["player"].unique())
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    style_axes(ax)

    bar_width = 0.36
    for i, player in enumerate(players):
        summary = move_summary(df[df["player"] == player])
        n = int(summary["count"].sum())
        positions = [x + (i - 0.5) * (bar_width + 0.02) for x in range(len(MOVES))]
        bars = ax.bar(positions, summary["proportion"], width=bar_width,
                      color=PLAYER_COLORS[i], label=f"{player} ({n} rounds)")
        for bar, value in zip(bars, summary["proportion"]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.0%}",
                    ha="center", va="bottom", fontsize=9, color=TEXT_MUTED)

    ax.axhline(1 / 3, color=REFERENCE, linestyle="--", linewidth=1.2)
    ax.text(len(MOVES) - 0.45, 1 / 3 + 0.01, "random (1/3)", color=TEXT_MUTED,
            fontsize=9, ha="right", va="bottom")

    ax.set_xticks(range(len(MOVES)), MOVES)
    ax.set_ylabel("Proportion of moves", color=TEXT_MUTED)
    ax.set_ylim(0, 0.55)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("How often is each move chosen?", color=TEXT,
                 fontsize=13, loc="left", pad=12)
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=TEXT_MUTED)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_transition_heatmaps(df, path):
    """One heatmap per player: P(next move | previous move)."""
    players = sorted(df["player"].unique())
    fig, axes = plt.subplots(1, len(players), figsize=(5.2 * len(players), 4.6), facecolor=SURFACE)
    fig.subplots_adjust(wspace=0.35)
    scale = Normalize(vmin=0, vmax=0.7)

    for i, (ax, player) in enumerate(zip(axes, players)):
        probs = transition_probabilities(df[df["player"] == player])
        image = ax.imshow(probs.values, cmap=BLUE_RAMP, norm=scale)

        for r in range(len(MOVES)):
            for c in range(len(MOVES)):
                value = probs.values[r, c]
                ax.text(c, r, f"{value:.0%}", ha="center", va="center", fontsize=12,
                        color=text_color_for(BLUE_RAMP(scale(value))))

        ax.set_xticks(range(len(MOVES)), MOVES)
        ax.set_yticks(range(len(MOVES)), MOVES)
        ax.set_xlabel("Next move", color=TEXT_MUTED)
        if i == 0:
            ax.set_ylabel("Previous move", color=TEXT_MUTED)
        ax.tick_params(colors=TEXT_MUTED, length=0)
        ax.set_title(player, color=TEXT, fontsize=12, loc="left")
        for spine in ax.spines.values():
            spine.set_visible(False)

    colorbar = fig.colorbar(image, ax=axes, shrink=0.8, format=lambda v, _: f"{v:.0%}")
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=TEXT_MUTED)
    fig.suptitle("What is played next?  P(next move | previous move)",
                 color=TEXT, fontsize=13, x=0.02, ha="left")
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def accuracy_with_ci(correct, n):
    """Accuracy plus an exact 95% confidence interval."""
    ci = binomtest(correct, n).proportion_ci(confidence_level=0.95)
    return correct / n, ci.low, ci.high


def plot_model_accuracy(results, path):
    """Grouped bars: majority baseline vs Naive Bayes (walk-forward), with 95% CIs."""
    players = list(results)
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
    style_axes(ax)

    bar_width = 0.34
    models = [("majority", "Majority baseline (most frequent so far)", BASELINE_COLOR),
              ("naive_bayes", "Naive Bayes (previous move)", MODEL_COLOR)]
    for j, (column, label, color) in enumerate(models):
        for i, player in enumerate(players):
            preds = results[player]
            n = len(preds)
            correct = int((preds[column] == preds["actual"]).sum())
            acc, low, high = accuracy_with_ci(correct, n)
            x = i + (j - 0.5) * (bar_width + 0.02)
            ax.bar(x, acc, width=bar_width, color=color, label=label if i == 0 else None)
            ax.errorbar(x, acc, yerr=[[acc - low], [high - acc]], color=TEXT_MUTED,
                        capsize=4, linewidth=1.2)
            ax.text(x, 0.02, f"{acc:.0%}\n{correct}/{n}", ha="center", va="bottom",
                    fontsize=9, color="white" if color == MODEL_COLOR else TEXT)

    # McNemar p-value (Naive Bayes vs majority) above each player's pair of bars
    for i, player in enumerate(players):
        preds = results[player]
        nb_right = preds["naive_bayes"] == preds["actual"]
        maj_right = preds["majority"] == preds["actual"]
        b, c = int((nb_right & ~maj_right).sum()), int((maj_right & ~nb_right).sum())
        p = binomtest(b, b + c, 0.5, alternative="greater").pvalue if b + c else 1.0
        p_text = format_p(p)
        p_text = f"p {p_text}" if p_text.startswith("<") else f"p = {p_text}"
        ax.text(i, 0.86, f"NB vs baseline: {p_text}", ha="center", fontsize=9,
                color=TEXT, fontweight="bold" if p < 0.05 else "normal")

    ax.axhline(1 / 3, color=REFERENCE, linestyle="--", linewidth=1.2, label="Random guess (1/3)")

    ax.set_xticks(range(len(players)), [f"{p}\n({len(results[p])} predictions)" for p in players])
    ax.set_ylabel("Walk-forward accuracy", color=TEXT_MUTED)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Can the model predict the next move better than simple baselines?",
                 color=TEXT, fontsize=13, loc="left", pad=12)
    ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=8.5, labelcolor=TEXT_MUTED,
              bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.01, "Error bars: exact 95% confidence intervals. "
             "p-values: one-sided exact McNemar test.", fontsize=8, color=TEXT_MUTED)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_confusion_matrices(results, path):
    """One heatmap per player: actual move vs predicted move (colour = share of each actual move)."""
    players = list(results)
    fig, axes = plt.subplots(1, len(players), figsize=(5.2 * len(players), 4.6), facecolor=SURFACE)
    fig.subplots_adjust(wspace=0.35)
    scale = Normalize(vmin=0, vmax=1)

    for i, (ax, player) in enumerate(zip(axes, players)):
        preds = results[player]
        counts = confusion_table(preds["actual"], preds["naive_bayes"])
        shares = counts.div(counts.sum(axis=1), axis=0)
        image = ax.imshow(shares.values, cmap=BLUE_RAMP, norm=scale)

        for r in range(len(MOVES)):
            for c in range(len(MOVES)):
                ax.text(c, r, f"{counts.values[r, c]}\n({shares.values[r, c]:.0%})",
                        ha="center", va="center", fontsize=11,
                        color=text_color_for(BLUE_RAMP(scale(shares.values[r, c]))))

        ax.set_xticks(range(len(MOVES)), MOVES)
        ax.set_yticks(range(len(MOVES)), MOVES)
        ax.set_xlabel("Predicted move", color=TEXT_MUTED)
        if i == 0:
            ax.set_ylabel("Actual move", color=TEXT_MUTED)
        ax.tick_params(colors=TEXT_MUTED, length=0)
        accuracy = (preds["naive_bayes"] == preds["actual"]).mean()
        ax.set_title(f"{player}  (accuracy {accuracy:.0%})", color=TEXT, fontsize=12, loc="left")
        for spine in ax.spines.values():
            spine.set_visible(False)

    colorbar = fig.colorbar(image, ax=axes, shrink=0.8, format=lambda v, _: f"{v:.0%}")
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(colors=TEXT_MUTED)
    fig.suptitle("Where is the model right and wrong?  (Naive Bayes, walk-forward)",
                 color=TEXT, fontsize=13, x=0.02, ha="left")
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    IMAGES.mkdir(exist_ok=True)
    df, skipped = keep_players_with_enough_data(load_data())
    print_skipped(skipped)

    # Stage 3 charts
    plot_move_distribution(df, IMAGES / "move_distribution.png")
    plot_transition_heatmaps(df, IMAGES / "transition_matrix.png")

    # Stage 5 charts (walk-forward predictions, computed once per player)
    results = {player: walk_forward(player_df)
               for player, player_df in df.groupby("player")}
    plot_model_accuracy(results, IMAGES / "model_accuracy.png")
    plot_confusion_matrices(results, IMAGES / "confusion_matrix.png")

    print("Saved charts:")
    for file in sorted(IMAGES.glob("*.png")):
        print(" ", file.name)
