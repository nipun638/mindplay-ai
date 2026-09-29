"""MindPlay AI — Streamlit app (Stage 7: adaptive AI)."""

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from scipy.stats import binomtest

from ai import choose_ai_move, predict_next_move
from evaluate import COUNTER, WARMUP, ai_outcomes, confusion_table, walk_forward
from game_logic import MOVES, determine_result, get_random_move
from stats import (MIN_ROUNDS, RESULTS, format_p, independence_test, load_data, move_summary,
                   outcome_rates, repeat_rate, stay_shift_table, stay_shift_test,
                   transition_probabilities, uniformity_test)
from visualize import BLUE_RAMP

st.set_page_config(page_title="MindPlay AI", page_icon="🎮")

EMOJIS = {"Rock": "🪨", "Paper": "📄", "Scissors": "✂️"}
MODES = {"Random computer": "random", "Adaptive AI": "adaptive"}

# Where the dataset is saved (a "data" folder next to app.py)
DATA_FILE = Path(__file__).parent / "data" / "rounds.csv"


def upgrade_data_file():
    """One-time fix for the schema change: older files have no ai_mode / ai_prediction columns.

    All older rounds were played against the random computer, so they get ai_mode = "random".
    """
    if DATA_FILE.exists():
        data = pd.read_csv(DATA_FILE)
        if "ai_mode" not in data.columns:
            data["ai_mode"] = "random"
            data["ai_prediction"] = ""
            data.to_csv(DATA_FILE, index=False)


def load_player_history(player):
    """All saved rounds for one player (every mode), used to train the AI."""
    if not DATA_FILE.exists():
        return pd.DataFrame()
    data = pd.read_csv(DATA_FILE)
    return data[data["player"] == player]


def save_round(row):
    """Append one round to the CSV file (create the file if needed)."""
    DATA_FILE.parent.mkdir(exist_ok=True)
    file_exists = DATA_FILE.exists()
    pd.DataFrame([row]).to_csv(
        DATA_FILE, mode="a", header=not file_exists, index=False
    )


def reset_game():
    """Start a new game: reset scores and give it a new game ID."""
    st.session_state.scores = {"Win": 0, "Loss": 0, "Draw": 0}
    st.session_state.round_number = 0
    st.session_state.last_round = None
    st.session_state.history = []
    st.session_state.game_id = datetime.now().strftime("%Y%m%d-%H%M%S")


def clean_name(name):
    """Tidy a player name so 'prabod ' and 'Prabod' count as the same player."""
    return name.strip().title()


# ---------- Set up once ----------
upgrade_data_file()
if "scores" not in st.session_state:
    reset_game()


def play_round(player_move):
    """Play one round, update the scores, and record the round."""
    player = clean_name(st.session_state.player_name)
    ai_mode = MODES[st.session_state.mode_label]

    prediction, probability = None, None
    if ai_mode == "adaptive":
        # The AI decides using only PAST rounds: it never looks at player_move.
        last_move = st.session_state.history[-1]["player_move"] if st.session_state.history else None
        computer_move, prediction, probability = choose_ai_move(load_player_history(player), last_move)
    else:
        computer_move = get_random_move()

    result = determine_result(player_move, computer_move)
    st.session_state.round_number += 1
    st.session_state.scores[result] += 1
    st.session_state.last_round = {
        "player": player_move,
        "computer": computer_move,
        "result": result,
        "prediction": prediction,
        "probability": probability,
    }
    row = {
        "player": player,
        "game_id": st.session_state.game_id,
        "round": st.session_state.round_number,
        "player_move": player_move,
        "computer_move": computer_move,
        "result": result,
        "ai_mode": ai_mode,
        "ai_prediction": prediction or "",
    }
    st.session_state.history.append(row)
    save_round(row)


# ---------- The sections of the page ----------
def render_game():
    """The game: name, opponent, move buttons, result, score and round history."""
    # Player name and opponent (changing either starts a new game)
    st.text_input("Player name", key="player_name", on_change=reset_game)
    st.radio("Opponent", list(MODES), key="mode_label", horizontal=True, on_change=reset_game)
    if not clean_name(st.session_state.player_name):
        st.info("Enter your name to start playing.")
        return

    if MODES[st.session_state.mode_label] == "adaptive":
        st.caption(
            f"The AI plays randomly until it has seen at least {WARMUP} of your moves. "
            "Then it predicts your next move from your previous move and plays the move that beats it."
        )

    st.subheader("Choose your move")
    columns = st.columns(3)
    for column, move in zip(columns, MOVES):
        column.button(
            f"{EMOJIS[move]} {move}",
            on_click=play_round,
            args=(move,),
            width="stretch",
        )

    # Result of the most recent round
    last = st.session_state.last_round
    if last is None:
        st.info("Click a move to start playing.")
    else:
        st.subheader(f"Round {st.session_state.round_number}")
        st.write(
            f"You played **{EMOJIS[last['player']]} {last['player']}** · "
            f"Computer played **{EMOJIS[last['computer']]} {last['computer']}**"
        )
        if last["result"] == "Win":
            st.success("You win! 🎉")
        elif last["result"] == "Loss":
            st.error("You lose.")
        else:
            st.warning("It's a draw.")

        if last["prediction"]:
            st.info(
                f"The AI predicted you would play **{last['prediction']}** "
                f"({last['probability']:.0%} likely), so it played **{last['computer']}**.",
                icon="🤖",
            )
        elif MODES[st.session_state.mode_label] == "adaptive":
            st.info("The AI played randomly this round (not enough information yet).", icon="🤖")

    # Scoreboard
    st.subheader("Score")
    scores = st.session_state.scores
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rounds", st.session_state.round_number)
    c2.metric("Wins", scores["Win"])
    c3.metric("Losses", scores["Loss"])
    c4.metric("Draws", scores["Draw"])

    st.button("🔄 New game", on_click=reset_game)

    # Round history of the current game
    st.subheader("📋 Round history (this game)")
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), hide_index=True)
    else:
        st.write("No rounds played yet.")


def p_sentence(p_value, yes_text, no_text):
    """Plain-language conclusion of a test at alpha = 0.05."""
    shown = format_p(p_value)
    shown = f"p {shown}" if shown.startswith("<") else f"p = {shown}"
    if p_value < 0.05:
        return f"✅ **{yes_text}** ({shown}, significant at α = 0.05)"
    return f"➖ {no_text} ({shown}, not significant at α = 0.05)"


def choose_player(data, key):
    """Player drop-down: the current player if they have data, otherwise the one with most rounds."""
    players = data["player"].value_counts().index.tolist()  # most rounds first
    current = clean_name(st.session_state.get("player_name", ""))
    return st.selectbox("Player", players, key=key,
                        index=players.index(current) if current in players else 0)


def render_statistics():
    """Descriptive statistics and hypothesis tests for one player (random-computer games only)."""
    if not DATA_FILE.exists():
        st.info("No data yet. Play some rounds first.")
        return
    data = load_data(DATA_FILE)  # random-computer rounds only
    if data.empty:
        st.info("No rounds against the random computer yet.")
        return

    player = choose_player(data, key="stats_player")
    pdf = data[data["player"] == player]
    n = len(pdf)
    st.caption(
        f"Based on {n} rounds against the **random** computer. Games against the adaptive AI "
        "are not included, because players may behave differently against an opponent that adapts."
    )

    # 1. How often each move is chosen
    st.subheader("1. How often does this player choose each move?")
    summary = move_summary(pdf)
    cols = st.columns(3)
    for col, move in zip(cols, MOVES):
        share = summary.loc[move, "proportion"]
        col.metric(f"{EMOJIS[move]} {move}", f"{share:.0%}",
                   delta=f"{round((share - 1/3) * 100):+d} pts vs random", delta_color="off")
    st.bar_chart(summary["proportion"], x_label="", y_label="Proportion of moves",
                 color="#2a78d6", horizontal=True, sort=False, height=180)

    # 2. Results
    st.subheader("2. Results against the random computer")
    rates = outcome_rates(pdf)
    cols = st.columns(3)
    for col, result in zip(cols, RESULTS):
        col.metric(f"{result} rate", f"{rates[result]:.0%}")
    st.caption("Against a random opponent every strategy wins, loses and draws about 1/3 of the "
               "time, so these rates mainly confirm that the computer is random.")

    if n < MIN_ROUNDS:
        st.warning(f"{player} has {n} rounds. At least {MIN_ROUNDS} are needed for the "
                   "statistical tests below, so they are not shown yet.")
        return

    test = uniformity_test(pdf)
    st.markdown("**Chi-square goodness-of-fit test** — are all three moves equally likely?  \n"
                + p_sentence(test["p_value"], f"{player} does not choose the moves equally often.",
                             "No evidence of a move preference."))

    # 3. Patterns: what comes next?
    st.subheader("3. What does this player play next?")
    st.caption("P(next move | previous move). Each row adds up to 100%. Random would be 33% everywhere.")
    probs = transition_probabilities(pdf)
    st.dataframe(probs.style.format("{:.0%}")
                 .background_gradient(cmap=BLUE_RAMP, vmin=0, vmax=0.7, axis=None,
                                      text_color_threshold=0.2))
    ind = independence_test(pdf)
    st.markdown("**Chi-square test of independence** — does the next move depend on the previous one?  \n"
                + p_sentence(ind["p_value"], "The next move depends on the previous move.",
                             "No evidence that the next move depends on the previous move."))
    st.metric("Repeat rate (same move twice in a row)", f"{repeat_rate(pdf):.0%}",
              delta=f"{round((repeat_rate(pdf) - 1/3) * 100):+d} pts vs random", delta_color="off")

    # 4. Win-stay / lose-shift
    st.subheader("4. Does winning or losing change the next move?")
    stay = stay_shift_table(pdf)
    cols = st.columns(3)
    for col, result in zip(cols, RESULTS):
        col.metric(f"Repeats after a {result.lower()}", f"{stay.loc[result, 'stay_rate']:.0%}")
    ws = stay_shift_test(pdf)
    st.markdown("**Chi-square test** — does the previous result affect repeating?  \n"
                + p_sentence(ws["p_value"], "The previous result affects whether the player repeats.",
                             "No evidence of win-stay / lose-shift."))


@st.cache_data
def cached_walk_forward(player_df):
    """Walk-forward evaluation (Stage 5). Cached: only recomputed when the data changes."""
    return walk_forward(player_df)


def render_ai():
    """How the AI works, how well it predicts, and how it did in real adaptive games."""
    if not DATA_FILE.exists():
        st.info("No data yet. Play some rounds first.")
        return
    data = load_data(DATA_FILE)  # random-computer rounds: used for training and evaluation
    if data.empty:
        st.info("No rounds against the random computer yet.")
        return

    player = choose_player(data, key="ai_player")
    pdf = data[data["player"] == player]

    # 1. The model
    st.subheader("1. How the AI thinks")
    st.markdown(
        "**Model:** Naive Bayes with one feature, the player's **previous move**. "
        "It estimates P(next move | previous move) from past rounds, predicts the most likely "
        "next move, and plays the move that beats it."
    )
    predictions = {previous: predict_next_move(pdf, previous) for previous in MOVES}
    if any(prediction is None for prediction, _ in predictions.values()):
        st.warning(f"{player} needs more rounds (at least {WARMUP} consecutive pairs of moves) "
                   "before the AI can learn.")
        return

    rows = []
    for previous, (prediction, probability) in predictions.items():
        rows.append({"If the previous move was": f"{EMOJIS[previous]} {previous}",
                     "AI predicts": f"{EMOJIS[prediction]} {prediction}",
                     "Probability": f"{probability:.0%}",
                     "AI plays": f"{EMOJIS[COUNTER[prediction]]} {COUNTER[prediction]}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(f"Learned from {player}'s {len(pdf)} rounds against the random computer.")

    if len(pdf) < MIN_ROUNDS:
        st.warning(f"{player} has {len(pdf)} rounds. At least {MIN_ROUNDS} are needed "
                   "for a reliable evaluation, so it is not shown yet.")
        return

    # 2. Walk-forward evaluation
    st.subheader("2. Does it predict better than simple guessing?")
    st.caption("Walk-forward test: for every round, the model learns only from EARLIER rounds "
               "and predicts the next one, exactly as it would in a real game.")
    preds = cached_walk_forward(pdf)
    n = len(preds)
    nb_right = preds["naive_bayes"] == preds["actual"]
    maj_right = preds["majority"] == preds["actual"]
    nb_acc, maj_acc = nb_right.mean(), maj_right.mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Random guess", "33%")
    c2.metric("Always the most frequent move", f"{maj_acc:.0%}")
    c3.metric("Naive Bayes", f"{nb_acc:.0%}",
              delta=f"{round((nb_acc - maj_acc) * 100):+d} pts vs most frequent", delta_color="off")
    ci = binomtest(int(nb_right.sum()), n).proportion_ci()
    st.caption(f"{int(nb_right.sum())} of {n} predictions correct · "
               f"95% confidence interval {ci.low:.0%} – {ci.high:.0%}")

    vs_random = binomtest(int(nb_right.sum()), n, p=1/3, alternative="greater").pvalue
    b, c = int((nb_right & ~maj_right).sum()), int((maj_right & ~nb_right).sum())
    vs_majority = binomtest(b, b + c, 0.5, alternative="greater").pvalue if b + c else 1.0
    st.markdown(
        "**Binomial test** — better than random guessing?  \n"
        + p_sentence(vs_random, "Yes, better than random guessing.", "Not better than random guessing.")
        + "  \n**McNemar test** — better than always guessing the most frequent move?  \n"
        + p_sentence(vs_majority, "Yes, better than the most-frequent-move baseline.",
                     "Not significantly better than the most-frequent-move baseline.")
    )

    outcomes = ai_outcomes(preds["actual"], preds["naive_bayes"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Simulated AI win rate", f"{outcomes['Win']:.0%}")
    c2.metric("Simulated AI loss rate", f"{outcomes['Loss']:.0%}")
    c3.metric("Simulated draw rate", f"{outcomes['Draw']:.0%}")
    st.caption("If the AI had played the counter-move in these recorded games. A random AI would "
               "win, lose and draw 33% each.")

    # 3. Confusion matrix
    st.subheader("3. Where is it right and wrong?")
    counts = confusion_table(preds["actual"], preds["naive_bayes"])
    st.dataframe(counts.style.background_gradient(cmap=BLUE_RAMP, axis=None,
                                                  text_color_threshold=0.2))
    st.caption("Rows = the player's real move, columns = the AI's prediction. "
               "The diagonal shows correct predictions.")
    never = [m for m in MOVES if counts[m].sum() == 0]
    if never:
        st.caption(f"The model never predicts {', '.join(never)}: after every previous move, "
                   "another move is more likely.")

    # 4. Real games against the adaptive AI
    st.subheader("4. How did it do in real games?")
    all_data = pd.read_csv(DATA_FILE)
    live = all_data[(all_data["player"] == player) & (all_data["ai_mode"] == "adaptive")
                    & all_data["ai_prediction"].notna()]
    if live.empty:
        st.info(f"{player} has not played against the adaptive AI yet.")
    else:
        live_acc = (live["ai_prediction"] == live["player_move"]).mean()
        ai_win = (live["result"] == "Loss").mean()   # the result column is from the player's view
        ai_loss = (live["result"] == "Win").mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("Predictions made", len(live))
        c2.metric("Prediction accuracy", f"{live_acc:.0%}")
        c3.metric("AI win / loss", f"{ai_win:.0%} / {ai_loss:.0%}")
        if len(live) < 30:
            word = "prediction" if len(live) == 1 else "predictions"
            st.caption(f"Only {len(live)} {word} so far, so these numbers are still very "
                       "uncertain. Players may also change how they play once they notice the AI.")
        else:
            st.caption("Compare with the walk-forward test above: a big drop could mean the player "
                       "changed how they play against an opponent that adapts.")

    st.caption("**Limitations:** only a few players; each played one long session; the walk-forward "
               "test uses games against a random computer, and players may react differently to an "
               "adaptive AI.")


def render_saved_data():
    """Summary of everything saved so far."""
    st.subheader("💾 Saved dataset (all games)")
    if DATA_FILE.exists():
        all_data = pd.read_csv(DATA_FILE)
        d1, d2, d3 = st.columns(3)
        d1.metric("Total rounds saved", len(all_data))
        d2.metric("Games played", all_data["game_id"].nunique())
        d3.metric("Players", all_data["player"].nunique())
        st.write("Rounds per player and opponent:")
        st.dataframe(pd.crosstab(all_data["player"], all_data["ai_mode"]))
    else:
        st.write("No data saved yet.")


# ---------- Page layout ----------
st.title("🎮 MindPlay AI")
st.caption("Adaptive Rock-Paper-Scissors · Statistics and machine learning on how people play")

tab_game, tab_stats, tab_ai = st.tabs(["🎮 Game", "📊 Player statistics", "🤖 AI / ML"])
with tab_game:
    render_game()
with tab_stats:
    render_statistics()
    st.divider()
    render_saved_data()
with tab_ai:
    render_ai()
