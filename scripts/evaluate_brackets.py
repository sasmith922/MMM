"""Bracket-focused evaluation for March Madness model predictions.

This script complements per-game metrics with bracket-quality and simulation-based
analysis. It does not modify existing evaluation outputs; it writes to
``outputs/bracket_reports`` by default.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "predictions_v2" / "model_predictions_by_season_v2.csv"
DEFAULT_ACTUALS_PATH = PROJECT_ROOT / "data" / "processed_v2" / "tournament_matchups_v2.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "bracket_reports"

ROUND_ORDER = ["R64", "R32", "S16", "E8", "F4", "CHAMP"]
ROUND_INDEX = {round_name: index for index, round_name in enumerate(ROUND_ORDER)}
ROUND_WEIGHTS = {
    "R64": 1,
    "R32": 2,
    "S16": 4,
    "E8": 8,
    "F4": 16,
    "CHAMP": 32,
}


@dataclass
class BracketGameNode:
    game_id: str
    round_name: str
    season: int
    teamA_actual: int
    teamB_actual: int
    actual_winner: int
    dep_left_game_id: str | None
    dep_right_game_id: str | None


@dataclass
class SimulationRunResult:
    winners_by_game: dict[str, int]
    round_reached_by_team: dict[int, str]


ROUND_ALIASES = {
    "R64": "R64",
    "ROUND64": "R64",
    "ROUND_OF_64": "R64",
    "ROUND OF 64": "R64",
    "FIRST ROUND": "R64",
    "1": "R64",
    "R32": "R32",
    "ROUND32": "R32",
    "ROUND_OF_32": "R32",
    "ROUND OF 32": "R32",
    "SECOND ROUND": "R32",
    "2": "R32",
    "S16": "S16",
    "SWEET16": "S16",
    "SWEET 16": "S16",
    "3": "S16",
    "E8": "E8",
    "ELITE8": "E8",
    "ELITE 8": "E8",
    "4": "E8",
    "F4": "F4",
    "FINAL4": "F4",
    "FINAL FOUR": "F4",
    "FINALFOUR": "F4",
    "5": "F4",
    "CHAMP": "CHAMP",
    "CHAMPIONSHIP": "CHAMP",
    "TITLE": "CHAMP",
    "6": "CHAMP",
}


def _canonical_round(value: Any) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer, float, np.floating)) and not math.isnan(float(value)):
        integer_value = int(value)
        mapping = {1: "R64", 2: "R32", 3: "S16", 4: "E8", 5: "F4", 6: "CHAMP"}
        return mapping.get(integer_value)

    text = str(value).strip().upper()
    text = text.replace("-", " ").replace("/", " ")
    text = " ".join(text.split())
    text_no_space = text.replace(" ", "")
    return ROUND_ALIASES.get(text) or ROUND_ALIASES.get(text_no_space)


def _normalize_matchups(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "team_a_id": "teamA_id",
        "team_b_id": "teamB_id",
        "label": "target",
        "y": "target",
        "round_num_guess": "round",
    }
    if "season" not in df.columns and "test_season" in df.columns:
        rename_map["test_season"] = "season"
    existing = {old: new for old, new in rename_map.items() if old in df.columns}
    return df.rename(columns=existing)


def _require_columns(df: pd.DataFrame, required: list[str], table_name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"{table_name} missing required columns {missing}. Available: {list(df.columns)}")


def load_inputs(predictions_path: Path, actuals_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load predictions and actual matchup outcomes for bracket evaluation."""
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    if not actuals_path.exists():
        raise FileNotFoundError(f"Actual matchups file not found: {actuals_path}")

    predictions_df = _normalize_matchups(pd.read_csv(predictions_path))
    actuals_df = _normalize_matchups(pd.read_csv(actuals_path))

    _require_columns(predictions_df, ["season", "teamA_id", "teamB_id", "pred_prob"], "predictions")
    _require_columns(actuals_df, ["season", "teamA_id", "teamB_id", "target"], "actuals")

    if "model_name" not in predictions_df.columns:
        predictions_df["model_name"] = "unknown_model"

    predictions_df["season"] = pd.to_numeric(predictions_df["season"], errors="coerce")
    predictions_df["teamA_id"] = pd.to_numeric(predictions_df["teamA_id"], errors="coerce")
    predictions_df["teamB_id"] = pd.to_numeric(predictions_df["teamB_id"], errors="coerce")
    predictions_df["pred_prob"] = pd.to_numeric(predictions_df["pred_prob"], errors="coerce")

    actuals_df["season"] = pd.to_numeric(actuals_df["season"], errors="coerce")
    actuals_df["teamA_id"] = pd.to_numeric(actuals_df["teamA_id"], errors="coerce")
    actuals_df["teamB_id"] = pd.to_numeric(actuals_df["teamB_id"], errors="coerce")
    actuals_df["target"] = pd.to_numeric(actuals_df["target"], errors="coerce")

    predictions_df = predictions_df.dropna(subset=["season", "teamA_id", "teamB_id", "pred_prob"]).copy()
    actuals_df = actuals_df.dropna(subset=["season", "teamA_id", "teamB_id", "target"]).copy()

    predictions_df["season"] = predictions_df["season"].astype(int)
    predictions_df["teamA_id"] = predictions_df["teamA_id"].astype(int)
    predictions_df["teamB_id"] = predictions_df["teamB_id"].astype(int)
    predictions_df["pred_prob"] = predictions_df["pred_prob"].clip(1e-6, 1 - 1e-6)

    actuals_df["season"] = actuals_df["season"].astype(int)
    actuals_df["teamA_id"] = actuals_df["teamA_id"].astype(int)
    actuals_df["teamB_id"] = actuals_df["teamB_id"].astype(int)
    actuals_df["target"] = actuals_df["target"].astype(int)

    if "round" in actuals_df.columns:
        actuals_df["round_name"] = actuals_df["round"].map(_canonical_round)
    elif "round_name" in actuals_df.columns:
        actuals_df["round_name"] = actuals_df["round_name"].map(_canonical_round)
    else:
        raise KeyError("Actual matchup file must include a round or round_name column.")

    if actuals_df["round_name"].isna().any():
        round_col = "round" if "round" in actuals_df.columns else "round_name"
        unknown_rounds = actuals_df.loc[actuals_df["round_name"].isna(), round_col].unique()
        raise ValueError(f"Could not canonicalize one or more round labels: {unknown_rounds}")

    actuals_df["actual_winner"] = np.where(
        actuals_df["target"] == 1,
        actuals_df["teamA_id"],
        actuals_df["teamB_id"],
    )

    print(f"Loaded predictions rows: {len(predictions_df):,}")
    print(f"Loaded actual rows: {len(actuals_df):,}")
    print(f"Prediction seasons: {sorted(predictions_df['season'].unique().tolist())}")
    print(f"Actual seasons: {sorted(actuals_df['season'].unique().tolist())}")

    return predictions_df, actuals_df


def align_predictions_to_actuals(predictions_df: pd.DataFrame, actuals_df: pd.DataFrame) -> pd.DataFrame:
    """Align prediction rows to actual games, handling swapped team order."""
    direct = predictions_df[["season", "model_name", "teamA_id", "teamB_id", "pred_prob"]].copy()
    direct["is_swapped"] = 0

    swapped = direct.rename(columns={"teamA_id": "teamB_id", "teamB_id": "teamA_id"}).copy()
    swapped["pred_prob"] = 1.0 - swapped["pred_prob"]
    swapped["is_swapped"] = 1

    expanded = pd.concat([direct, swapped], ignore_index=True)
    expanded = expanded.sort_values(["season", "model_name", "teamA_id", "teamB_id", "is_swapped"])
    expanded = expanded.drop_duplicates(["season", "model_name", "teamA_id", "teamB_id"], keep="first")

    base_cols = ["season", "teamA_id", "teamB_id", "target", "actual_winner", "round_name"]
    extra_cols = ["region"] if "region" in actuals_df.columns else []
    game_keys = actuals_df[base_cols + extra_cols]
    merged = game_keys.merge(expanded, on=["season", "teamA_id", "teamB_id"], how="left")

    missing_predictions = int(merged["pred_prob"].isna().sum())
    if missing_predictions > 0:
        print(f"[warn] Missing predictions for {missing_predictions} actual game rows; dropping those rows")
        merged = merged.dropna(subset=["pred_prob"]).copy()

    merged["pred_winner"] = np.where(merged["pred_prob"] >= 0.5, merged["teamA_id"], merged["teamB_id"])
    merged["pred_class"] = (merged["pred_prob"] >= 0.5).astype(int)

    return merged


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _weighted_log_loss(y_true: np.ndarray, y_prob: np.ndarray, weights: np.ndarray) -> float:
    return float(log_loss(y_true, y_prob, sample_weight=weights, labels=[0, 1]))


def _weighted_brier(y_true: np.ndarray, y_prob: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average((y_true - y_prob) ** 2, weights=weights))


def _weighted_accuracy(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average((y_true == y_pred).astype(float), weights=weights))


def _build_season_game_nodes(actual_season_df: pd.DataFrame) -> list[BracketGameNode]:
    """Build game graph nodes from actual game rows by tracing prior-round provenance."""
    rows = actual_season_df.copy()
    rows["round_idx"] = rows["round_name"].map(ROUND_INDEX)

    if rows["round_idx"].isna().any():
        raise ValueError("Unexpected non-canonical round labels in actual season rows.")

    rows = rows.sort_values(["round_idx", "teamA_id", "teamB_id"]).reset_index(drop=True)

    game_nodes_by_round: dict[str, list[BracketGameNode]] = {rn: [] for rn in ROUND_ORDER}

    for round_name in ROUND_ORDER:
        round_rows = rows[rows["round_name"] == round_name].copy().reset_index(drop=True)
        if round_rows.empty:
            continue

        for idx, row in round_rows.iterrows():
            game_id = f"{round_name}_{idx + 1}"
            dep_left = None
            dep_right = None

            if round_name != "R64":
                prev_round = ROUND_ORDER[ROUND_INDEX[round_name] - 1]
                prior_games = game_nodes_by_round.get(prev_round, [])

                def _find_source_game(team_id: int) -> str | None:
                    for prior_game in prior_games:
                        if team_id in {prior_game.teamA_actual, prior_game.teamB_actual}:
                            return prior_game.game_id
                    return None

                dep_left = _find_source_game(int(row["teamA_id"]))
                dep_right = _find_source_game(int(row["teamB_id"]))

                if dep_left is None or dep_right is None or dep_left == dep_right:
                    dep_left = None
                    dep_right = None

            node = BracketGameNode(
                game_id=game_id,
                round_name=round_name,
                season=int(row["season"]),
                teamA_actual=int(row["teamA_id"]),
                teamB_actual=int(row["teamB_id"]),
                actual_winner=int(row["actual_winner"]),
                dep_left_game_id=dep_left,
                dep_right_game_id=dep_right,
            )
            game_nodes_by_round[round_name].append(node)

    nodes: list[BracketGameNode] = []
    for round_name in ROUND_ORDER:
        nodes.extend(game_nodes_by_round.get(round_name, []))
    return nodes


def _build_probability_lookup(season_model_df: pd.DataFrame) -> dict[tuple[int, int], float]:
    lookup: dict[tuple[int, int], float] = {}
    for row in season_model_df.itertuples(index=False):
        team_a = int(getattr(row, "teamA_id"))
        team_b = int(getattr(row, "teamB_id"))
        prob_a = float(getattr(row, "pred_prob"))
        lookup[(team_a, team_b)] = prob_a
        lookup[(team_b, team_a)] = 1.0 - prob_a
    return lookup


def _simulate_from_nodes(
    game_nodes: list[BracketGameNode],
    prob_lookup: dict[tuple[int, int], float],
    *,
    deterministic: bool,
    rng: np.random.Generator,
) -> SimulationRunResult:
    winners_by_game: dict[str, int] = {}
    round_reached_by_team: dict[int, str] = {}

    for node in game_nodes:
        team_a, team_b = _resolve_game_teams(node, winners_by_game)

        prob_a = prob_lookup.get((team_a, team_b), 0.5)
        prob_a = float(np.clip(prob_a, 1e-6, 1 - 1e-6))

        if deterministic:
            winner = team_a if prob_a >= 0.5 else team_b
        else:
            winner = team_a if rng.random() < prob_a else team_b

        loser = team_b if winner == team_a else team_a
        winners_by_game[node.game_id] = winner
        round_reached_by_team[winner] = node.round_name
        round_reached_by_team[loser] = node.round_name

    return SimulationRunResult(winners_by_game=winners_by_game, round_reached_by_team=round_reached_by_team)


def _resolve_game_teams(node: BracketGameNode, winners_by_game: dict[str, int]) -> tuple[int, int]:
    if node.dep_left_game_id is None:
        team_a = node.teamA_actual
    else:
        team_a = winners_by_game[node.dep_left_game_id]

    if node.dep_right_game_id is None:
        team_b = node.teamB_actual
    else:
        team_b = winners_by_game[node.dep_right_game_id]

    return team_a, team_b


def _deterministic_bracket_metrics(
    game_nodes: list[BracketGameNode],
    deterministic_result: SimulationRunResult,
) -> dict[str, Any]:
    score = 0
    for node in game_nodes:
        predicted = deterministic_result.winners_by_game[node.game_id]
        if predicted == node.actual_winner:
            score += ROUND_WEIGHTS[node.round_name]

    champ_node = next(node for node in game_nodes if node.round_name == "CHAMP")
    true_champion = champ_node.actual_winner
    pred_champion = deterministic_result.winners_by_game[champ_node.game_id]

    champ_participants = {champ_node.teamA_actual, champ_node.teamB_actual}
    predicted_finalists = set()
    for node in game_nodes:
        if node.round_name == "CHAMP":
            left_winner, right_winner = _resolve_game_teams(node, deterministic_result.winners_by_game)
            predicted_finalists.update({left_winner, right_winner})

    predicted_round_teams: dict[str, set[int]] = {rn: set() for rn in ROUND_ORDER}
    actual_round_teams: dict[str, set[int]] = {rn: set() for rn in ROUND_ORDER}

    for node in game_nodes:
        actual_round_teams[node.round_name].update({node.teamA_actual, node.teamB_actual})
        team_a, team_b = _resolve_game_teams(node, deterministic_result.winners_by_game)
        predicted_round_teams[node.round_name].update({team_a, team_b})

    true_runner_up = next((team for team in champ_participants if team != true_champion), None)
    pred_runner_up = (
        next((team for team in predicted_finalists if team != pred_champion), None)
        if len(predicted_finalists) == 2
        else None
    )

    return {
        "predicted_bracket_score": int(score),
        "predicted_champion": int(pred_champion),
        "true_champion": int(true_champion),
        "champion_correct": int(pred_champion == true_champion),
        "predicted_runner_up": int(pred_runner_up) if pred_runner_up is not None else np.nan,
        "true_runner_up": int(true_runner_up) if true_runner_up is not None else np.nan,
        "runner_up_correct": int(pred_runner_up == true_runner_up) if pred_runner_up is not None else 0,
        "champ_game_participants_correct": int(len(predicted_finalists.intersection(champ_participants))),
        "final_four_correct": int(len(predicted_round_teams["F4"].intersection(actual_round_teams["F4"]))),
        "elite_eight_correct": int(len(predicted_round_teams["E8"].intersection(actual_round_teams["E8"]))),
        "sweet_16_correct": int(len(predicted_round_teams["S16"].intersection(actual_round_teams["S16"]))),
    }


def evaluate(
    predictions_path: Path,
    actuals_path: Path,
    output_dir: Path,
    n_simulations: int,
    random_state: int,
) -> dict[str, pd.DataFrame]:
    """Run bracket-focused evaluation and write all required report CSVs."""
    predictions_df, actuals_df = load_inputs(predictions_path, actuals_path)
    merged = align_predictions_to_actuals(predictions_df, actuals_df)

    output_dir.mkdir(parents=True, exist_ok=True)

    season_metrics_rows: list[dict[str, Any]] = []
    champion_prob_rows: list[dict[str, Any]] = []
    round_prob_rows: list[dict[str, Any]] = []
    simulation_summary_rows: list[dict[str, Any]] = []

    grouped = merged.groupby(["model_name", "season"], as_index=False)

    for model_name, season in grouped.groups:
        season_model_df = grouped.get_group((model_name, season)).copy()

        y_true = season_model_df["target"].to_numpy(dtype=int)
        y_prob = season_model_df["pred_prob"].to_numpy(dtype=float)
        y_pred = season_model_df["pred_class"].to_numpy(dtype=int)
        weights = season_model_df["round_name"].map(ROUND_WEIGHTS).to_numpy(dtype=float)

        game_nodes = _build_season_game_nodes(season_model_df[["season", "teamA_id", "teamB_id", "actual_winner", "round_name"]])
        prob_lookup = _build_probability_lookup(season_model_df)

        model_seed_offset = int(hashlib.sha256(str(model_name).encode("utf-8")).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed=random_state + int(season) + model_seed_offset)
        deterministic_result = _simulate_from_nodes(game_nodes, prob_lookup, deterministic=True, rng=rng)

        deterministic_metrics = _deterministic_bracket_metrics(game_nodes, deterministic_result)

        sim_scores: list[int] = []
        champion_counts: dict[int, int] = {}
        round_reach_counts: dict[int, dict[str, int]] = {}

        for _ in range(n_simulations):
            sim_result = _simulate_from_nodes(game_nodes, prob_lookup, deterministic=False, rng=rng)
            sim_score = 0
            for node in game_nodes:
                if sim_result.winners_by_game[node.game_id] == node.actual_winner:
                    sim_score += ROUND_WEIGHTS[node.round_name]
            sim_scores.append(sim_score)

            champ_game = next(node for node in game_nodes if node.round_name == "CHAMP")
            champ_team = sim_result.winners_by_game[champ_game.game_id]
            champion_counts[champ_team] = champion_counts.get(champ_team, 0) + 1

            for team_id, reached_round in sim_result.round_reached_by_team.items():
                team_counts = round_reach_counts.setdefault(team_id, {rn: 0 for rn in ROUND_ORDER})
                reached_idx = ROUND_INDEX[reached_round]
                for rn in ROUND_ORDER:
                    if ROUND_INDEX[rn] <= reached_idx:
                        team_counts[rn] += 1

        champion_probs = {
            team_id: count / n_simulations for team_id, count in champion_counts.items()
        }

        for team_id, prob in champion_probs.items():
            champion_prob_rows.append(
                {
                    "model_name": model_name,
                    "season": int(season),
                    "team_id": int(team_id),
                    "champion_prob": float(prob),
                }
            )

        for team_id, counts in round_reach_counts.items():
            row = {
                "model_name": model_name,
                "season": int(season),
                "team_id": int(team_id),
            }
            for rn in ROUND_ORDER:
                row[f"reach_{rn.lower()}_prob"] = float(counts[rn] / n_simulations)
            round_prob_rows.append(row)

        champion_ranking = sorted(champion_probs.items(), key=lambda item: item[1], reverse=True)
        top_ranked_teams = [team_id for team_id, _ in champion_ranking]
        true_champion = int(deterministic_metrics["true_champion"])
        actual_round_reached: dict[int, str] = {}
        for node in game_nodes:
            actual_round_reached[node.teamA_actual] = node.round_name
            actual_round_reached[node.teamB_actual] = node.round_name
        top_teams_for_round_metric = top_ranked_teams[:8]
        top_team_round_values = [
            ROUND_INDEX[actual_round_reached[team_id]] + 1
            for team_id in top_teams_for_round_metric
            if team_id in actual_round_reached
        ]
        avg_actual_round_top_predicted = (
            float(np.mean(top_team_round_values))
            if top_team_round_values
            else float("nan")
        )

        season_metrics_rows.append(
            {
                "model_name": model_name,
                "season": int(season),
                "n_games": int(len(season_model_df)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
                "brier_score": float(brier_score_loss(y_true, y_prob)),
                "roc_auc": _safe_roc_auc(y_true, y_prob),
                "round_weighted_accuracy": _weighted_accuracy(y_true, y_pred, weights),
                "round_weighted_log_loss": _weighted_log_loss(y_true, y_prob, weights),
                "round_weighted_brier_score": _weighted_brier(y_true, y_prob, weights),
                **deterministic_metrics,
                "avg_simulated_bracket_score": float(np.mean(sim_scores)) if sim_scores else float("nan"),
                "best_simulated_bracket_score": int(max(sim_scores)) if sim_scores else np.nan,
                "true_champion_in_top1": int(true_champion in top_ranked_teams[:1]),
                "true_champion_in_top2": int(true_champion in top_ranked_teams[:2]),
                "true_champion_in_top4": int(true_champion in top_ranked_teams[:4]),
                "true_champion_in_top8": int(true_champion in top_ranked_teams[:8]),
                "avg_actual_round_reached_top8_predicted": avg_actual_round_top_predicted,
            }
        )

        likely_champion = champion_ranking[0][0] if champion_ranking else np.nan
        predicted_round_teams: dict[str, set[int]] = {rn: set() for rn in ROUND_ORDER}
        for node in game_nodes:
            team_a, team_b = _resolve_game_teams(node, deterministic_result.winners_by_game)
            predicted_round_teams[node.round_name].update({team_a, team_b})
        simulation_summary_rows.append(
            {
                "model_name": model_name,
                "season": int(season),
                "n_simulations": int(n_simulations),
                "most_likely_champion": int(likely_champion) if pd.notna(likely_champion) else np.nan,
                "most_likely_champion_prob": float(champion_ranking[0][1]) if champion_ranking else np.nan,
                "most_likely_finalists": ",".join(str(team) for team in sorted(predicted_round_teams["CHAMP"])),
                "most_likely_final_four": ",".join(str(team) for team in sorted(predicted_round_teams["F4"])),
                "most_likely_elite_eight": ",".join(str(team) for team in sorted(predicted_round_teams["E8"])),
                "predicted_bracket_score": deterministic_metrics["predicted_bracket_score"],
                "avg_simulated_bracket_score": float(np.mean(sim_scores)) if sim_scores else float("nan"),
                "best_simulated_bracket_score": int(max(sim_scores)) if sim_scores else np.nan,
            }
        )

        print(f"Evaluated model={model_name}, season={season}, games={len(season_model_df)}, sims={n_simulations}")

    bracket_metrics_by_season = pd.DataFrame(season_metrics_rows).sort_values(["model_name", "season"]).reset_index(drop=True)
    bracket_summary_by_model = (
        bracket_metrics_by_season.groupby("model_name", as_index=False)
        .agg(
            seasons_evaluated=("season", "nunique"),
            mean_accuracy=("accuracy", "mean"),
            mean_log_loss=("log_loss", "mean"),
            mean_brier_score=("brier_score", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_round_weighted_accuracy=("round_weighted_accuracy", "mean"),
            mean_round_weighted_log_loss=("round_weighted_log_loss", "mean"),
            mean_round_weighted_brier_score=("round_weighted_brier_score", "mean"),
            champion_hit_rate=("champion_correct", "mean"),
            runner_up_hit_rate=("runner_up_correct", "mean"),
            avg_final_four_correct=("final_four_correct", "mean"),
            avg_elite_eight_correct=("elite_eight_correct", "mean"),
            avg_sweet_16_correct=("sweet_16_correct", "mean"),
            avg_predicted_bracket_score=("predicted_bracket_score", "mean"),
            avg_simulated_bracket_score=("avg_simulated_bracket_score", "mean"),
            avg_best_simulated_bracket_score=("best_simulated_bracket_score", "mean"),
            pct_true_champion_top1=("true_champion_in_top1", "mean"),
            pct_true_champion_top2=("true_champion_in_top2", "mean"),
            pct_true_champion_top4=("true_champion_in_top4", "mean"),
            pct_true_champion_top8=("true_champion_in_top8", "mean"),
            avg_actual_round_reached_top8_predicted=("avg_actual_round_reached_top8_predicted", "mean"),
        )
        .sort_values("mean_round_weighted_log_loss", ascending=True)
        .reset_index(drop=True)
    )

    champion_probs_by_season = pd.DataFrame(champion_prob_rows).sort_values(["model_name", "season", "champion_prob"], ascending=[True, True, False]).reset_index(drop=True)
    round_reach_probs_by_season = pd.DataFrame(round_prob_rows).sort_values(["model_name", "season", "team_id"]).reset_index(drop=True)
    bracket_simulation_summary = pd.DataFrame(simulation_summary_rows).sort_values(["model_name", "season"]).reset_index(drop=True)

    metrics_path = output_dir / "bracket_metrics_by_season.csv"
    summary_path = output_dir / "bracket_summary_by_model.csv"
    champion_probs_path = output_dir / "champion_probs_by_season.csv"
    round_reach_path = output_dir / "round_reach_probs_by_season.csv"
    simulation_summary_path = output_dir / "bracket_simulation_summary.csv"

    bracket_metrics_by_season.to_csv(metrics_path, index=False)
    bracket_summary_by_model.to_csv(summary_path, index=False)
    champion_probs_by_season.to_csv(champion_probs_path, index=False)
    round_reach_probs_by_season.to_csv(round_reach_path, index=False)
    bracket_simulation_summary.to_csv(simulation_summary_path, index=False)

    print("\nSaved bracket-focused outputs:")
    print(f"  {metrics_path}")
    print(f"  {summary_path}")
    print(f"  {champion_probs_path}")
    print(f"  {round_reach_path}")
    print(f"  {simulation_summary_path}")

    return {
        "bracket_metrics_by_season": bracket_metrics_by_season,
        "bracket_summary_by_model": bracket_summary_by_model,
        "champion_probs_by_season": champion_probs_by_season,
        "round_reach_probs_by_season": round_reach_probs_by_season,
        "bracket_simulation_summary": bracket_simulation_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model outputs with bracket-focused metrics.")
    parser.add_argument("--predictions-path", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--actuals-path", type=Path, default=DEFAULT_ACTUALS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-simulations", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        predictions_path=args.predictions_path,
        actuals_path=args.actuals_path,
        output_dir=args.output_dir,
        n_simulations=args.n_simulations,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
