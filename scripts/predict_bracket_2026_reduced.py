"""
predict_bracket_2026_reduced.py
-------------------------------
Train (or load) the 2026 reduced model and generate a full bracket-style
deterministic prediction breakdown.

Usage
-----
    python scripts/predict_bracket_2026_reduced.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config.features_2026_reduced import (
    BRACKET_BREAKDOWN_2026_PATH,
    BRACKET_SUMMARY_2026_PATH,
    FEATURE_LIST_2026_REDUCED_PATH,
    MODEL_ARTIFACT_2026_PATH,
    TEAM_FEATURES_2026_REDUCED_PATH,
    TOURNEY_MATCHUPS_2026_PATH,
    ensure_parent,
    parse_seed_number,
    parse_seed_region,
)
from madness_model.model_utils import load_model
from scripts.train_predict_2026_reduced import TEAM_FEATURE_TO_2026_COLUMNS, train_and_predict_2026_reduced

R64_SEED_PAIRS = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
PAIR_ORDER = {frozenset(pair): idx + 1 for idx, pair in enumerate(R64_SEED_PAIRS)}

ROUND_LABELS = {
    "R64": "Round of 64",
    "R32": "Round of 32",
    "S16": "Sweet 16",
    "E8": "Elite 8",
    "F4": "Final Four",
    "CHAMP": "Championship",
}

FINAL_FOUR_REGION_PAIRS = [("East", "West"), ("South", "Midwest")]
REGION_MAP = {"W": "West", "X": "East", "Y": "South", "Z": "Midwest"}


@dataclass
class TeamRecord:
    season: int
    team_name: str
    team_name_norm: str
    seed: str
    seed_num: float
    region: str
    stats: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate full 2026 reduced bracket predictions.")
    parser.add_argument("--team-features-2026-path", type=Path, default=TEAM_FEATURES_2026_REDUCED_PATH)
    parser.add_argument("--tourney-matchups-2026-path", type=Path, default=TOURNEY_MATCHUPS_2026_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_ARTIFACT_2026_PATH)
    parser.add_argument("--feature-list-path", type=Path, default=FEATURE_LIST_2026_REDUCED_PATH)
    parser.add_argument("--breakdown-output-path", type=Path, default=BRACKET_BREAKDOWN_2026_PATH)
    parser.add_argument("--summary-output-path", type=Path, default=BRACKET_SUMMARY_2026_PATH)
    parser.add_argument("--train-if-missing", action="store_true", default=True)
    parser.add_argument("--no-train-if-missing", action="store_true", help="Fail if model/feature list is missing.")
    return parser.parse_args()


def _region_display(region: str) -> str:
    return REGION_MAP.get(str(region).upper(), str(region))


def _load_feature_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Feature list file not found: {path}")
    features = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not features:
        raise ValueError(f"Feature list file is empty: {path}")
    return features


def _build_team_lookup(team_features: pd.DataFrame) -> dict[str, TeamRecord]:
    required = ["season", "team_name", "team_name_norm", "seed"]
    missing = [column for column in required if column not in team_features.columns]
    if missing:
        raise KeyError(f"2026 team features missing required columns: {missing}")

    teams = team_features.copy()
    teams["season"] = pd.to_numeric(teams["season"], errors="coerce").astype("Int64")
    teams["seed_num"] = teams["seed"].map(parse_seed_number)
    teams["region"] = teams["seed"].map(parse_seed_region).fillna("ALL").map(_region_display)

    lookup: dict[str, TeamRecord] = {}
    for _, row in teams.iterrows():
        key = str(row["team_name_norm"])
        stats = {
            column: float(pd.to_numeric(row.get(column), errors="coerce"))
            for _, (column, _) in TEAM_FEATURE_TO_2026_COLUMNS.items()
            if column in teams.columns
        }
        lookup[key] = TeamRecord(
            season=int(row["season"]),
            team_name=str(row["team_name"]),
            team_name_norm=key,
            seed=str(row["seed"]),
            seed_num=float(row["seed_num"]) if pd.notna(row["seed_num"]) else float("nan"),
            region=str(row["region"]),
            stats=stats,
        )
    return lookup


def _predict_prob_team1_wins(model, feature_cols: list[str], team1: TeamRecord, team2: TeamRecord) -> float:
    """Predict Team 1 win probability using reduced diffs, filling missing feature values with 0.0."""
    row = {}
    for feature_name in feature_cols:
        a_col, b_col = TEAM_FEATURE_TO_2026_COLUMNS.get(feature_name, (None, None))
        if a_col is None or b_col is None:
            raise KeyError(f"Unsupported reduced feature for matchup prediction: {feature_name}")
        value = float(team1.stats.get(a_col, np.nan)) - float(team2.stats.get(b_col, np.nan))
        if np.isnan(value):
            value = 0.0
        row[feature_name] = value

    model_features = None
    if hasattr(model, "feature_names_in_"):
        try:
            model_features = list(model.feature_names_in_)
        except Exception:
            model_features = None
    elif hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
            model_features = list(getattr(booster, "feature_names", []) or [])
        except Exception:
            model_features = None

    if model_features:
        row = {feature: float(row.get(feature, 0.0)) for feature in model_features}
        X = pd.DataFrame([row], columns=model_features).fillna(0.0)
    else:
        X = pd.DataFrame([row], columns=feature_cols).fillna(0.0)
    prob = float(model.predict_proba(X)[:, 1][0])
    if np.isnan(prob) or prob < 0.0 or prob > 1.0:
        raise ValueError(f"Model returned invalid probability for {team1.team_name} vs {team2.team_name}: {prob}")
    return prob


def _make_game_record(
    *,
    round_name: str,
    game_slot: str,
    next_round_slot: str | None,
    region: str,
    team1: TeamRecord,
    team2: TeamRecord,
    winner: TeamRecord,
    winner_prob: float,
    prob_team1: float,
) -> dict[str, object]:
    return {
        "round": round_name,
        "round_label": ROUND_LABELS.get(round_name, round_name),
        "region": region,
        "game_slot": game_slot,
        "next_round_slot": next_round_slot or "",
        "team_1": team1.team_name,
        "team_2": team2.team_name,
        "team_1_seed": team1.seed,
        "team_2_seed": team2.seed,
        "predicted_winner": winner.team_name,
        "predicted_win_probability": round(float(winner_prob), 6),
        "team_1_win_probability": round(float(prob_team1), 6),
        "team_1_name_norm": team1.team_name_norm,
        "team_2_name_norm": team2.team_name_norm,
        "predicted_winner_name_norm": winner.team_name_norm,
    }


def _predict_game(
    *,
    model,
    feature_cols: list[str],
    round_name: str,
    region: str,
    game_slot: str,
    next_round_slot: str | None,
    team1: TeamRecord,
    team2: TeamRecord,
) -> tuple[TeamRecord, dict[str, object]]:
    prob_team1 = _predict_prob_team1_wins(model, feature_cols, team1, team2)
    winner = team1 if prob_team1 >= 0.5 else team2
    winner_prob = prob_team1 if winner.team_name_norm == team1.team_name_norm else 1.0 - prob_team1
    row = _make_game_record(
        round_name=round_name,
        game_slot=game_slot,
        next_round_slot=next_round_slot,
        region=region,
        team1=team1,
        team2=team2,
        winner=winner,
        winner_prob=winner_prob,
        prob_team1=prob_team1,
    )
    return winner, row


def _get_r64_by_region(matchups_2026: pd.DataFrame, team_lookup: dict[str, TeamRecord]) -> dict[str, list[tuple[str, TeamRecord, TeamRecord]]]:
    required = ["teamA_name_norm", "teamB_name_norm"]
    missing = [column for column in required if column not in matchups_2026.columns]
    if missing:
        raise KeyError(f"2026 matchups missing required columns: {missing}")

    rows = matchups_2026.copy()
    if "region" not in rows.columns:
        rows["region"] = "ALL"

    rows["teamA_seed_num"] = pd.to_numeric(rows.get("teamA_seed_num"), errors="coerce")
    rows["teamB_seed_num"] = pd.to_numeric(rows.get("teamB_seed_num"), errors="coerce")
    if rows["teamA_seed_num"].isna().any():
        rows["teamA_seed_num"] = rows["teamA_name_norm"].map(lambda x: team_lookup[str(x)].seed_num)
    if rows["teamB_seed_num"].isna().any():
        rows["teamB_seed_num"] = rows["teamB_name_norm"].map(lambda x: team_lookup[str(x)].seed_num)

    rows["region"] = rows["region"].map(_region_display)
    rows["pair_order"] = rows.apply(
        lambda r: PAIR_ORDER.get(frozenset({int(r["teamA_seed_num"]), int(r["teamB_seed_num"])}), 999),
        axis=1,
    )
    rows = rows.sort_values(["region", "pair_order", "teamA_name_norm", "teamB_name_norm"]).reset_index(drop=True)

    by_region: dict[str, list[tuple[str, TeamRecord, TeamRecord]]] = {}
    for region, group in rows.groupby("region", dropna=False):
        games: list[tuple[str, TeamRecord, TeamRecord]] = []
        for idx, (_, row) in enumerate(group.iterrows(), start=1):
            a_key = str(row["teamA_name_norm"])
            b_key = str(row["teamB_name_norm"])
            if a_key not in team_lookup or b_key not in team_lookup:
                raise KeyError(f"2026 matchup references unknown teams: {a_key}, {b_key}")
            games.append((f"R64_{region}_{idx}", team_lookup[a_key], team_lookup[b_key]))
        by_region[str(region)] = games
    if not by_region:
        raise ValueError("No 2026 round-of-64 games available for bracket prediction.")
    return by_region


def _simulate_region_rounds(
    *,
    model,
    feature_cols: list[str],
    region: str,
    r64_games: list[tuple[str, TeamRecord, TeamRecord]],
) -> tuple[list[dict[str, object]], TeamRecord]:
    rows: list[dict[str, object]] = []
    r64_winners: list[TeamRecord] = []
    for idx, (slot, team1, team2) in enumerate(r64_games, start=1):
        next_slot = f"R32_{region}_{(idx + 1) // 2}"
        winner, row = _predict_game(
            model=model,
            feature_cols=feature_cols,
            round_name="R64",
            region=region,
            game_slot=slot,
            next_round_slot=next_slot,
            team1=team1,
            team2=team2,
        )
        rows.append(row)
        r64_winners.append(winner)

    r32_winners: list[TeamRecord] = []
    for idx in range(0, len(r64_winners), 2):
        if idx + 1 >= len(r64_winners):
            continue
        game_num = (idx // 2) + 1
        next_slot = f"S16_{region}_{(game_num + 1) // 2}"
        winner, row = _predict_game(
            model=model,
            feature_cols=feature_cols,
            round_name="R32",
            region=region,
            game_slot=f"R32_{region}_{game_num}",
            next_round_slot=next_slot,
            team1=r64_winners[idx],
            team2=r64_winners[idx + 1],
        )
        rows.append(row)
        r32_winners.append(winner)

    s16_winners: list[TeamRecord] = []
    for idx in range(0, len(r32_winners), 2):
        if idx + 1 >= len(r32_winners):
            continue
        game_num = (idx // 2) + 1
        winner, row = _predict_game(
            model=model,
            feature_cols=feature_cols,
            round_name="S16",
            region=region,
            game_slot=f"S16_{region}_{game_num}",
            next_round_slot=f"E8_{region}",
            team1=r32_winners[idx],
            team2=r32_winners[idx + 1],
        )
        rows.append(row)
        s16_winners.append(winner)

    if len(s16_winners) >= 2:
        region_winner, row = _predict_game(
            model=model,
            feature_cols=feature_cols,
            round_name="E8",
            region=region,
            game_slot=f"E8_{region}",
            next_round_slot="F4",
            team1=s16_winners[0],
            team2=s16_winners[1],
        )
        rows.append(row)
    elif len(s16_winners) == 1:
        region_winner = s16_winners[0]
    elif len(r32_winners) == 1:
        region_winner = r32_winners[0]
    elif len(r64_winners) == 1:
        region_winner = r64_winners[0]
    else:
        raise ValueError(f"No winners produced in region {region}.")

    return rows, region_winner


def _determine_final_four_pairs(region_winners: dict[str, TeamRecord]) -> list[tuple[str, str]]:
    regions = sorted(region_winners.keys())
    if len(regions) < 2:
        return []
    if len(regions) == 2:
        return [(regions[0], regions[1])]

    if set(FINAL_FOUR_REGION_PAIRS[0] + FINAL_FOUR_REGION_PAIRS[1]).issubset(set(regions)):
        return FINAL_FOUR_REGION_PAIRS

    pairs: list[tuple[str, str]] = []
    for idx in range(0, len(regions), 2):
        if idx + 1 < len(regions):
            pairs.append((regions[idx], regions[idx + 1]))
    return pairs


def _build_summary_text(bracket_df: pd.DataFrame) -> str:
    sections = ["R64", "R32", "S16", "E8", "F4", "CHAMP"]
    lines = []
    for round_name in sections:
        round_rows = bracket_df[bracket_df["round"] == round_name]
        if round_rows.empty:
            continue
        lines.append(f"{ROUND_LABELS.get(round_name, round_name)} winners:")
        for _, row in round_rows.iterrows():
            lines.append(
                f"- {row['predicted_winner']} wins matchup: {row['team_1']} vs {row['team_2']} "
                f"({row['predicted_win_probability']:.3f})"
            )
        lines.append("")
    champ_rows = bracket_df[bracket_df["round"] == "CHAMP"]
    if not champ_rows.empty:
        champion = champ_rows.iloc[-1]["predicted_winner"]
        lines.append(f"Champion: {champion}")
    return "\n".join(lines).strip() + "\n"


def predict_bracket_2026_reduced(
    *,
    team_features_2026_path: Path = TEAM_FEATURES_2026_REDUCED_PATH,
    tourney_matchups_2026_path: Path = TOURNEY_MATCHUPS_2026_PATH,
    model_path: Path = MODEL_ARTIFACT_2026_PATH,
    feature_list_path: Path = FEATURE_LIST_2026_REDUCED_PATH,
    breakdown_output_path: Path = BRACKET_BREAKDOWN_2026_PATH,
    summary_output_path: Path = BRACKET_SUMMARY_2026_PATH,
    train_if_missing: bool = True,
) -> tuple[Path, Path]:
    if train_if_missing and (not model_path.exists() or not feature_list_path.exists()):
        train_and_predict_2026_reduced(
            model_output_path=model_path,
            feature_list_output_path=feature_list_path,
            team_features_2026_path=team_features_2026_path,
            tourney_matchups_2026_path=tourney_matchups_2026_path,
        )

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not feature_list_path.exists():
        raise FileNotFoundError(f"Feature list not found: {feature_list_path}")
    if not team_features_2026_path.exists():
        raise FileNotFoundError(f"2026 team features not found: {team_features_2026_path}")
    if not tourney_matchups_2026_path.exists():
        raise FileNotFoundError(f"2026 matchup file not found: {tourney_matchups_2026_path}")

    model = load_model(model_path)
    feature_cols = _load_feature_list(feature_list_path)
    teams_df = pd.read_csv(team_features_2026_path)
    matchups_df = pd.read_csv(tourney_matchups_2026_path)
    team_lookup = _build_team_lookup(teams_df)
    r64_by_region = _get_r64_by_region(matchups_df, team_lookup)

    all_rows: list[dict[str, object]] = []
    region_winners: dict[str, TeamRecord] = {}
    for region, games in r64_by_region.items():
        rows, winner = _simulate_region_rounds(
            model=model,
            feature_cols=feature_cols,
            region=region,
            r64_games=games,
        )
        all_rows.extend(rows)
        region_winners[region] = winner

    f4_winners: list[TeamRecord] = []
    for idx, (region_a, region_b) in enumerate(_determine_final_four_pairs(region_winners), start=1):
        winner, row = _predict_game(
            model=model,
            feature_cols=feature_cols,
            round_name="F4",
            region="FinalFour",
            game_slot=f"F4_{idx}",
            next_round_slot="CHAMP",
            team1=region_winners[region_a],
            team2=region_winners[region_b],
        )
        all_rows.append(row)
        f4_winners.append(winner)

    if len(f4_winners) >= 2:
        _, champ_row = _predict_game(
            model=model,
            feature_cols=feature_cols,
            round_name="CHAMP",
            region="National",
            game_slot="CHAMP",
            next_round_slot=None,
            team1=f4_winners[0],
            team2=f4_winners[1],
        )
        all_rows.append(champ_row)
    elif len(f4_winners) == 1:
        lone_winner = f4_winners[0]
        all_rows.append(
            {
                "round": "CHAMP",
                "round_label": ROUND_LABELS["CHAMP"],
                "region": "National",
                "game_slot": "CHAMP",
                "next_round_slot": "",
                "team_1": lone_winner.team_name,
                "team_2": lone_winner.team_name,
                "team_1_seed": lone_winner.seed,
                "team_2_seed": lone_winner.seed,
                "predicted_winner": lone_winner.team_name,
                "predicted_win_probability": 1.0,
                "team_1_win_probability": 1.0,
                "team_1_name_norm": lone_winner.team_name_norm,
                "team_2_name_norm": lone_winner.team_name_norm,
                "predicted_winner_name_norm": lone_winner.team_name_norm,
            }
        )
    elif len(region_winners) == 1:
        lone_winner = next(iter(region_winners.values()))
        all_rows.append(
            {
                "round": "CHAMP",
                "round_label": ROUND_LABELS["CHAMP"],
                "region": "National",
                "game_slot": "CHAMP",
                "next_round_slot": "",
                "team_1": lone_winner.team_name,
                "team_2": lone_winner.team_name,
                "team_1_seed": lone_winner.seed,
                "team_2_seed": lone_winner.seed,
                "predicted_winner": lone_winner.team_name,
                "predicted_win_probability": 1.0,
                "team_1_win_probability": 1.0,
                "team_1_name_norm": lone_winner.team_name_norm,
                "team_2_name_norm": lone_winner.team_name_norm,
                "predicted_winner_name_norm": lone_winner.team_name_norm,
            }
        )

    if not all_rows:
        raise ValueError("No bracket games were generated for 2026 reduced prediction.")

    bracket_df = pd.DataFrame(all_rows)
    order_map = {"R64": 1, "R32": 2, "S16": 3, "E8": 4, "F4": 5, "CHAMP": 6}
    bracket_df["round_order"] = bracket_df["round"].map(order_map).fillna(999)
    bracket_df = bracket_df.sort_values(["round_order", "region", "game_slot"]).drop(columns=["round_order"])

    ensure_parent(breakdown_output_path)
    bracket_df.to_csv(breakdown_output_path, index=False)

    summary_text = _build_summary_text(bracket_df)
    ensure_parent(summary_output_path)
    summary_output_path.write_text(summary_text, encoding="utf-8")

    print(f"[save] bracket breakdown: {breakdown_output_path} ({len(bracket_df)} games)")
    print(f"[save] bracket summary: {summary_output_path}")
    return breakdown_output_path, summary_output_path


def main() -> None:
    args = parse_args()
    train_if_missing = args.train_if_missing and not args.no_train_if_missing
    predict_bracket_2026_reduced(
        team_features_2026_path=args.team_features_2026_path,
        tourney_matchups_2026_path=args.tourney_matchups_2026_path,
        model_path=args.model_path,
        feature_list_path=args.feature_list_path,
        breakdown_output_path=args.breakdown_output_path,
        summary_output_path=args.summary_output_path,
        train_if_missing=train_if_missing,
    )


if __name__ == "__main__":
    main()
