from pathlib import Path
import pandas as pd
import re
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent          # .../MMM/data/processed
DATA_DIR = SCRIPT_DIR.parent                          # .../MMM/data
RAW_DIR = DATA_DIR / "raw"
HIST_DIR = RAW_DIR / "MarchMadnessHistoricalDataSet"
KAGGLE_DIR = RAW_DIR / "march-machine-learning-mania-2026"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# MAIN OUTPUT FILES
# -----------------------------
TEAM_PROFILES_OUT = PROCESSED_DIR / "team_profiles.csv"
GAMES_BOXSCORES_OUT = PROCESSED_DIR / "games_boxscores.csv"
TOURNEY_MATCHUPS_OUT = PROCESSED_DIR / "tourney_matchups.csv"

# -----------------------------
# INPUT FILES
# -----------------------------
RAW_FILES = {
    # team profiles
    "dev": HIST_DIR / "DEV _ March Madness.csv",
    "m_regular_compact": KAGGLE_DIR / "MRegularSeasonCompactResults.csv",

    # games / boxscores
    "mbb_schedule": RAW_DIR / "mbb_schedule_2002_2026.csv",
    "mbb_team_boxscores": RAW_DIR / "mbb_team_boxscore_2002_2026.csv",

    # tournament matchups
    "m_tourney_detailed": KAGGLE_DIR / "MNCAATourneyDetailedResults.csv",
    "m_tourney_seed_round_slot": KAGGLE_DIR / "MNCAATourneySeedRoundSlots.csv",
    "m_tourney_seeds": KAGGLE_DIR / "MNCAATourneySeeds.csv",
    "m_tourney_slots": KAGGLE_DIR / "MNCAATourneySlots.csv",
    "m_teams": KAGGLE_DIR / "MTeams.csv",
}

# -----------------------------
# HELPERS
# -----------------------------
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)

def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"Saved: {path} | shape={df.shape}")

def norm_col(col_name: str) -> str:
    return (
        col_name.strip()
        .lower()
        .replace(".", "_")
        .replace("%", "pct")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("__", "_")
    )

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    return df

def normalize_team_name(value) -> str:
    if pd.isna(value):
        return pd.NA
    s = str(value).strip().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else pd.NA

def normalize_team_name_series(series: pd.Series) -> pd.Series:
    return series.apply(normalize_team_name)

def parse_seed_number(seed_value):
    if pd.isna(seed_value):
        return pd.NA
    digits = "".join(ch for ch in str(seed_value) if ch.isdigit())
    return int(digits) if digits else pd.NA

# -----------------------------
# BUILDERS
# -----------------------------
def build_team_mapping(raw_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    dev_df = standardize_columns(raw_data.get("dev", pd.DataFrame()).copy())
    teams_df = standardize_columns(raw_data.get("m_teams", pd.DataFrame()).copy())
    box_df = standardize_columns(raw_data.get("mbb_team_boxscores", pd.DataFrame()).copy())

    def make_alias_df(df: pd.DataFrame, alias_cols: list[str], id_col: str | None, source: str) -> pd.DataFrame:
        parts = []

        for col in alias_cols:
            if col not in df.columns:
                continue

            use_cols = [col]
            if id_col and id_col in df.columns:
                use_cols = [id_col, col]

            temp = df[use_cols].dropna().drop_duplicates().copy()
            temp = temp.rename(columns={col: "raw_team_name"})

            if id_col and id_col in temp.columns:
                temp = temp.rename(columns={id_col: "source_team_id"})
            else:
                temp["source_team_id"] = pd.NA

            temp["source"] = source
            temp["team_name_norm"] = normalize_team_name_series(temp["raw_team_name"])
            temp = temp.dropna(subset=["team_name_norm"])

            parts.append(temp[["source", "source_team_id", "raw_team_name", "team_name_norm"]])

        if not parts:
            return pd.DataFrame(columns=["source", "source_team_id", "raw_team_name", "team_name_norm"])

        return pd.concat(parts, ignore_index=True).drop_duplicates()

    kaggle_aliases = make_alias_df(
        teams_df,
        alias_cols=["teamname"],
        id_col="teamid",
        source="kaggle",
    ).rename(columns={
        "source_team_id": "kaggle_team_id",
        "raw_team_name": "kaggle_team_name",
    })

    dev_aliases = make_alias_df(
        dev_df,
        alias_cols=["mapped_espn_team_name", "full_team_name"],
        id_col=None,
        source="dev",
    ).rename(columns={
        "raw_team_name": "dev_team_name",
    })

    espn_aliases = make_alias_df(
        box_df,
        alias_cols=["team_display_name", "team_short_display_name", "team_name"],
        id_col="team_id",
        source="espn",
    ).rename(columns={
        "source_team_id": "espn_team_id",
        "raw_team_name": "espn_team_name",
    })

    all_norms = pd.concat(
        [
            kaggle_aliases[["team_name_norm"]],
            dev_aliases[["team_name_norm"]],
            espn_aliases[["team_name_norm"]],
        ],
        ignore_index=True,
    ).drop_duplicates()

    kaggle_first = (
        kaggle_aliases.sort_values(["team_name_norm", "kaggle_team_name"])
        .drop_duplicates(subset=["team_name_norm"], keep="first")
    )

    dev_first = (
        dev_aliases.sort_values(["team_name_norm", "dev_team_name"])
        .drop_duplicates(subset=["team_name_norm"], keep="first")
    )

    espn_first = (
        espn_aliases.sort_values(["team_name_norm", "espn_team_name"])
        .drop_duplicates(subset=["team_name_norm"], keep="first")
    )

    team_mapping = all_norms.merge(
        kaggle_first[["team_name_norm", "kaggle_team_id", "kaggle_team_name"]],
        on="team_name_norm",
        how="left",
    ).merge(
        dev_first[["team_name_norm", "dev_team_name"]],
        on="team_name_norm",
        how="left",
    ).merge(
        espn_first[["team_name_norm", "espn_team_id", "espn_team_name"]],
        on="team_name_norm",
        how="left",
    )

    team_mapping["canonical_team_name"] = (
        team_mapping["kaggle_team_name"]
        .combine_first(team_mapping["dev_team_name"])
        .combine_first(team_mapping["espn_team_name"])
    )

    return team_mapping.sort_values("canonical_team_name", na_position="last").reset_index(drop=True)

def build_regular_season_resume_stats(
    raw_data: dict[str, pd.DataFrame],
    team_mapping: pd.DataFrame,
    base_elo: float = 1500.0,
    k_factor: float = 20.0,
    home_advantage: float = 60.0,
) -> pd.DataFrame:
    games_df = standardize_columns(raw_data["m_regular_compact"].copy())
    teams_df = standardize_columns(raw_data["m_teams"].copy()) if "m_teams" in raw_data else pd.DataFrame()
    dev_df = standardize_columns(raw_data["dev"].copy()) if "dev" in raw_data else pd.DataFrame()
    team_mapping = team_mapping.copy()

    # -----------------------------
    # Build conference map by season + kaggle_team_id
    # -----------------------------
    conference_map = pd.DataFrame(columns=["season", "kaggle_team_id", "conference"])

    if not dev_df.empty:
        dev_conf_cols = []
        if "season" in dev_df.columns:
            dev_conf_cols.append("season")
        if "mapped_espn_team_name" in dev_df.columns:
            dev_conf_cols.append("mapped_espn_team_name")
        if "mapped_conference_name" in dev_df.columns:
            dev_conf_cols.append("mapped_conference_name")

        if len(dev_conf_cols) == 3:
            conf_df = dev_df[dev_conf_cols].drop_duplicates().copy()
            conf_df["team_name_norm"] = normalize_team_name_series(conf_df["mapped_espn_team_name"])

            mapping_cols = ["team_name_norm", "kaggle_team_id"]
            conf_df = conf_df.merge(
                team_mapping[mapping_cols].drop_duplicates(),
                on="team_name_norm",
                how="left",
            )

            conference_map = conf_df.rename(columns={
                "mapped_conference_name": "conference"
            })[["season", "kaggle_team_id", "conference"]].drop_duplicates()

    # -----------------------------
    # Keep needed game columns
    # -----------------------------
    keep_cols = ["season", "daynum", "wteamid", "wscore", "lteamid", "lscore", "wloc", "numot"]
    keep_cols = [c for c in keep_cols if c in games_df.columns]
    games_df = games_df[keep_cols].copy()
    games_df = games_df.sort_values(["season", "daynum"]).reset_index(drop=True)

    def expected_score(rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def init_state():
        return {
            "games": 0,
            "wins": 0,
            "losses": 0,
            "home_wins": 0,
            "home_losses": 0,
            "away_wins": 0,
            "away_losses": 0,
            "neutral_wins": 0,
            "neutral_losses": 0,
            "conference_wins": 0,
            "conference_losses": 0,
            "nonconf_wins": 0,
            "nonconf_losses": 0,
            "ot_games": 0,
            "score_for_sum": 0,
            "score_against_sum": 0,
            "margin_sum": 0,
            "opp_pregame_elo_sum": 0.0,
            "expected_wins_sum": 0.0,
            "elo_pre_tourney": base_elo,
            "top25_wins": 0,
            "top50_wins": 0,
            "bad_losses_100plus": 0,
        }

    ratings = defaultdict(lambda: base_elo)
    states = {}
    current_season = None

    # Save per-game pregame info so we can do second-pass resume stats
    game_log_rows = []

    def get_state(season: int, team_id: int):
        key = (season, team_id)
        if key not in states:
            states[key] = init_state()
        return states[key]

    # Conference lookup dict
    conf_lookup = {}
    if not conference_map.empty:
        for _, row in conference_map.dropna(subset=["season", "kaggle_team_id"]).iterrows():
            conf_lookup[(int(row["season"]), int(row["kaggle_team_id"]))] = row["conference"]

    # -----------------------------
    # Pass 1: Elo + base season stats
    # -----------------------------
    for _, row in games_df.iterrows():
        season = int(row["season"])

        if current_season is None or season != current_season:
            ratings = defaultdict(lambda: base_elo)
            current_season = season

        wteam = int(row["wteamid"])
        lteam = int(row["lteamid"])
        wscore = int(row["wscore"])
        lscore = int(row["lscore"])
        wloc = str(row["wloc"]).upper() if pd.notna(row["wloc"]) else "N"
        numot = int(row["numot"]) if pd.notna(row["numot"]) else 0

        winner_pre = ratings[wteam]
        loser_pre = ratings[lteam]

        if wloc == "H":
            winner_bonus = home_advantage
            loser_bonus = -home_advantage
        elif wloc == "A":
            winner_bonus = -home_advantage
            loser_bonus = home_advantage
        else:
            winner_bonus = 0.0
            loser_bonus = 0.0

        winner_expected = expected_score(winner_pre + winner_bonus, loser_pre + loser_bonus)
        loser_expected = 1.0 - winner_expected

        winner_post = winner_pre + k_factor * (1.0 - winner_expected)
        loser_post = loser_pre + k_factor * (0.0 - loser_expected)

        ratings[wteam] = winner_post
        ratings[lteam] = loser_post

        w_state = get_state(season, wteam)
        l_state = get_state(season, lteam)

        # winner updates
        w_state["games"] += 1
        w_state["wins"] += 1
        w_state["score_for_sum"] += wscore
        w_state["score_against_sum"] += lscore
        w_state["margin_sum"] += (wscore - lscore)
        w_state["opp_pregame_elo_sum"] += loser_pre
        w_state["expected_wins_sum"] += winner_expected
        w_state["elo_pre_tourney"] = winner_post

        # loser updates
        l_state["games"] += 1
        l_state["losses"] += 1
        l_state["score_for_sum"] += lscore
        l_state["score_against_sum"] += wscore
        l_state["margin_sum"] += (lscore - wscore)
        l_state["opp_pregame_elo_sum"] += winner_pre
        l_state["expected_wins_sum"] += loser_expected
        l_state["elo_pre_tourney"] = loser_post

        if numot > 0:
            w_state["ot_games"] += 1
            l_state["ot_games"] += 1

        if wloc == "H":
            w_state["home_wins"] += 1
            l_state["away_losses"] += 1
        elif wloc == "A":
            w_state["away_wins"] += 1
            l_state["home_losses"] += 1
        else:
            w_state["neutral_wins"] += 1
            l_state["neutral_losses"] += 1

        winner_conf = conf_lookup.get((season, wteam), pd.NA)
        loser_conf = conf_lookup.get((season, lteam), pd.NA)

        same_conf = pd.notna(winner_conf) and pd.notna(loser_conf) and winner_conf == loser_conf
        if same_conf:
            w_state["conference_wins"] += 1
            l_state["conference_losses"] += 1
        else:
            w_state["nonconf_wins"] += 1
            l_state["nonconf_losses"] += 1

        game_log_rows.append({
            "season": season,
            "winner_team_id": wteam,
            "loser_team_id": lteam,
            "winner_pregame_elo": winner_pre,
            "loser_pregame_elo": loser_pre,
            "winner_expected": winner_expected,
            "loser_expected": loser_expected,
        })

    # -----------------------------
    # Final Elo rank per season
    # -----------------------------
    rows = []
    for (season, team_id), s in states.items():
        games = s["games"]
        rows.append({
            "season": season,
            "kaggle_team_id": team_id,
            "wins": s["wins"],
            "losses": s["losses"],
            "win_pct": s["wins"] / games if games else pd.NA,
            "home_wins": s["home_wins"],
            "home_losses": s["home_losses"],
            "away_wins": s["away_wins"],
            "away_losses": s["away_losses"],
            "neutral_wins": s["neutral_wins"],
            "neutral_losses": s["neutral_losses"],
            "conference_wins": s["conference_wins"],
            "conference_losses": s["conference_losses"],
            "nonconf_wins": s["nonconf_wins"],
            "nonconf_losses": s["nonconf_losses"],
            "ot_games": s["ot_games"],
            "avg_score_for": s["score_for_sum"] / games if games else pd.NA,
            "avg_score_against": s["score_against_sum"] / games if games else pd.NA,
            "avg_margin": s["margin_sum"] / games if games else pd.NA,
            "avg_opp_pregame_elo": s["opp_pregame_elo_sum"] / games if games else pd.NA,
            "expected_wins": s["expected_wins_sum"],
            "elo_pre_tourney": s["elo_pre_tourney"],
        })

    resume_df = pd.DataFrame(rows)

    if not resume_df.empty:
        resume_df["elo_rank"] = resume_df.groupby("season")["elo_pre_tourney"].rank(
            ascending=False, method="min"
        )

    # -----------------------------
    # Pass 2: top wins / bad losses using final Elo ranks
    # -----------------------------
    if not resume_df.empty:
        final_rank_lookup = {
            (int(r["season"]), int(r["kaggle_team_id"])): int(r["elo_rank"])
            for _, r in resume_df.dropna(subset=["elo_rank"]).iterrows()
        }

        top25_wins = defaultdict(int)
        top50_wins = defaultdict(int)
        bad_losses_100plus = defaultdict(int)

        for g in game_log_rows:
            season = g["season"]
            wteam = g["winner_team_id"]
            lteam = g["loser_team_id"]

            loser_rank = final_rank_lookup.get((season, lteam))
            winner_rank = final_rank_lookup.get((season, wteam))

            if loser_rank is not None and loser_rank <= 25:
                top25_wins[(season, wteam)] += 1
            if loser_rank is not None and loser_rank <= 50:
                top50_wins[(season, wteam)] += 1
            if winner_rank is not None and winner_rank > 100:
                bad_losses_100plus[(season, lteam)] += 1

        resume_df["top25_wins"] = resume_df.apply(
            lambda r: top25_wins.get((int(r["season"]), int(r["kaggle_team_id"])), 0),
            axis=1,
        )
        resume_df["top50_wins"] = resume_df.apply(
            lambda r: top50_wins.get((int(r["season"]), int(r["kaggle_team_id"])), 0),
            axis=1,
        )
        resume_df["bad_losses_100plus"] = resume_df.apply(
            lambda r: bad_losses_100plus.get((int(r["season"]), int(r["kaggle_team_id"])), 0),
            axis=1,
        )

        resume_df["resume_delta"] = resume_df["wins"] - resume_df["expected_wins"]
        resume_df["sos_elo"] = resume_df["avg_opp_pregame_elo"]

    if not teams_df.empty and {"teamid", "teamname"}.issubset(teams_df.columns):
        team_lookup = teams_df[["teamid", "teamname"]].drop_duplicates().rename(
            columns={"teamid": "kaggle_team_id", "teamname": "kaggle_team_name"}
        )
        resume_df = resume_df.merge(team_lookup, on="kaggle_team_id", how="left")

    return resume_df

def build_team_profiles(raw_data: dict[str, pd.DataFrame], team_mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Build team_profiles.csv

    One row per team-season.
    """
    DEV_KEEP_COLS_RAW = [
        "Season",
        "Mapped Conference Name",
        "Mapped ESPN Team Name",
        "Full Team Name",
        "Current Coach",
        "Since",
        "Active Coaching Length",
        "Seed",
        "Region",
        "Pre-Tournament.Tempo",
        "Pre-Tournament.AdjTempo",
        "Pre-Tournament.AdjOE",
        "Pre-Tournament.AdjDE",
        "Pre-Tournament.AdjEM",
        "Avg Possession Length (Offense)",
        "Avg Possession Length (Defense)",
        "eFGPct",
        "TOPct",
        "ORPct",
        "FTRate",
        "FG2Pct",
        "FG3Pct",
        "FTPct",
        "BlockPct",
        "OppFG2Pct",
        "OppFG3Pct",
        "OppFTPct",
        "OppBlockPct",
        "FG3Rate",
        "OppFG3Rate",
        "ARate",
        "OppARate",
        "StlRate",
        "OppStlRate",
        "NSTRate",
        "OppNSTRate",
        "AvgHeight",
        "EffectiveHeight",
        "Experience",
        "Bench",
        "Net Rating",
    ]

    dev_df = raw_data["dev"].copy()
    dev_df = standardize_columns(dev_df)

    # get team_profile attribute stats
    dev_keep_cols = [norm_col(c) for c in DEV_KEEP_COLS_RAW]
    dev_keep_cols = [c for c in dev_keep_cols if c in dev_df.columns]

    team_profiles = dev_df[dev_keep_cols].copy()

    team_profiles = team_profiles.rename(columns={
        norm_col("Mapped ESPN Team Name"): "team_name",
        norm_col("Mapped Conference Name"): "conference",
        norm_col("Full Team Name"): "full_team_name",
        norm_col("Pre-Tournament.Tempo"): "pre_tourney_tempo",
        norm_col("Pre-Tournament.AdjTempo"): "pre_tourney_adjtempo",
        norm_col("Pre-Tournament.AdjOE"): "pre_tourney_adjoe",
        norm_col("Pre-Tournament.AdjDE"): "pre_tourney_adjde",
        norm_col("Pre-Tournament.AdjEM"): "pre_tourney_adjem",
        norm_col("Avg Possession Length (Offense)"): "avg_possession_length_offense",
        norm_col("Avg Possession Length (Defense)"): "avg_possession_length_defense",
        norm_col("Net Rating"): "net_rating",
    })

    # get season resume stats
    # Normalize names and map to canonical / Kaggle ID
    if "team_name" in team_profiles.columns:
        team_profiles["team_name_norm"] = normalize_team_name_series(team_profiles["team_name"])

    mapping_cols = [
        "team_name_norm",
        "canonical_team_name",
        "kaggle_team_id",
        "kaggle_team_name",
        "espn_team_id",
    ]
    mapping_cols = [c for c in mapping_cols if c in team_mapping.columns]

    team_profiles = team_profiles.merge(
        team_mapping[mapping_cols].drop_duplicates(),
        on="team_name_norm",
        how="left",
    )

    team_profiles["canonical_team_name"] = team_profiles["canonical_team_name"].combine_first(team_profiles["team_name"])

    # Build and merge resume / Elo stats
    resume_stats = build_regular_season_resume_stats(raw_data, team_mapping)

    if not resume_stats.empty:
        team_profiles = team_profiles.merge(
            resume_stats,
            on=["season", "kaggle_team_id"],
            how="left",
            suffixes=("", "_resume"),
        )

    # Optional cleanup
    if "canonical_team_name" in team_profiles.columns:
        team_profiles = team_profiles.sort_values(["season", "canonical_team_name"], na_position="last")

    team_profiles = team_profiles.drop_duplicates(subset=["season", "team_name"], keep="first").reset_index(drop=True)
    return team_profiles

def build_games_boxscores(raw_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build games_boxscores.csv

    One row per game or team-game.
    """
    TEAM_BOX_KEEP_COLS_RAW = [
        "game_id",
        "season",
        "season_type",
        "game_date",
        "game_date_time",
        "team_id",
        "team_name",
        "team_display_name",
        "team_short_display_name",
        "team_home_away",
        "team_score",
        "team_winner",
        "opponent_team_id",
        "opponent_team_name",
        "opponent_team_display_name",
        "opponent_team_short_display_name",
        "opponent_team_score",
        "assists",
        "blocks",
        "defensive_rebounds",
        "field_goal_pct",
        "field_goals_made",
        "field_goals_attempted",
        "free_throw_pct",
        "free_throws_made",
        "free_throws_attempted",
        "offensive_rebounds",
        "steals",
        "three_point_field_goal_pct",
        "three_point_field_goals_made",
        "three_point_field_goals_attempted",
        "total_rebounds",
        "turnovers",
        "largest_lead",
        "fast_break_points",
        "points_in_paint",
        "turnover_points",
        "lead_changes",
        "lead_percentage",
    ]
    SCHEDULE_KEEP_COLS_RAW = [
        "game_id",
        "season",
        "season_type",
        "date",
        "game_date",
        "game_date_time",
        "neutral_site",
        "conference_competition",
        "attendance",
        "venue_full_name",
        "venue_address_city",
        "venue_address_state",
        "status_type_completed",
        "home_id",
        "home_display_name",
        "home_score",
        "away_id",
        "away_display_name",
        "away_score",
        "home_current_rank",
        "away_current_rank",
        "tournament_id",
    ]

    team_box_df = raw_data["mbb_team_boxscores"].copy()
    schedule_df = raw_data["mbb_schedule"].copy()

    team_box_df = standardize_columns(team_box_df)
    schedule_df = standardize_columns(schedule_df)

    team_box_keep = [norm_col(c) for c in TEAM_BOX_KEEP_COLS_RAW]
    schedule_keep = [norm_col(c) for c in SCHEDULE_KEEP_COLS_RAW]

    team_box_keep = [c for c in team_box_keep if c in team_box_df.columns]
    schedule_keep = [c for c in schedule_keep if c in schedule_df.columns]

    team_box_df = team_box_df[team_box_keep].copy()
    schedule_df = schedule_df[schedule_keep].copy()

    games_boxscores = team_box_df.merge(
        schedule_df,
        on=["game_id", "season", "season_type"],
        how="left",
        suffixes=("", "_sched"),
    )

    if "team_score" in games_boxscores.columns and "opponent_team_score" in games_boxscores.columns:
        games_boxscores["score_margin"] = (
            games_boxscores["team_score"] - games_boxscores["opponent_team_score"]
        )

    if "team_winner" in games_boxscores.columns:
        games_boxscores["win"] = games_boxscores["team_winner"].astype("Int64")

    if "team_home_away" in games_boxscores.columns:
        games_boxscores["is_home"] = (games_boxscores["team_home_away"].astype(str).str.lower() == "home").astype(int)
        games_boxscores["is_away"] = (games_boxscores["team_home_away"].astype(str).str.lower() == "away").astype(int)
        games_boxscores["is_neutral"] = (games_boxscores["team_home_away"].astype(str).str.lower() == "neutral").astype(int)

    return games_boxscores

def build_tourney_matchups(raw_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    results_df = raw_data["m_tourney_detailed"].copy()
    seeds_df = raw_data["m_tourney_seeds"].copy()
    teams_df = raw_data["m_teams"].copy()

    results_df = standardize_columns(results_df)
    seeds_df = standardize_columns(seeds_df)
    teams_df = standardize_columns(teams_df)

    # -----------------------------
    # Keep only needed columns
    # -----------------------------
    results_keep = [
        "season", "daynum", "wteamid", "wscore", "lteamid", "lscore", "wloc", "numot",
        "wfgm", "wfga", "wfgm3", "wfga3", "wftm", "wfta", "wor", "wdr", "wast", "wto", "wstl", "wblk", "wpf",
        "lfgm", "lfga", "lfgm3", "lfga3", "lftm", "lfta", "lor", "ldr", "last", "lto", "lstl", "lblk", "lpf",
    ]
    seeds_keep = ["season", "seed", "teamid"]
    teams_keep = ["teamid", "teamname"]

    results_keep = [c for c in results_keep if c in results_df.columns]
    seeds_keep = [c for c in seeds_keep if c in seeds_df.columns]
    teams_keep = [c for c in teams_keep if c in teams_df.columns]

    results_df = results_df[results_keep].copy()
    seeds_df = seeds_df[seeds_keep].copy()
    teams_df = teams_df[teams_keep].copy()

    # -----------------------------
    # Team + seed lookup tables
    # -----------------------------
    team_lookup = teams_df.rename(columns={
        "teamid": "team_id",
        "teamname": "team_name",
    })

    seed_lookup = seeds_df.rename(columns={
        "teamid": "team_id",
        "seed": "seed",
    })

    # -----------------------------
    # Winner-perspective rows
    # -----------------------------
    winner_rows = pd.DataFrame({
        "season": results_df["season"],
        "daynum": results_df["daynum"],
        "team_a_id": results_df["wteamid"],
        "team_b_id": results_df["lteamid"],
        "team_a_score": results_df["wscore"],
        "team_b_score": results_df["lscore"],
        "wloc": results_df["wloc"],
        "numot": results_df["numot"],
        "target": 1,

        "team_a_fgm": results_df["wfgm"],
        "team_a_fga": results_df["wfga"],
        "team_a_fgm3": results_df["wfgm3"],
        "team_a_fga3": results_df["wfga3"],
        "team_a_ftm": results_df["wftm"],
        "team_a_fta": results_df["wfta"],
        "team_a_or": results_df["wor"],
        "team_a_dr": results_df["wdr"],
        "team_a_ast": results_df["wast"],
        "team_a_to": results_df["wto"],
        "team_a_stl": results_df["wstl"],
        "team_a_blk": results_df["wblk"],
        "team_a_pf": results_df["wpf"],

        "team_b_fgm": results_df["lfgm"],
        "team_b_fga": results_df["lfga"],
        "team_b_fgm3": results_df["lfgm3"],
        "team_b_fga3": results_df["lfga3"],
        "team_b_ftm": results_df["lftm"],
        "team_b_fta": results_df["lfta"],
        "team_b_or": results_df["lor"],
        "team_b_dr": results_df["ldr"],
        "team_b_ast": results_df["last"],
        "team_b_to": results_df["lto"],
        "team_b_stl": results_df["lstl"],
        "team_b_blk": results_df["lblk"],
        "team_b_pf": results_df["lpf"],
    })

    # -----------------------------
    # Loser-perspective rows
    # -----------------------------
    loser_rows = pd.DataFrame({
        "season": results_df["season"],
        "daynum": results_df["daynum"],
        "team_a_id": results_df["lteamid"],
        "team_b_id": results_df["wteamid"],
        "team_a_score": results_df["lscore"],
        "team_b_score": results_df["wscore"],
        "wloc": results_df["wloc"],
        "numot": results_df["numot"],
        "target": 0,

        "team_a_fgm": results_df["lfgm"],
        "team_a_fga": results_df["lfga"],
        "team_a_fgm3": results_df["lfgm3"],
        "team_a_fga3": results_df["lfga3"],
        "team_a_ftm": results_df["lftm"],
        "team_a_fta": results_df["lfta"],
        "team_a_or": results_df["lor"],
        "team_a_dr": results_df["ldr"],
        "team_a_ast": results_df["last"],
        "team_a_to": results_df["lto"],
        "team_a_stl": results_df["lstl"],
        "team_a_blk": results_df["lblk"],
        "team_a_pf": results_df["lpf"],

        "team_b_fgm": results_df["wfgm"],
        "team_b_fga": results_df["wfga"],
        "team_b_fgm3": results_df["wfgm3"],
        "team_b_fga3": results_df["wfga3"],
        "team_b_ftm": results_df["wftm"],
        "team_b_fta": results_df["wfta"],
        "team_b_or": results_df["wor"],
        "team_b_dr": results_df["wdr"],
        "team_b_ast": results_df["wast"],
        "team_b_to": results_df["wto"],
        "team_b_stl": results_df["wstl"],
        "team_b_blk": results_df["wblk"],
        "team_b_pf": results_df["wpf"],
    })

    tourney_matchups = pd.concat([winner_rows, loser_rows], ignore_index=True)

    # -----------------------------
    # Team names
    # -----------------------------
    tourney_matchups = tourney_matchups.merge(
        team_lookup.rename(columns={
            "team_id": "team_a_id",
            "team_name": "team_a_name",
        }),
        on="team_a_id",
        how="left",
    )

    tourney_matchups = tourney_matchups.merge(
        team_lookup.rename(columns={
            "team_id": "team_b_id",
            "team_name": "team_b_name",
        }),
        on="team_b_id",
        how="left",
    )

    # -----------------------------
    # Seeds
    # -----------------------------
    tourney_matchups = tourney_matchups.merge(
        seed_lookup.rename(columns={
            "team_id": "team_a_id",
            "seed": "team_a_seed",
        }),
        on=["season", "team_a_id"],
        how="left",
    )

    tourney_matchups = tourney_matchups.merge(
        seed_lookup.rename(columns={
            "team_id": "team_b_id",
            "seed": "team_b_seed",
        }),
        on=["season", "team_b_id"],
        how="left",
    )

    # -----------------------------
    # Derived fields
    # -----------------------------
    tourney_matchups["score_margin"] = (
        tourney_matchups["team_a_score"] - tourney_matchups["team_b_score"]
    )

    tourney_matchups["team_a_seed_num"] = tourney_matchups["team_a_seed"].apply(parse_seed_number)
    tourney_matchups["team_b_seed_num"] = tourney_matchups["team_b_seed"].apply(parse_seed_number)

    # Convert winner-location into team A perspective
    # WLoc is winner's location: H/A/N
    # For winner rows, team A is winner
    # For loser rows, team A is loser, so H/A flips
    def convert_team_a_location(row):
        wloc = row["wloc"]
        target = row["target"]

        if pd.isna(wloc):
            return pd.NA
        wloc = str(wloc).upper()

        if wloc == "N":
            return "N"
        if target == 1:
            return wloc
        # loser perspective: flip H/A
        if wloc == "H":
            return "A"
        if wloc == "A":
            return "H"
        return pd.NA

    tourney_matchups["team_a_loc"] = tourney_matchups.apply(convert_team_a_location, axis=1)
    tourney_matchups["is_neutral"] = (tourney_matchups["team_a_loc"] == "N").astype(int)
    tourney_matchups["is_home"] = (tourney_matchups["team_a_loc"] == "H").astype(int)
    tourney_matchups["is_away"] = (tourney_matchups["team_a_loc"] == "A").astype(int)

    # Simple round guess from sorted DayNum within each season
    unique_days = (
        tourney_matchups[["season", "daynum"]]
        .drop_duplicates()
        .sort_values(["season", "daynum"])
        .copy()
    )
    unique_days["round_num_guess"] = unique_days.groupby("season").cumcount() + 1

    tourney_matchups = tourney_matchups.merge(
        unique_days,
        on=["season", "daynum"],
        how="left",
    )

    # Optional simple seed diff
    tourney_matchups["seed_diff"] = (
        pd.to_numeric(tourney_matchups["team_a_seed_num"], errors="coerce")
        - pd.to_numeric(tourney_matchups["team_b_seed_num"], errors="coerce")
    )

    # Final ordering
    preferred_order = [
        "season",
        "daynum",
        "round_num_guess",
        "team_a_id",
        "team_a_name",
        "team_a_seed",
        "team_a_seed_num",
        "team_b_id",
        "team_b_name",
        "team_b_seed",
        "team_b_seed_num",
        "team_a_score",
        "team_b_score",
        "score_margin",
        "seed_diff",
        "team_a_loc",
        "is_home",
        "is_away",
        "is_neutral",
        "numot",
        "target",
    ]
    existing_front = [c for c in preferred_order if c in tourney_matchups.columns]
    remaining = [c for c in tourney_matchups.columns if c not in existing_front]
    tourney_matchups = tourney_matchups[existing_front + remaining]

    return tourney_matchups

# -----------------------------
# MAIN
# -----------------------------
def main():
    raw_data = {}

    for name, path in RAW_FILES.items():
        print(f"Loading {name}: {path}")
        raw_data[name] = load_csv(path)

    team_mapping = build_team_mapping(raw_data)
    save_csv(team_mapping, PROCESSED_DIR / "team_mapping.csv")

    team_profiles = build_team_profiles(raw_data, team_mapping)
    games_boxscores = build_games_boxscores(raw_data)
    tourney_matchups = build_tourney_matchups(raw_data)

    save_csv(team_profiles, TEAM_PROFILES_OUT)
    save_csv(games_boxscores, GAMES_BOXSCORES_OUT)
    save_csv(tourney_matchups, TOURNEY_MATCHUPS_OUT)

if __name__ == "__main__":
    main()