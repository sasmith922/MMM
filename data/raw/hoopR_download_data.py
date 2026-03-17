from pathlib import Path
import pandas as pd
from sportsdataverse import mbb

# -----------------------------
# CONFIG
# -----------------------------
START_SEASON = 2002
END_SEASON = 2026  # inclusive
SAVE_PLAYER_BOX = False
SAVE_PLAY_BY_PLAY = False   # keep False for now
SAVE_CALENDAR = False       # can turn on later

SEASONS = list(range(START_SEASON, END_SEASON + 1))
SEASON_TAG = f"{START_SEASON}_{END_SEASON}"

# Save directly in current folder
SAVE_DIR = Path(".")

print(f"Pulling men's college basketball data for seasons {START_SEASON}-{END_SEASON}...")

# -----------------------------
# HELPER
# -----------------------------
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make known problematic columns consistent across seasons.
    """
    if "game_json_url" in df.columns:
        df["game_json_url"] = df["game_json_url"].astype("string")
    return df

def load_seasonal_data(loader_func, name: str, seasons: list[int]) -> pd.DataFrame:
    """
    Load one season at a time, normalize columns, and concatenate safely.
    """
    frames = []

    for season in seasons:
        try:
            print(f"Downloading {name} for {season}...")
            df = loader_func(seasons=[season], return_as_pandas=True)
            df = normalize_df(df)
            df["season_downloaded"] = season
            frames.append(df)
            print(f"  success: {season} | shape={df.shape}")
        except Exception as e:
            print(f"  skipped {season} because of error: {e}")

    if not frames:
        print(f"No data downloaded for {name}.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    print(f"Combined {name} shape: {combined.shape}")
    return combined

# -----------------------------
# TEAMS
# -----------------------------
try:
    print("Downloading teams...")
    teams_df = mbb.espn_mbb_teams(return_as_pandas=True)
    teams_df.to_csv(SAVE_DIR / f"mbb_teams_{SEASON_TAG}.csv", index=False)
    print(f"Saved teams | shape={teams_df.shape}")
except Exception as e:
    print(f"Could not download teams: {e}")

# -----------------------------
# SCHEDULE
# -----------------------------
schedule_df = load_seasonal_data(mbb.load_mbb_schedule, "schedule", SEASONS)
if not schedule_df.empty:
    schedule_df.to_csv(SAVE_DIR / f"mbb_schedule_{SEASON_TAG}.csv", index=False)
    print("Saved schedule CSV.")

# -----------------------------
# TEAM BOXSCORE
# -----------------------------
team_box_df = load_seasonal_data(mbb.load_mbb_team_boxscore, "team_boxscore", SEASONS)
if not team_box_df.empty:
    team_box_df.to_csv(SAVE_DIR / f"mbb_team_boxscore_{SEASON_TAG}.csv", index=False)
    print("Saved team boxscore CSV.")

# -----------------------------
# PLAYER BOXSCORE
# -----------------------------
if SAVE_PLAYER_BOX:
    player_box_df = load_seasonal_data(mbb.load_mbb_player_boxscore, "player_boxscore", SEASONS)
    if not player_box_df.empty:
        player_box_df.to_csv(SAVE_DIR / f"mbb_player_boxscore_{SEASON_TAG}.csv", index=False)
        print("Saved player boxscore CSV.")

# -----------------------------
# CALENDAR
# -----------------------------
if SAVE_CALENDAR:
    for season in SEASONS:
        try:
            print(f"Downloading calendar for {season}...")
            cal_df = mbb.espn_mbb_calendar(season=season, return_as_pandas=True)
            cal_df.to_csv(SAVE_DIR / f"mbb_calendar_{season}.csv", index=False)
            print(f"  saved calendar {season} | shape={cal_df.shape}")
        except Exception as e:
            print(f"  skipped calendar {season}: {e}")

# -----------------------------
# PLAY-BY-PLAY
# -----------------------------
if SAVE_PLAY_BY_PLAY:
    pbp_df = load_seasonal_data(mbb.load_mbb_pbp, "play_by_play", SEASONS)
    if not pbp_df.empty:
        pbp_df.to_csv(SAVE_DIR / f"mbb_pbp_{SEASON_TAG}.csv", index=False)
        print("Saved play-by-play CSV.")

print("Done.")