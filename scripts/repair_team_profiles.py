from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import re

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RAW_DIR = ROOT_DIR / "data" / "raw"

TEAM_PROFILES_PATH = PROCESSED_DIR / "team_profiles.csv"
TEAM_PROFILES_BACKUP_PATH = PROCESSED_DIR / "team_profiles.pre_repair_backup.csv"
TEAM_PROFILES_REPAIRED_PATH = PROCESSED_DIR / "team_profiles_repaired.csv"
TEAM_MAPPING_PATH = PROCESSED_DIR / "team_mapping.csv"
TOURNEY_MATCHUPS_PATH = PROCESSED_DIR / "tourney_matchups.csv"
MTEAMS_PATH = RAW_DIR / "march-machine-learning-mania-2026" / "MTeams.csv"

ABBREV_ALIAS_MAP: dict[str, set[str]] = {
    "c michigan": {"central michigan"},
    "e washington": {"eastern washington"},
    "s illinois": {"southern illinois"},
    "s carolina st": {"south carolina state"},
    "sam houston st": {"sam houston state", "sam houston"},
    "tx southern": {"texas southern"},
    "ut san antonio": {"utsa", "texas san antonio"},
    "wku": {"western kentucky"},
    "wi milwaukee": {"wisconsin milwaukee", "green bay", "milwaukee"},
    "il chicago": {"illinois chicago", "uic"},
    "st joseph s pa": {"saint joseph s", "st joseph s", "saint josephs", "st josephs"},
    "penn": {"pennsylvania"},
    "etsu": {"east tennessee state", "east tennessee st"},
    "connecticut": {"uconn"},
    "iupui": {
        "iupui",
        "indiana purdue indianapolis",
        "indiana purdue university indianapolis",
        "indiana purdue indianapolis jaguars",
    },
    "monmouth nj": {"monmouth"},
    "f dickinson": {"fairleigh dickinson"},
    "g washington": {"george washington", "gw"},
    "st mary s ca": {"saint mary s", "saint marys", "st marys", "saint mary s california"},
    "suny albany": {"albany", "albany ny", "albany new york", "ualbany"},
    "kent": {"kent state"},
    "northwestern la": {"northwestern state", "northwestern state la"},
    "southern univ": {"southern", "southern university"},
    "central conn": {"central connecticut", "central connecticut state"},
    "tam c christi": {"texas a and m corpus christi", "texas a&m corpus christi", "texas am corpus christi"},
    "american univ": {"american", "american university"},
    "cs fullerton": {"cal state fullerton", "california state fullerton"},
    "cs northridge": {"cal state northridge", "california state northridge"},
    "miami fl": {"miami", "miami florida"},
    "ms valley st": {"mississippi valley state"},
    "mt st mary s": {"mount st mary s", "mount saint mary s"},
    "n dakota st": {"north dakota state"},
    "s dakota st": {"south dakota state"},
    "sf austin": {"stephen f austin", "stephen f austin state"},
    "ark pine bluff": {"arkansas pine bluff"},
    "ark little rock": {"arkansas little rock", "little rock", "ualr"},
    "liu brooklyn": {"long island brooklyn", "long island university brooklyn", "long island university", "liu"},
    "loyola md": {"loyola maryland"},
    "fgcu": {"florida gulf coast"},
    "mississippi": {"ole miss"},
    "mtsu": {"middle tennessee", "middle tennessee state"},
    "prairie view": {"prairie view a and m", "prairie view a&m", "prairie view am"},
    "appalachian st": {"appalachian state", "app state"},
    "fl atlantic": {"florida atlantic", "fau"},
    "kennesaw": {"kennesaw state"},
    "se missouri st": {"southeast missouri state", "se missouri state"},
    "mcneese st": {"mcneese state", "mcneese"},
    "siue": {"southern illinois edwardsville", "siu edwardsville"},
    "ne omaha": {"omaha", "nebraska omaha"},
    "coastal car": {"coastal carolina"},
    "nc central": {"north carolina central"},
    "col charleston": {"college of charleston", "charleston"},
    "abilene chr": {"abilene christian"},
    "detroit": {"detroit mercy"},
    "nc a t": {"north carolina a and t", "north carolina a t"},
}

FINAL_SEASON_TEAM_FIXES = [
    {"season": 2003, "team_id": 1237, "aliases": {"iupui", "indiana purdue indianapolis", "indiana purdue university indianapolis"}},
    {"season": 2021, "team_id": 1111, "aliases": {"appalachian st", "appalachian state", "app state"}},
    {"season": 2024, "team_id": 1270, "aliases": {"mcneese st", "mcneese state", "mcneese"}},
    {"season": 2025, "team_id": 1270, "aliases": {"mcneese st", "mcneese state", "mcneese"}},
    {"season": 2025, "team_id": 1188, "aliases": {"siue", "southern illinois edwardsville", "siu edwardsville"}},
    {"season": 2003, "team_id": 1237, "aliases": {"iupui", "iu indianapolis", "iu indy", "iu indy jaguars"}},
    {"season": 2016, "team_id": 1453, "aliases": {"wi green bay", "green bay", "wisconsin green bay", "green bay phoenix"}},
]

NAME_COLS = [
    "canonical_team_name",
    "kaggle_team_name",
    "kaggle_team_name_resume",
    "team_name",
    "full_team_name",
    "team_name_norm",
]


def normalize_name(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip().lower()
    s = s.replace("&", " and ")
    s = s.replace("/", " ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def make_name_keys(value) -> set[str]:
    base = normalize_name(value)
    if not base:
        return set()

    keys = {base}
    keys.add(re.sub(r"\buniversity\b", "", base).strip())
    keys.add(re.sub(r"\bcollege\b", "", base).strip())
    keys.add(re.sub(r"\bsaint\b", "st", base).strip())
    keys.add(re.sub(r"\bst\b", "saint", base).strip())
    keys.add(re.sub(r"\bstate\b", "st", base).strip())
    keys.add(re.sub(r"\bst\b", "state", base).strip())

    expanded = set()
    for k in list(keys):
        expanded.add(re.sub(r"^c\s+", "central ", k).strip())
        expanded.add(re.sub(r"^e\s+", "eastern ", k).strip())
        expanded.add(re.sub(r"^w\s+", "western ", k).strip())
        expanded.add(re.sub(r"^s\s+", "southern ", k).strip())
        expanded.add(re.sub(r"^n\s+", "northern ", k).strip())
        expanded.add(re.sub(r"^tx\s+", "texas ", k).strip())
        expanded.add(re.sub(r"^ut\s+", "texas ", k).strip())
        expanded.add(re.sub(r"^wi\s+", "wisconsin ", k).strip())
        expanded.add(re.sub(r"^il\s+", "illinois ", k).strip())
        expanded.add(re.sub(r"\buniv\b", "university", k).strip())
        expanded.add(re.sub(r"\bmt\b", "mount", k).strip())
        expanded.add(re.sub(r"\bconn\b", "connecticut", k).strip())
        expanded.add(re.sub(r"\bfullerton\b", "state fullerton", k).strip())
        expanded.add(re.sub(r"\bchristi\b", "corpus christi", k).strip())
        expanded.add(re.sub(r"\bark\b", "arkansas", k).strip())
        expanded.add(re.sub(r"\bcs\b", "cal state", k).strip())
        expanded.add(re.sub(r"\bmd\b", "maryland", k).strip())
        expanded.add(re.sub(r"\bmtsu\b", "middle tennessee state", k).strip())
        expanded.add(re.sub(r"\bfgcu\b", "florida gulf coast", k).strip())
        expanded.add(re.sub(r"\bsf\b", "stephen f", k).strip())
        expanded.add(re.sub(r"\bndakota\b", "north dakota", k).strip())
        expanded.add(re.sub(r"\bsdakota\b", "south dakota", k).strip())
        expanded.add(re.sub(r"\bchr\b", "christian", k).strip())
        expanded.add(re.sub(r"\bcar\b", "carolina", k).strip())
        expanded.add(re.sub(r"\bcol\b", "college", k).strip())
        expanded.add(re.sub(r"\bnc\b", "north carolina", k).strip())
        expanded.add(re.sub(r"\bliu\b", "long island university", k).strip())
        expanded.add(re.sub(r"\ba t\b", "a and t", k).strip())
        expanded.add(re.sub(r"\bfl\b", "florida", k).strip())
        expanded.add(re.sub(r"\bse\b", "southeast", k).strip())
        expanded.add(re.sub(r"\bsiue\b", "southern illinois edwardsville", k).strip())
        expanded.add(re.sub(r"\bne\b", "nebraska", k).strip())
        expanded.add(re.sub(r"\bst\b", "state", k).strip())

        if k in ABBREV_ALIAS_MAP:
            expanded.update(ABBREV_ALIAS_MAP[k])

    keys.update(expanded)
    keys = {re.sub(r"\s+", " ", k).strip() for k in keys if k}
    return {k for k in keys if k}


def row_name_keys(row: pd.Series, name_cols: list[str]) -> set[str]:
    keys = set()
    for col in name_cols:
        if col in row.index:
            keys.update(make_name_keys(row.get(col)))
    return keys


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def load_tourney_matchups() -> pd.DataFrame:
    df = pd.read_csv(TOURNEY_MATCHUPS_PATH)
    rename_map = {
        "team_a_id": "teamA_id",
        "team_b_id": "teamB_id",
        "team_a_name": "teamA_name",
        "team_b_name": "teamB_name",
    }
    existing = {old: new for old, new in rename_map.items() if old in df.columns}
    if existing:
        df = df.rename(columns=existing)
    return df


def build_season_name_id_lookup_from_matchups(df: pd.DataFrame) -> dict[tuple[int, str], int]:
    mapping: dict[tuple[int, str], set[int]] = defaultdict(set)

    for side in ["A", "B"]:
        id_col = f"team{side}_id"
        name_col = f"team{side}_name"
        if id_col not in df.columns or name_col not in df.columns:
            continue

        temp = df[["season", id_col, name_col]].copy()
        temp["season"] = pd.to_numeric(temp["season"], errors="coerce")
        temp[id_col] = pd.to_numeric(temp[id_col], errors="coerce")
        temp = temp.dropna(subset=["season", id_col, name_col]).copy()
        temp["season"] = temp["season"].astype(int)
        temp[id_col] = temp[id_col].astype(int)

        for _, row in temp.iterrows():
            season = int(row["season"])
            team_id = int(row[id_col])
            for key in make_name_keys(row[name_col]):
                mapping[(season, key)].add(team_id)

    return {k: next(iter(v)) for k, v in mapping.items() if len(v) == 1}


def build_global_lookup_from_df(
    df: pd.DataFrame,
    id_col_candidates: list[str],
    name_col_candidates: list[str],
) -> dict[str, int]:
    id_col = next((c for c in id_col_candidates if c in df.columns), None)
    if id_col is None:
        return {}

    temp = df.copy()
    temp[id_col] = pd.to_numeric(temp[id_col], errors="coerce")
    temp = temp[temp[id_col].notna()].copy()
    temp[id_col] = temp[id_col].astype(int)

    mapping: dict[str, set[int]] = defaultdict(set)
    for name_col in name_col_candidates:
        if name_col not in temp.columns:
            continue
        for _, row in temp[[id_col, name_col]].dropna().iterrows():
            team_id = int(row[id_col])
            for key in make_name_keys(row[name_col]):
                mapping[key].add(team_id)

    return {k: next(iter(v)) for k, v in mapping.items() if len(v) == 1}


def find_best_same_season_fuzzy_match(
    team_profiles: pd.DataFrame,
    *,
    season: int,
    alias_keys: set[str],
    name_cols: list[str],
    min_score: float = 0.82,
    min_gap: float = 0.05,
) -> int | None:
    candidates = team_profiles[
        (pd.to_numeric(team_profiles["season"], errors="coerce") == season)
        & (pd.to_numeric(team_profiles["team_id"], errors="coerce").isna())
    ].copy()

    if candidates.empty:
        return None

    scored: list[tuple[float, int]] = []
    for idx, row in candidates.iterrows():
        cand_keys = row_name_keys(row, name_cols)
        if not cand_keys:
            continue
        best = 0.0
        for a in alias_keys:
            for b in cand_keys:
                best = max(best, similarity(a, b))
        if best > 0:
            scored.append((best, idx))

    if not scored:
        return None

    scored.sort(reverse=True)
    best_score, best_idx = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    if best_score >= min_score and (best_score - second_score) >= min_gap:
        return best_idx
    return None


def fill_from_remaining_tourney_rows(
    team_profiles: pd.DataFrame,
    tourney_matchups: pd.DataFrame,
    name_cols: list[str],
) -> tuple[pd.DataFrame, int]:
    tp = team_profiles.copy()

    covered = tp.dropna(subset=["season", "team_id"]).copy()
    covered["season"] = pd.to_numeric(covered["season"], errors="coerce").astype(int)
    covered["team_id"] = pd.to_numeric(covered["team_id"], errors="coerce").astype(int)
    covered_keys = set(zip(covered["season"], covered["team_id"]))

    left = tourney_matchups[["season", "teamA_id", "teamA_name"]].rename(
        columns={"teamA_id": "team_id", "teamA_name": "team_name"}
    )
    right = tourney_matchups[["season", "teamB_id", "teamB_name"]].rename(
        columns={"teamB_id": "team_id", "teamB_name": "team_name"}
    )
    missing_rows = pd.concat([left, right], ignore_index=True).drop_duplicates().copy()
    missing_rows["season"] = pd.to_numeric(missing_rows["season"], errors="coerce")
    missing_rows["team_id"] = pd.to_numeric(missing_rows["team_id"], errors="coerce")
    missing_rows = missing_rows.dropna(subset=["season", "team_id", "team_name"]).copy()
    missing_rows["season"] = missing_rows["season"].astype(int)
    missing_rows["team_id"] = missing_rows["team_id"].astype(int)
    missing_rows = missing_rows[
        ~missing_rows.apply(lambda r: (r["season"], r["team_id"]) in covered_keys, axis=1)
    ].copy()

    filled = 0
    for _, miss in missing_rows.iterrows():
        season = int(miss["season"])
        team_id = int(miss["team_id"])
        tourney_keys = make_name_keys(miss["team_name"])

        candidates = tp[
            (pd.to_numeric(tp["season"], errors="coerce") == season)
            & (pd.to_numeric(tp["team_id"], errors="coerce").isna())
        ].copy()

        matched_indices = []
        for idx, cand in candidates.iterrows():
            if row_name_keys(cand, name_cols) & tourney_keys:
                matched_indices.append(idx)

        if len(matched_indices) == 1:
            idx = matched_indices[0]
            tp.at[idx, "team_id"] = team_id
            if "kaggle_team_id" in tp.columns:
                tp.at[idx, "kaggle_team_id"] = team_id
            filled += 1

    return tp, filled


def apply_final_manual_team_fixes(
    team_profiles: pd.DataFrame,
    name_cols: list[str],
) -> tuple[pd.DataFrame, int]:
    tp = team_profiles.copy()
    filled = 0

    for fix in FINAL_SEASON_TEAM_FIXES:
        season = int(fix["season"])
        team_id = int(fix["team_id"])
        alias_keys = set()
        for alias in fix["aliases"]:
            alias_keys.update(make_name_keys(alias))

        candidates = tp[
            (pd.to_numeric(tp["season"], errors="coerce") == season)
            & (pd.to_numeric(tp["team_id"], errors="coerce").isna())
        ].copy()

        matched_indices = []
        for idx, row in candidates.iterrows():
            if row_name_keys(row, name_cols) & alias_keys:
                matched_indices.append(idx)

        if len(matched_indices) == 1:
            idx = matched_indices[0]
            tp.at[idx, "team_id"] = team_id
            if "kaggle_team_id" in tp.columns:
                tp.at[idx, "kaggle_team_id"] = team_id
            filled += 1

    return tp, filled


def apply_final_fuzzy_fixes(
    team_profiles: pd.DataFrame,
    tourney_matchups: pd.DataFrame,
    name_cols: list[str],
) -> tuple[pd.DataFrame, int]:
    tp = team_profiles.copy()

    covered = tp.dropna(subset=["season", "team_id"]).copy()
    covered["season"] = pd.to_numeric(covered["season"], errors="coerce").astype(int)
    covered["team_id"] = pd.to_numeric(covered["team_id"], errors="coerce").astype(int)
    covered_keys = set(zip(covered["season"], covered["team_id"]))

    left = tourney_matchups[["season", "teamA_id", "teamA_name"]].rename(
        columns={"teamA_id": "team_id", "teamA_name": "team_name"}
    )
    right = tourney_matchups[["season", "teamB_id", "teamB_name"]].rename(
        columns={"teamB_id": "team_id", "teamB_name": "team_name"}
    )
    missing_rows = pd.concat([left, right], ignore_index=True).drop_duplicates().copy()
    missing_rows["season"] = pd.to_numeric(missing_rows["season"], errors="coerce")
    missing_rows["team_id"] = pd.to_numeric(missing_rows["team_id"], errors="coerce")
    missing_rows = missing_rows.dropna(subset=["season", "team_id", "team_name"]).copy()
    missing_rows["season"] = missing_rows["season"].astype(int)
    missing_rows["team_id"] = missing_rows["team_id"].astype(int)
    missing_rows = missing_rows[
        ~missing_rows.apply(lambda r: (r["season"], r["team_id"]) in covered_keys, axis=1)
    ].copy()

    filled = 0
    for _, miss in missing_rows.iterrows():
        season = int(miss["season"])
        team_id = int(miss["team_id"])
        alias_keys = make_name_keys(miss["team_name"])

        idx = find_best_same_season_fuzzy_match(
            tp,
            season=season,
            alias_keys=alias_keys,
            name_cols=name_cols,
            min_score=0.82,
            min_gap=0.05,
        )

        if idx is not None:
            tp.at[idx, "team_id"] = team_id
            if "kaggle_team_id" in tp.columns:
                tp.at[idx, "kaggle_team_id"] = team_id
            filled += 1

    return tp, filled


def validate_tourney_join_coverage(team_profiles_df: pd.DataFrame, tourney_matchups_df: pd.DataFrame) -> None:
    profile_keys = set(
        zip(
            pd.to_numeric(team_profiles_df["season"], errors="coerce").astype(int),
            pd.to_numeric(team_profiles_df["team_id"], errors="coerce").astype(int),
        )
    )

    matchup_ids = pd.concat(
        [
            tourney_matchups_df[["season", "teamA_id"]].rename(columns={"teamA_id": "team_id"}),
            tourney_matchups_df[["season", "teamB_id"]].rename(columns={"teamB_id": "team_id"}),
        ],
        ignore_index=True,
    )
    matchup_ids["season"] = pd.to_numeric(matchup_ids["season"], errors="coerce")
    matchup_ids["team_id"] = pd.to_numeric(matchup_ids["team_id"], errors="coerce")
    matchup_ids = matchup_ids.dropna(subset=["season", "team_id"]).copy()
    matchup_ids["season"] = matchup_ids["season"].astype(int)
    matchup_ids["team_id"] = matchup_ids["team_id"].astype(int)

    required_keys = set(zip(matchup_ids["season"], matchup_ids["team_id"]))
    missing = sorted(k for k in required_keys if k not in profile_keys)

    if missing:
        missing_df = pd.DataFrame(missing, columns=["season", "team_id"])
        left = tourney_matchups_df[["season", "teamA_id", "teamA_name"]].rename(
            columns={"teamA_id": "team_id", "teamA_name": "team_name"}
        )
        right = tourney_matchups_df[["season", "teamB_id", "teamB_name"]].rename(
            columns={"teamB_id": "team_id", "teamB_name": "team_name"}
        )
        name_lookup = pd.concat([left, right], ignore_index=True).drop_duplicates()
        sample = (
            missing_df.merge(name_lookup, on=["season", "team_id"], how="left")
            .drop_duplicates()
            .head(20)
        )
        raise ValueError(
            "After repair, tournament teams are still missing from team_profiles.\n"
            f"Sample missing rows:\n{sample.to_string(index=False)}"
        )


def main() -> None:
    team_profiles = pd.read_csv(TEAM_PROFILES_PATH)
    original = team_profiles.copy()

    # Standardize primary ID column name
    if "team_id" not in team_profiles.columns and "kaggle_team_id" in team_profiles.columns:
        team_profiles = team_profiles.rename(columns={"kaggle_team_id": "team_id"})

    team_profiles["team_id"] = pd.to_numeric(team_profiles["team_id"], errors="coerce")
    team_profiles["season"] = pd.to_numeric(team_profiles["season"], errors="coerce")

    before_missing = int(team_profiles["team_id"].isna().sum())
    print(f"Missing team_id before repair: {before_missing}")

    tourney_matchups = load_tourney_matchups()
    season_lookup = build_season_name_id_lookup_from_matchups(tourney_matchups)

    global_lookups: list[dict[str, int]] = []

    # Existing non-missing IDs already in team_profiles
    global_lookups.append(
        build_global_lookup_from_df(
            team_profiles,
            id_col_candidates=["team_id"],
            name_col_candidates=NAME_COLS,
        )
    )

    # team_mapping.csv if available
    if TEAM_MAPPING_PATH.exists():
        team_mapping = pd.read_csv(TEAM_MAPPING_PATH)
        global_lookups.append(
            build_global_lookup_from_df(
                team_mapping,
                id_col_candidates=["team_id", "kaggle_team_id", "TeamID"],
                name_col_candidates=[
                    "canonical_team_name",
                    "kaggle_team_name",
                    "team_name",
                    "team_name_norm",
                    "TeamName",
                ],
            )
        )

    # Kaggle MTeams.csv if available
    if MTEAMS_PATH.exists():
        mteams = pd.read_csv(MTEAMS_PATH)
        global_lookups.append(
            build_global_lookup_from_df(
                mteams,
                id_col_candidates=["TeamID"],
                name_col_candidates=["TeamName"],
            )
        )

    # First pass: season-specific tournament name lookup, then global lookups
    filled = 0
    missing_idx = team_profiles.index[team_profiles["team_id"].isna()].tolist()

    for idx in missing_idx:
        row = team_profiles.loc[idx]
        resolved_id = None

        season = row.get("season")
        if pd.notna(season):
            season = int(season)
            for key in row_name_keys(row, NAME_COLS):
                if (season, key) in season_lookup:
                    resolved_id = season_lookup[(season, key)]
                    break

        if resolved_id is None:
            for key in row_name_keys(row, NAME_COLS):
                for lookup in global_lookups:
                    if key in lookup:
                        resolved_id = lookup[key]
                        break
                if resolved_id is not None:
                    break

        if resolved_id is not None:
            team_profiles.at[idx, "team_id"] = int(resolved_id)
            if "kaggle_team_id" in team_profiles.columns:
                team_profiles.at[idx, "kaggle_team_id"] = int(resolved_id)
            filled += 1

    # Second pass: direct same-season tournament row matching
    team_profiles, second_pass_filled = fill_from_remaining_tourney_rows(
        team_profiles=team_profiles,
        tourney_matchups=tourney_matchups,
        name_cols=NAME_COLS,
    )
    filled += second_pass_filled

    # Manual season/team fixes
    team_profiles, final_manual_filled = apply_final_manual_team_fixes(
        team_profiles=team_profiles,
        name_cols=NAME_COLS,
    )
    filled += final_manual_filled

    # Fuzzy fallback
    team_profiles, final_fuzzy_filled = apply_final_fuzzy_fixes(
        team_profiles=team_profiles,
        tourney_matchups=tourney_matchups,
        name_cols=NAME_COLS,
    )
    filled += final_fuzzy_filled

    # Final exact patch for the known 2016 Green Bay row
    final_exact_filled = 0
    season_num = pd.to_numeric(team_profiles["season"], errors="coerce")
    team_id_num = pd.to_numeric(team_profiles["team_id"], errors="coerce")
    team_name_norm = team_profiles["team_name_norm"].fillna("").astype(str).str.strip().str.lower()

    green_bay_mask = (
        (season_num == 2016)
        & (team_id_num.isna())
        & (team_name_norm == "green bay")
    )

    if int(green_bay_mask.sum()) == 1:
        team_profiles.loc[green_bay_mask, "team_id"] = 1453
        if "kaggle_team_id" in team_profiles.columns:
            team_profiles.loc[green_bay_mask, "kaggle_team_id"] = 1453
        filled += 1
        final_exact_filled = 1

    after_missing = int(pd.to_numeric(team_profiles["team_id"], errors="coerce").isna().sum())
    print(f"Filled missing team_id rows: {filled}")
    print(f"Second-pass tournament-specific fills: {second_pass_filled}")
    print(f"Final manual fills: {final_manual_filled}")
    print(f"Final fuzzy fills: {final_fuzzy_filled}")
    print(f"Final exact fills: {final_exact_filled}")
    print(f"Missing team_id after repair: {after_missing}")

    # Save repaired preview
    team_profiles.to_csv(TEAM_PROFILES_REPAIRED_PATH, index=False)
    print(f"Saved repaired preview to: {TEAM_PROFILES_REPAIRED_PATH}")

    # Validate against tournament teams
    validated_profiles = team_profiles.dropna(subset=["team_id", "season"]).copy()
    validated_profiles["season"] = pd.to_numeric(validated_profiles["season"], errors="coerce").astype(int)
    validated_profiles["team_id"] = pd.to_numeric(validated_profiles["team_id"], errors="coerce").astype(int)

    validate_tourney_join_coverage(validated_profiles, tourney_matchups)

    # Backup original once
    if not TEAM_PROFILES_BACKUP_PATH.exists():
        original.to_csv(TEAM_PROFILES_BACKUP_PATH, index=False)
        print(f"Backed up original team_profiles to: {TEAM_PROFILES_BACKUP_PATH}")

    # Promote repaired file
    team_profiles.to_csv(TEAM_PROFILES_PATH, index=False)
    print(f"Overwrote main file with repaired version: {TEAM_PROFILES_PATH}")


if __name__ == "__main__":
    main()