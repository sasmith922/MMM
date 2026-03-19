#!/usr/bin/env python3
"""
Single-file Bart Torvik + cbb26 merger for a 2026 reduced-feature model.

Usage:
    python3 build_torvik_2026_single.py --year 2026 --cbb26 cbb26.csv --outdir data/processed

Requires:
    pip install pandas requests
Optional:
    pip install cloudscraper
"""

from __future__ import annotations

import argparse
import io
import json
import re
import time
from difflib import get_close_matches
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    import requests
except Exception as e:
    raise SystemExit("This script needs the 'requests' package. Install it with: pip install requests") from e

try:
    import cloudscraper  # type: ignore
except Exception:
    cloudscraper = None


MANUAL_ALIASES = {
    "uconn": "connecticut",
    "unc wilmington": "uncw",
    "uncg": "north carolina greensboro",
    "ole miss": "mississippi",
    "miami fl": "miami",
    "saint marys": "saint mary's",
    "st marys": "saint mary's",
    "st johns": "st john's",
    "nc state": "north carolina state",
    "miss state": "mississippi state",
    "pitt": "pittsburgh",
    "vcu": "virginia commonwealth",
    "usc": "southern california",
    "lsu": "louisiana state",
    "smu": "southern methodist",
    "utah st": "utah state",
    "boise st": "boise state",
    "san diego st": "san diego state",
    "colorado st": "colorado state",
    "michigan st": "michigan state",
    "oklahoma st": "oklahoma state",
    "arizona st": "arizona state",
    "washington st": "washington state",
    "oregon st": "oregon state",
    "iowa st": "iowa state",
    "kansas st": "kansas state",
    "ball st": "ball state",
    "kent st": "kent state",
    "murray st": "murray state",
    "wichita st": "wichita state",
    "cleveland st": "cleveland state",
    "fresno st": "fresno state",
    "jackson st": "jackson state",
    "montana st": "montana state",
    "app state": "appalachian state",
    "ut martin": "tennessee martin",
    "uc san diego": "uc san diego",
    "uc irvine": "uc irvine",
    "uc davis": "uc davis",
    "ucsb": "uc santa barbara",
    "csu bakersfield": "cal state bakersfield",
    "csu northridge": "cal state northridge",
    "tamucc": "texas a&m corpus christi",
    "texas am corpus christi": "texas a&m corpus christi",
    "texas am": "texas a&m",
    "texas am commerce": "texas a&m commerce",
    "liu": "long island",
    "southern miss": "southern mississippi",
    "florida intl": "florida international",
    "fiu": "florida international",
    "byu": "brigham young",
}


def snake(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("%", "pct")
    s = s.replace("+/-", "plus_minus")
    s = s.replace("#", "num")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def normalize_team_name(name: str) -> str:
    s = str(name).strip().lower()
    s = s.replace("&", " and ")
    s = s.replace("saint", "st")
    s = s.replace("st.", "st")
    s = s.replace("state", "st")
    s = s.replace("university", "")
    s = s.replace("college", "")
    s = s.replace("the ", "")
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = MANUAL_ALIASES.get(s, s)
    return s


CBB26_RENAME = {
    "rk": "rank",
    "team": "team_name",
    "conf": "conference",
    "g": "games",
    "w": "wins",
    "adjoe": "pre_tourney_adjoe",
    "adjde": "pre_tourney_adjde",
    "barthag": "pre_tourney_barthag",
    "efg_o": "efgpct",
    "efg_d": "opp_efgpct",
    "tor": "topct",
    "tord": "opp_topct",
    "orb": "orpct",
    "drb": "opp_orpct",
    "ftr": "ftrate",
    "ftrd": "opp_ftrate",
    "2p_o": "fg2pct",
    "2p_d": "oppfg2pct",
    "3p_o": "fg3pct",
    "3p_d": "oppfg3pct",
    "adj_t": "pre_tourney_adjtempo",
    "wab": "pre_tourney_wab",
    "seed": "seed",
}


RAW_SOURCES = {
    "ratings": [
        "https://barttorvik.com/teamslicejson.php?year={year}&csv=1&type=R",
        "https://barttorvik.com/trank.php?year={year}&type=R&csv=1",
        "https://barttorvik.com/?year={year}&type=R&csv=1",
    ],
    "teamstats": [
        "https://barttorvik.com/teamstats.php?year={year}&csv=1",
        "https://barttorvik.com/teamstats.php?sort=2&year={year}&csv=1",
    ],
    "team_tables": [
        "https://barttorvik.com/team-tables_each.php?year={year}&csv=1",
        "https://barttorvik.com/team-tables_each.php?csv=1&year={year}",
    ],
    "teamsheets": [
        "https://barttorvik.com/teamsheets.php?year={year}&csv=1",
    ],
    "sos": [
        "https://barttorvik.com/sos.php?year={year}&csv=1",
    ],
}


def get_session():
    if cloudscraper is not None:
        session = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "darwin", "mobile": False}
        )
    else:
        session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept": "text/csv,application/json,text/plain,text/html,*/*",
            "Referer": "https://barttorvik.com/",
        }
    )
    return session


def looks_like_html(text: str) -> bool:
    lower = text.lower()
    return (
        "<html" in lower
        or "<!doctype" in lower
        or "verifying your browser" in lower
        or "just a moment" in lower
        or "cf-browser-verification" in lower
    )


def read_table_from_text(text: str) -> pd.DataFrame:
    text = text.strip("\ufeff\n\r \t")
    if not text:
        raise ValueError("empty response")
    if looks_like_html(text):
        raise ValueError("got HTML/browser-check instead of CSV")
    df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    if df.empty or df.shape[1] <= 1:
        raise ValueError(f"parsed unusable table shape={df.shape}")
    return df


def fetch_first_csv(session, url_templates: List[str], year: int) -> Tuple[pd.DataFrame, str]:
    errors = []
    for template in url_templates:
        url = template.format(year=year)
        try:
            resp = session.get(url, timeout=45)
            resp.raise_for_status()
            df = read_table_from_text(resp.text)
            return df, url
        except Exception as e:
            errors.append(f"{url} -> {type(e).__name__}: {e}")
            time.sleep(1)
    raise RuntimeError("All URL attempts failed:\n" + "\n".join(errors))


def find_team_col(columns: Iterable[str]) -> Optional[str]:
    cols = list(columns)
    preferred = ["team", "team_name", "school", "teamname", "tm", "team_full"]
    for p in preferred:
        if p in cols:
            return p
    for c in cols:
        if "team" in c and c != "opponent_team":
            return c
    return None


def clean_source_df(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [snake(c) for c in df.columns]

    team_col = find_team_col(df.columns)
    if team_col is None:
        raise ValueError(f"Could not identify team column for source '{source_name}'. Columns: {list(df.columns)}")

    df = df.rename(columns={team_col: "team_name_raw"})
    df["team_name_norm"] = df["team_name_raw"].map(normalize_team_name)

    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)

    protected = {"team_name_raw", "team_name_norm"}
    rename_map = {c: f"{source_name}_{c}" for c in df.columns if c not in protected}
    df = df.rename(columns=rename_map)

    if df["team_name_norm"].duplicated().any():
        completeness = df.notna().sum(axis=1)
        df = (
            df.assign(_complete=completeness)
              .sort_values(["team_name_norm", "_complete"], ascending=[True, False])
              .drop_duplicates(subset=["team_name_norm"], keep="first")
              .drop(columns=["_complete"])
        )

    return df


def fuzzy_fill(base_keys: List[str], source_keys: List[str], cutoff: float = 0.92) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    source_set = set(source_keys)
    for key in base_keys:
        if key in source_set:
            mapping[key] = key
            continue
        match = get_close_matches(key, source_keys, n=1, cutoff=cutoff)
        if match:
            mapping[key] = match[0]
    return mapping


def merge_source(base: pd.DataFrame, source: pd.DataFrame, source_name: str) -> pd.DataFrame:
    base = base.copy()
    source = source.copy()

    base_keys = base["team_name_norm"].astype(str).tolist()
    source_keys = source["team_name_norm"].astype(str).tolist()
    key_map = fuzzy_fill(base_keys, source_keys)
    base[f"_merge_key_{source_name}"] = base["team_name_norm"].map(key_map)

    merged = base.merge(
        source,
        left_on=f"_merge_key_{source_name}",
        right_on="team_name_norm",
        how="left",
        suffixes=("", f"_{source_name}_dup"),
    )

    if "team_name_norm_x" in merged.columns:
        merged = merged.rename(columns={"team_name_norm_x": "team_name_norm"})
    merged = merged.drop(
        columns=[c for c in [f"_merge_key_{source_name}", "team_name_norm_y"] if c in merged.columns],
        errors="ignore",
    )
    return merged


def add_cbb26_backbone(df: pd.DataFrame, year: int) -> pd.DataFrame:
    out = df.copy()
    out.columns = [snake(c) for c in out.columns]
    out = out.rename(columns={c: CBB26_RENAME.get(c, c) for c in out.columns})

    if "team_name" not in out.columns:
        raise ValueError("cbb26.csv must contain a TEAM column")

    out["season"] = year
    if "games" in out.columns and "wins" in out.columns:
        out["losses"] = out["games"] - out["wins"]
        out["win_pct"] = out["wins"] / out["games"]
    out["pre_tourney_adjem"] = out["pre_tourney_adjoe"] - out["pre_tourney_adjde"]
    out["team_name_norm"] = out["team_name"].map(normalize_team_name)
    return out


def choose_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred_front = [
        "season",
        "conference",
        "team_name",
        "team_name_norm",
        "seed",
        "games",
        "wins",
        "losses",
        "win_pct",
        "pre_tourney_adjtempo",
        "pre_tourney_adjoe",
        "pre_tourney_adjde",
        "pre_tourney_adjem",
        "pre_tourney_barthag",
        "pre_tourney_wab",
        "efgpct",
        "opp_efgpct",
        "topct",
        "opp_topct",
        "orpct",
        "opp_orpct",
        "ftrate",
        "opp_ftrate",
        "fg2pct",
        "oppfg2pct",
        "fg3pct",
        "oppfg3pct",
    ]
    existing_front = [c for c in preferred_front if c in df.columns]
    remaining = [c for c in df.columns if c not in existing_front]
    return df[existing_front + remaining]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--cbb26", type=str, default="cbb26.csv")
    parser.add_argument("--outdir", type=str, default="data/processed")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cbb_path = Path(args.cbb26)
    if not cbb_path.exists():
        raise SystemExit(f"Could not find cbb26 file: {cbb_path}")

    print(f"[1/5] Reading backbone file: {cbb_path}")
    cbb_raw = pd.read_csv(cbb_path)
    base = add_cbb26_backbone(cbb_raw, args.year)

    session = get_session()
    pulled: Dict[str, pd.DataFrame] = {}
    source_meta = {}

    print("[2/5] Pulling Bart Torvik CSVs...")
    for source_name, urls in RAW_SOURCES.items():
        try:
            raw_df, used_url = fetch_first_csv(session, urls, args.year)
            raw_path = outdir / f"torvik_{source_name}_{args.year}.csv"
            raw_df.to_csv(raw_path, index=False)
            cleaned = clean_source_df(raw_df, source_name)
            pulled[source_name] = cleaned
            source_meta[source_name] = {
                "used_url": used_url,
                "raw_path": str(raw_path),
                "rows": int(raw_df.shape[0]),
                "cols": int(raw_df.shape[1]),
                "raw_columns": list(raw_df.columns),
                "clean_columns": list(cleaned.columns),
            }
            print(f"  ✓ {source_name}: {used_url} -> {raw_df.shape}")
        except Exception as e:
            source_meta[source_name] = {"error": str(e)}
            print(f"  ! {source_name}: failed -> {e}")

    print("[3/5] Merging Torvik pulls into cbb26 backbone...")
    merged = base.copy()
    for source_name, df in pulled.items():
        merged = merge_source(merged, df, source_name)

    print("[4/5] Writing outputs...")
    merged = choose_columns(merged)
    final_path = outdir / f"team_features_{args.year}_reduced.csv"
    merged.to_csv(final_path, index=False)

    torvik_cols = merged.filter(regex=r"^(ratings_|teamstats_|team_tables_|teamsheets_|sos_)")
    if torvik_cols.shape[1] > 0:
        unmatched = merged[torvik_cols.isna().all(axis=1)][
            [c for c in ["team_name", "team_name_norm", "conference", "seed"] if c in merged.columns]
        ]
    else:
        unmatched = merged[[c for c in ["team_name", "team_name_norm", "conference", "seed"] if c in merged.columns]].copy()

    unmatched_path = outdir / f"team_features_{args.year}_unmatched_teams.csv"
    unmatched.to_csv(unmatched_path, index=False)

    report = {
        "year": args.year,
        "input_cbb26": str(cbb_path),
        "output_final_csv": str(final_path),
        "output_unmatched_csv": str(unmatched_path),
        "rows": int(merged.shape[0]),
        "cols": int(merged.shape[1]),
        "sources": source_meta,
        "final_columns": list(merged.columns),
    }
    report_path = outdir / f"team_features_{args.year}_build_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(f"  ✓ final merged file: {final_path}")
    print(f"  ✓ unmatched teams file: {unmatched_path}")
    print(f"  ✓ build report: {report_path}")
    print("[5/5] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
