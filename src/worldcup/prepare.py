#!/usr/bin/env python3
"""Prepare international football match data for ML training.

The script cleans the Kaggle-style international football CSVs, standardizes
former country names, adds chronological form/head-to-head/Elo features, and
writes a model-ready table for tree models such as XGBoost or Random Forests.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Iterable, Tuple

import numpy as np
import pandas as pd


MIN_MATCH_DATE = pd.Timestamp("1990-01-01")
MATCH_KEYS = ["date", "home_team", "away_team"]
TEAM_FEATURE_WINDOWS = (5, 10)


@dataclass
class TeamHistory:
    """Chronological team state used to build pre-match features."""

    matches: Deque[Tuple[int, int, int, int, int]] = field(default_factory=deque)
    elo: float = 1500.0


@dataclass
class HeadToHeadHistory:
    """Chronological pairwise state for two teams."""

    home_wins: int = 0
    away_wins: int = 0
    draws: int = 0

    @property
    def total(self) -> int:
        return self.home_wins + self.away_wins + self.draws


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and feature-engineer international football match data."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing results.csv, shootouts.csv, goalscorers.csv, former_names.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/prepared_world_cup_training_data.csv"),
        help="CSV path for the model-ready dataframe.",
    )
    parser.add_argument(
        "--encoding-map-output",
        type=Path,
        default=Path("data/category_encoding_maps.json"),
        help="JSON path for categorical encoding maps.",
    )
    return parser.parse_args()


def read_csv(data_dir: Path, filename: str, usecols: Iterable[str], **kwargs) -> pd.DataFrame:
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path, usecols=list(usecols), low_memory=False, **kwargs)


def build_team_name_map(former_names: pd.DataFrame) -> Dict[str, str]:
    former_names = former_names.dropna(subset=["current", "former"]).copy()
    former_names["current"] = former_names["current"].astype("string").str.strip()
    former_names["former"] = former_names["former"].astype("string").str.strip()
    return dict(zip(former_names["former"], former_names["current"]))


def standardize_team_names(df: pd.DataFrame, columns: Iterable[str], name_map: Dict[str, str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            clean = df[column].astype("string").str.strip()
            df[column] = clean.replace(name_map)
    return df


def load_and_clean(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    former_names = read_csv(
        data_dir,
        "former_names.csv",
        usecols=["current", "former", "start_date", "end_date"],
        dtype="string",
    )
    name_map = build_team_name_map(former_names)

    results = read_csv(
        data_dir,
        "results.csv",
        usecols=[
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "tournament",
            "city",
            "country",
            "neutral",
        ],
        dtype={
            "home_team": "string",
            "away_team": "string",
            "tournament": "string",
            "city": "string",
            "country": "string",
            "neutral": "boolean",
        },
    )
    shootouts = read_csv(
        data_dir,
        "shootouts.csv",
        usecols=["date", "home_team", "away_team", "winner"],
        dtype={
            "home_team": "string",
            "away_team": "string",
            "winner": "string",
        },
    )
    goalscorers = read_csv(
        data_dir,
        "goalscorers.csv",
        usecols=[
            "date",
            "home_team",
            "away_team",
            "team",
            "minute",
            "own_goal",
            "penalty",
        ],
        dtype={
            "home_team": "string",
            "away_team": "string",
            "team": "string",
            "own_goal": "boolean",
            "penalty": "boolean",
        },
    )

    for df in (results, shootouts, goalscorers):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df.dropna(subset=["date", "home_team", "away_team"], inplace=True)
        df.query("date >= @MIN_MATCH_DATE", inplace=True)

    standardize_team_names(results, ["home_team", "away_team", "country"], name_map)
    standardize_team_names(shootouts, ["home_team", "away_team", "winner"], name_map)
    standardize_team_names(goalscorers, ["home_team", "away_team", "team"], name_map)

    results["tournament"] = results["tournament"].fillna("Unknown").astype("string").str.strip()
    results["city"] = results["city"].fillna("Unknown").astype("string").str.strip()
    results["country"] = results["country"].fillna("Unknown").astype("string").str.strip()
    results["neutral"] = results["neutral"].fillna(False).astype(bool)
    results["home_score"] = pd.to_numeric(results["home_score"], errors="coerce").astype("Int16")
    results["away_score"] = pd.to_numeric(results["away_score"], errors="coerce").astype("Int16")
    results.dropna(subset=["home_score", "away_score"], inplace=True)

    goalscorers["minute"] = pd.to_numeric(goalscorers["minute"], errors="coerce")
    goalscorers["own_goal"] = goalscorers["own_goal"].fillna(False).astype(bool)
    goalscorers["penalty"] = goalscorers["penalty"].fillna(False).astype(bool)

    return results, shootouts, goalscorers


def aggregate_goalscorers(goalscorers: pd.DataFrame) -> pd.DataFrame:
    aggregate_columns = [
        "scorer_events",
        "penalty_goals",
        "own_goals",
        "late_goals",
        "stoppage_goals",
        "home_goal_events",
        "away_goal_events",
        "home_penalty_goals",
        "away_penalty_goals",
        "home_late_goals",
        "away_late_goals",
    ]
    if goalscorers.empty:
        return pd.DataFrame(columns=MATCH_KEYS + aggregate_columns)

    goals = goalscorers.copy()
    goals["is_late_goal"] = goals["minute"].ge(75).fillna(False)
    goals["is_stoppage_goal"] = goals["minute"].gt(90).fillna(False)
    goals["is_home_goal_event"] = goals["team"].eq(goals["home_team"])
    goals["is_away_goal_event"] = goals["team"].eq(goals["away_team"])
    goals["is_home_penalty"] = goals["penalty"] & goals["is_home_goal_event"]
    goals["is_away_penalty"] = goals["penalty"] & goals["is_away_goal_event"]
    goals["is_home_late_goal"] = goals["is_late_goal"] & goals["is_home_goal_event"]
    goals["is_away_late_goal"] = goals["is_late_goal"] & goals["is_away_goal_event"]

    aggregations = {
        "scorer_events": ("team", "size"),
        "penalty_goals": ("penalty", "sum"),
        "own_goals": ("own_goal", "sum"),
        "late_goals": ("is_late_goal", "sum"),
        "stoppage_goals": ("is_stoppage_goal", "sum"),
        "home_goal_events": ("is_home_goal_event", "sum"),
        "away_goal_events": ("is_away_goal_event", "sum"),
        "home_penalty_goals": ("is_home_penalty", "sum"),
        "away_penalty_goals": ("is_away_penalty", "sum"),
        "home_late_goals": ("is_home_late_goal", "sum"),
        "away_late_goals": ("is_away_late_goal", "sum"),
    }
    aggregated = goals.groupby(MATCH_KEYS, sort=False, observed=True).agg(**aggregations).reset_index()
    aggregated[aggregate_columns] = aggregated[aggregate_columns].astype("Int16")
    return aggregated


def merge_match_context(
    results: pd.DataFrame, shootouts: pd.DataFrame, goal_features: pd.DataFrame
) -> pd.DataFrame:
    shootout_winners = shootouts.dropna(subset=["winner"]).drop_duplicates(MATCH_KEYS, keep="last")
    matches = results.merge(
        shootout_winners[MATCH_KEYS + ["winner"]],
        on=MATCH_KEYS,
        how="left",
        validate="m:1",
    )
    matches.rename(columns={"winner": "shootout_winner"}, inplace=True)
    matches = matches.merge(goal_features, on=MATCH_KEYS, how="left", validate="m:1")

    goal_columns = [
        "scorer_events",
        "penalty_goals",
        "own_goals",
        "late_goals",
        "stoppage_goals",
        "home_goal_events",
        "away_goal_events",
        "home_penalty_goals",
        "away_penalty_goals",
        "home_late_goals",
        "away_late_goals",
    ]
    for column in goal_columns:
        if column in matches.columns:
            matches[column] = matches[column].fillna(0).astype("Int16")

    matches["goal_difference"] = matches["home_score"].astype(int) - matches["away_score"].astype(int)
    matches["regular_time_target"] = np.sign(matches["goal_difference"]).astype("int8")
    matches["regular_time_draw"] = matches["regular_time_target"].eq(0)
    matches["home_shootout_win"] = matches["shootout_winner"].eq(matches["home_team"]).fillna(False)
    matches["away_shootout_win"] = matches["shootout_winner"].eq(matches["away_team"]).fillna(False)
    matches["target"] = matches["regular_time_target"].copy()
    matches.loc[matches["regular_time_draw"] & matches["home_shootout_win"], "target"] = 1
    matches.loc[matches["regular_time_draw"] & matches["away_shootout_win"], "target"] = -1
    matches["target"] = matches["target"].astype("int8")

    matches["is_world_cup"] = matches["tournament"].eq("FIFA World Cup")
    matches["is_friendly"] = matches["tournament"].eq("Friendly")
    matches["is_competitive"] = ~matches["is_friendly"]
    matches["sample_weight"] = np.select(
        [matches["is_world_cup"], matches["is_competitive"]],
        [3.0, 1.5],
        default=0.75,
    ).astype("float32")

    matches.sort_values(["date", "home_team", "away_team"], inplace=True, kind="mergesort")
    matches.reset_index(drop=True, inplace=True)
    return matches


def summarize_recent_form(
    history: Deque[Tuple[int, int, int, int, int]], window: int, prefix: str
) -> Dict[str, float]:
    recent = list(history)[-window:]
    if not recent:
        return {
            f"{prefix}_matches_last_{window}": 0,
            f"{prefix}_win_rate_last_{window}": 0.0,
            f"{prefix}_avg_goals_for_last_{window}": 0.0,
            f"{prefix}_avg_goals_against_last_{window}": 0.0,
            f"{prefix}_avg_points_last_{window}": 0.0,
            f"{prefix}_avg_penalties_for_last_{window}": 0.0,
            f"{prefix}_avg_late_goals_for_last_{window}": 0.0,
        }

    arr = np.asarray(recent, dtype=np.float32)
    outcomes = arr[:, 0]
    goals_for = arr[:, 1]
    goals_against = arr[:, 2]
    penalties_for = arr[:, 3]
    late_goals_for = arr[:, 4]
    points = np.where(outcomes > 0, 3.0, np.where(outcomes == 0, 1.0, 0.0))
    return {
        f"{prefix}_matches_last_{window}": len(recent),
        f"{prefix}_win_rate_last_{window}": float(np.mean(outcomes > 0)),
        f"{prefix}_avg_goals_for_last_{window}": float(np.mean(goals_for)),
        f"{prefix}_avg_goals_against_last_{window}": float(np.mean(goals_against)),
        f"{prefix}_avg_points_last_{window}": float(np.mean(points)),
        f"{prefix}_avg_penalties_for_last_{window}": float(np.mean(penalties_for)),
        f"{prefix}_avg_late_goals_for_last_{window}": float(np.mean(late_goals_for)),
    }


def elo_expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def elo_k_factor(tournament: str) -> float:
    if tournament == "FIFA World Cup":
        return 50.0
    if tournament == "Friendly":
        return 20.0
    return 35.0


def build_chronological_features(matches: pd.DataFrame) -> pd.DataFrame:
    team_histories: Dict[str, TeamHistory] = defaultdict(TeamHistory)
    h2h_histories: Dict[frozenset, HeadToHeadHistory] = defaultdict(HeadToHeadHistory)
    rows = []

    for match in matches.itertuples(index=False):
        home_team = match.home_team
        away_team = match.away_team
        home_state = team_histories[home_team]
        away_state = team_histories[away_team]

        features = {
            "home_elo_pre": home_state.elo,
            "away_elo_pre": away_state.elo,
            "elo_diff_pre": home_state.elo - away_state.elo,
        }
        for window in TEAM_FEATURE_WINDOWS:
            features.update(summarize_recent_form(home_state.matches, window, "home"))
            features.update(summarize_recent_form(away_state.matches, window, "away"))

        pair_key = frozenset((home_team, away_team))
        h2h = h2h_histories[pair_key]
        h2h_total = h2h.total
        if h2h_total:
            features["h2h_matches"] = h2h_total
            features["h2h_home_team_win_rate"] = h2h.home_wins / h2h_total
            features["h2h_away_team_win_rate"] = h2h.away_wins / h2h_total
            features["h2h_draw_rate"] = h2h.draws / h2h_total
        else:
            features["h2h_matches"] = 0
            features["h2h_home_team_win_rate"] = 0.0
            features["h2h_away_team_win_rate"] = 0.0
            features["h2h_draw_rate"] = 0.0
        rows.append(features)

        home_score = int(match.home_score)
        away_score = int(match.away_score)
        home_result = int(np.sign(home_score - away_score))
        away_result = -home_result

        home_state.matches.append(
            (
                home_result,
                home_score,
                away_score,
                int(match.home_penalty_goals),
                int(match.home_late_goals),
            )
        )
        away_state.matches.append(
            (
                away_result,
                away_score,
                home_score,
                int(match.away_penalty_goals),
                int(match.away_late_goals),
            )
        )
        max_history = max(TEAM_FEATURE_WINDOWS)
        while len(home_state.matches) > max_history:
            home_state.matches.popleft()
        while len(away_state.matches) > max_history:
            away_state.matches.popleft()

        expected_home = elo_expected_score(home_state.elo, away_state.elo)
        actual_home = 1.0 if home_result > 0 else 0.5 if home_result == 0 else 0.0
        goal_diff_multiplier = np.log1p(abs(home_score - away_score)) if home_score != away_score else 1.0
        k_value = elo_k_factor(match.tournament) * float(goal_diff_multiplier)
        elo_delta = k_value * (actual_home - expected_home)
        home_state.elo += elo_delta
        away_state.elo -= elo_delta

        if home_result > 0:
            h2h.home_wins += 1
        elif home_result < 0:
            h2h.away_wins += 1
        else:
            h2h.draws += 1

    feature_df = pd.DataFrame(rows, index=matches.index)
    integer_columns = [column for column in feature_df.columns if column.endswith("_matches") or "_matches_last_" in column]
    for column in integer_columns:
        feature_df[column] = feature_df[column].astype("Int16")
    float_columns = feature_df.columns.difference(integer_columns)
    feature_df[float_columns] = feature_df[float_columns].astype("float32")
    return pd.concat([matches, feature_df], axis=1)


def add_date_features(matches: pd.DataFrame) -> pd.DataFrame:
    matches["year"] = matches["date"].dt.year.astype("Int16")
    matches["month"] = matches["date"].dt.month.astype("Int8")
    matches["day_of_week"] = matches["date"].dt.dayofweek.astype("Int8")
    return matches


def encode_categoricals(matches: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    import json

    encoded = matches.copy()
    category_columns = ["home_team", "away_team", "tournament", "country"]
    encoding_maps = {}

    for column in category_columns:
        categories = sorted(encoded[column].dropna().astype(str).unique())
        encoding_maps[column] = {category: code for code, category in enumerate(categories)}
        encoded[f"{column}_code"] = (
            encoded[column].astype(str).map(encoding_maps[column]).fillna(-1).astype("int16")
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(encoding_maps, indent=2, sort_keys=True), encoding="utf-8")
    return encoded


def finalize_model_frame(matches: pd.DataFrame, encoding_map_output: Path) -> pd.DataFrame:
    matches = add_date_features(matches)
    matches = encode_categoricals(matches, encoding_map_output)

    boolean_columns = [
        "neutral",
        "is_world_cup",
        "is_friendly",
        "is_competitive",
    ]
    for column in boolean_columns:
        matches[column] = matches[column].astype("int8")

    drop_columns = [
        "date",
        "home_team",
        "away_team",
        "tournament",
        "city",
        "country",
        "home_score",
        "away_score",
        "scorer_events",
        "penalty_goals",
        "own_goals",
        "late_goals",
        "stoppage_goals",
        "home_goal_events",
        "away_goal_events",
        "home_penalty_goals",
        "away_penalty_goals",
        "home_late_goals",
        "away_late_goals",
        "goal_difference",
        "regular_time_target",
        "regular_time_draw",
        "home_shootout_win",
        "away_shootout_win",
        "shootout_winner",
    ]
    final_df = matches.drop(columns=[column for column in drop_columns if column in matches.columns])

    numeric_columns = final_df.select_dtypes(include=["number", "bool"]).columns
    final_df = final_df[numeric_columns].copy()

    target_columns = ["target"]
    feature_columns = [column for column in final_df.columns if column not in target_columns]
    ordered_columns = feature_columns + target_columns
    return final_df[ordered_columns]


def prepare_training_data(data_dir: Path, encoding_map_output: Path) -> pd.DataFrame:
    results, shootouts, goalscorers = load_and_clean(data_dir)
    goal_features = aggregate_goalscorers(goalscorers)
    matches = merge_match_context(results, shootouts, goal_features)
    matches = build_chronological_features(matches)
    return finalize_model_frame(matches, encoding_map_output)


def main() -> None:
    args = parse_args()
    final_df = prepare_training_data(args.data_dir, args.encoding_map_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output, index=False)
    print(f"Wrote {len(final_df):,} rows x {len(final_df.columns):,} columns to {args.output}")
    print(f"Wrote categorical encoding maps to {args.encoding_map_output}")


if __name__ == "__main__":
    main()
