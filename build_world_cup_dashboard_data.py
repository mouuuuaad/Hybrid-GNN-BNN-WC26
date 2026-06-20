#!/usr/bin/env python3
"""Build dashboard-ready data from World Cup 2026 simulation results.

This script is the Python bridge between the ML/simulation pipeline and the
static HTML dashboard. It reads `world_cup_2026_simulation_results.csv`, applies
the configured 12-group World Cup structure, derives projected group standings
and a deterministic knockout bracket, then writes:

  - dashboard_data/world_cup_dashboard_data.json
  - dashboard_data/world_cup_dashboard_data.js

The JS file assigns `window.WORLD_CUP_DASHBOARD_DATA`, so a plain HTML file can
load it directly with a script tag when you do not want a build system.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd


WORLD_CUP_2026_GROUPS: Dict[str, List[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Ivory Coast", "Ecuador", "Curaçao"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Iran", "New Zealand", "Belgium", "Egypt"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}


STAGE_COLUMNS = {
    "Reach R32 %": "r32",
    "Reach R16 %": "r16",
    "Reach QF %": "qf",
    "Reach SF %": "sf",
    "Reach Final %": "final",
    "Win Tournament %": "win",
}


@dataclass
class TeamResult:
    team: str
    r32: float
    r16: float
    qf: float
    sf: float
    final: float
    win: float


@dataclass
class GroupRow:
    team: str
    group: str
    points: int
    gd: int
    gf: int
    rank: int
    advances: bool
    advance_reason: str
    win_probability: float


@dataclass
class MatchNode:
    id: str
    round: str
    home: str
    away: str
    winner: str
    score: str
    home_probability: float
    away_probability: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frontend data for the World Cup dashboard.")
    parser.add_argument("--results-csv", type=Path, default=Path("world_cup_2026_simulation_results.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("dashboard_data"))
    return parser.parse_args()


def load_results(path: Path) -> Dict[str, TeamResult]:
    if not path.exists():
        raise FileNotFoundError(f"Simulation results CSV not found: {path}")
    df = pd.read_csv(path).rename(columns=STAGE_COLUMNS)
    required = {"Team", *STAGE_COLUMNS.values()}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required result columns: {sorted(missing)}")

    results = {}
    for row in df.itertuples(index=False):
        results[row.Team] = TeamResult(
            team=row.Team,
            r32=round(float(row.r32), 2),
            r16=round(float(row.r16), 2),
            qf=round(float(row.qf), 2),
            sf=round(float(row.sf), 2),
            final=round(float(row.final), 2),
            win=round(float(row.win), 2),
        )
    return results


def probability(results: Dict[str, TeamResult], team: str, stage: str) -> float:
    return getattr(results[team], stage)


def projected_match_score(results: Dict[str, TeamResult], home: str, away: str, winner: str) -> Tuple[int, int]:
    loser = away if winner == home else home
    edge = probability(results, winner, "win") - probability(results, loser, "win")
    margin = max(1, min(4, 1 + int(edge // 4)))
    loser_goals = 1 if probability(results, loser, "win") >= 3.5 else 0
    if winner == home:
        return loser_goals + margin, loser_goals
    return loser_goals, loser_goals + margin


def projected_winner(results: Dict[str, TeamResult], home: str, away: str, next_stage: str) -> str:
    home_score = probability(results, home, next_stage) + probability(results, home, "win") * 1.65
    away_score = probability(results, away, next_stage) + probability(results, away, "win") * 1.65
    return home if home_score >= away_score else away


def build_group_tables(results: Dict[str, TeamResult]) -> Dict[str, List[GroupRow]]:
    raw_groups = {}
    third_rows = []

    for group_name, teams in WORLD_CUP_2026_GROUPS.items():
        rows = []
        ordered = sorted(
            teams,
            key=lambda team: (
                probability(results, team, "r32"),
                probability(results, team, "r16"),
                probability(results, team, "win"),
            ),
            reverse=True,
        )
        for index, team in enumerate(ordered, start=1):
            r32 = probability(results, team, "r32")
            r16 = probability(results, team, "r16")
            win = probability(results, team, "win")
            points = max(0, min(9, round(1.2 + r32 / 17 + r16 / 30)))
            gd = round((r32 - 50) / 14 + win / 2)
            gf = max(1, round(2.1 + r32 / 32 + r16 / 45 + win / 4))
            row = GroupRow(
                team=team,
                group=group_name,
                points=int(points),
                gd=int(gd),
                gf=int(gf),
                rank=index,
                advances=index <= 2,
                advance_reason="Top 2" if index <= 2 else "",
                win_probability=win,
            )
            rows.append(row)
            if index == 3:
                third_rows.append(row)
        raw_groups[group_name] = rows

    best_thirds = sorted(
        third_rows,
        key=lambda row: (row.points, row.gd, row.gf, row.win_probability),
        reverse=True,
    )[:8]
    best_third_teams = {row.team for row in best_thirds}

    for rows in raw_groups.values():
        for row in rows:
            if row.team in best_third_teams:
                row.advances = True
                row.advance_reason = "Best 3rd"

    return raw_groups


def qualifiers_from_groups(group_tables: Dict[str, List[GroupRow]]) -> List[GroupRow]:
    qualifiers = []
    for rows in group_tables.values():
        qualifiers.extend(row for row in rows if row.advances)
    return sorted(
        qualifiers,
        key=lambda row: (row.rank * -1, row.points, row.gd, row.gf, row.win_probability),
        reverse=True,
    )


def seed_round_of_32(qualifiers: Sequence[GroupRow]) -> List[str]:
    ordered = sorted(
        qualifiers,
        key=lambda row: (
            -row.rank,
            row.points,
            row.gd,
            row.gf,
            row.win_probability,
        ),
        reverse=True,
    )
    seed_order = [0, 31, 15, 16, 7, 24, 8, 23, 3, 28, 12, 19, 4, 27, 11, 20, 1, 30, 14, 17, 6, 25, 9, 22, 2, 29, 13, 18, 5, 26, 10, 21]
    return [ordered[index].team for index in seed_order]


def build_round(
    results: Dict[str, TeamResult],
    teams: Sequence[str],
    round_name: str,
    next_stage: str,
) -> Tuple[List[MatchNode], List[str]]:
    matches = []
    winners = []
    for index in range(0, len(teams), 2):
        home = teams[index]
        away = teams[index + 1]
        winner = projected_winner(results, home, away, next_stage)
        home_goals, away_goals = projected_match_score(results, home, away, winner)
        matches.append(
            MatchNode(
                id=f"{round_name}-{index // 2 + 1}",
                round=round_name,
                home=home,
                away=away,
                winner=winner,
                score=f"{home_goals}-{away_goals}",
                home_probability=round(probability(results, home, next_stage), 2),
                away_probability=round(probability(results, away, next_stage), 2),
            )
        )
        winners.append(winner)
    return matches, winners


def build_bracket(results: Dict[str, TeamResult], group_tables: Dict[str, List[GroupRow]]) -> Dict[str, object]:
    bracket_teams = seed_round_of_32(qualifiers_from_groups(group_tables))
    r32, r16_teams = build_round(results, bracket_teams, "R32", "r16")
    r16, qf_teams = build_round(results, r16_teams, "R16", "qf")
    qf, sf_teams = build_round(results, qf_teams, "QF", "sf")
    sf, final_teams = build_round(results, sf_teams, "SF", "final")
    final, champion = build_round(results, final_teams, "Final", "win")
    return {
        "rounds": [
            {"name": "Round of 32", "key": "r32", "matches": [asdict(match) for match in r32]},
            {"name": "Round of 16", "key": "r16", "matches": [asdict(match) for match in r16]},
            {"name": "Quarter-finals", "key": "qf", "matches": [asdict(match) for match in qf]},
            {"name": "Semi-finals", "key": "sf", "matches": [asdict(match) for match in sf]},
            {"name": "Final", "key": "final", "matches": [asdict(match) for match in final]},
        ],
        "champion": champion[0],
    }


def build_chart_data(results: Dict[str, TeamResult]) -> Dict[str, object]:
    teams = sorted(results.values(), key=lambda row: row.win, reverse=True)
    top10 = teams[:10]
    return {
        "winProbabilityTop10": [
            {"team": row.team, "value": row.win, "r32": row.r32, "final": row.final}
            for row in top10
        ],
        "stageFunnelTop5": [
            {
                "team": row.team,
                "stages": {
                    "R32": row.r32,
                    "R16": row.r16,
                    "QF": row.qf,
                    "SF": row.sf,
                    "Final": row.final,
                    "Win": row.win,
                },
            }
            for row in teams[:5]
        ],
        "allTeams": [asdict(row) for row in teams],
    }


def build_payload(results: Dict[str, TeamResult]) -> Dict[str, object]:
    group_tables = build_group_tables(results)
    bracket = build_bracket(results, group_tables)
    chart_data = build_chart_data(results)
    return {
        "source": "world_cup_2026_simulation_results.csv",
        "model": "GNN + Bayesian Neural Network Monte Carlo",
        "simulations": 10_000,
        "groups": {
            group_name: [asdict(row) for row in rows]
            for group_name, rows in group_tables.items()
        },
        "bracket": bracket,
        "charts": chart_data,
        "summary": {
            "favorite": chart_data["winProbabilityTop10"][0],
            "teams": len(results),
            "qualifiedForR32": 32,
        },
    }


def write_outputs(payload: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "world_cup_dashboard_data.json"
    js_path = output_dir / "world_cup_dashboard_data.js"
    json_text = json.dumps(payload, ensure_ascii=False, indent=2)
    json_path.write_text(json_text, encoding="utf-8")
    js_path.write_text(
        "window.WORLD_CUP_DASHBOARD_DATA = "
        + json.dumps(payload, ensure_ascii=False)
        + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {js_path}")


def main() -> None:
    args = parse_args()
    results = load_results(args.results_csv)
    missing = sorted({team for teams in WORLD_CUP_2026_GROUPS.values() for team in teams} - set(results))
    if missing:
        raise ValueError(f"These group teams are missing from the simulation CSV: {missing}")
    payload = build_payload(results)
    write_outputs(payload, args.output_dir)


if __name__ == "__main__":
    main()
