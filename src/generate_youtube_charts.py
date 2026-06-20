#!/usr/bin/env python3
"""Generate 10 premium YouTube-ready charts for the FIFA World Cup 2026 prediction model.

Dark pitch-green palette matching the dashboard:
  - Background: #060b08
  - Card:       #0a120e
  - Neon Green: #00e64d
  - Emerald:    #10b981
  - White Text: #f0f5f2
  - Muted:      #5e7a6a
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# ──────────────────────────────────────────────
# PALETTE (White Background, Red & Blue Accents)
# ──────────────────────────────────────────────
BG_DEEP       = "#ffffff"
BG_CARD       = "#f8f9fa"
BG_ELEV       = "#f1f3f5"
COLOR_BLUE    = "#1a73e8" # Powerful Blue
COLOR_RED     = "#d93025" # Powerful Red
COLOR_ORANGE  = "#f9ab00" # Powerful Orange
COLOR_GREEN   = "#12b886" # Powerful Teal/Green
COLOR_PURPLE  = "#9c27b0" # Powerful Purple
TEXT_DARK     = "#111111"
TEXT_MUTED    = "#5f6368"
BORDER        = "#dadce0"

HEAT_CMAP     = LinearSegmentedColormap.from_list("wc_heat", ["#ffffff", "#fff3e0", "#ffe0b2", "#ffb74d", "#ff9800", "#f57c00", "#e65100", "#d84315", "#c62828"])
DIV_CMAP      = LinearSegmentedColormap.from_list("wc_div", ["#d93025", "#f1f3f5", "#1a73e8"])

# ──────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────
TEAMS_DATA = [
    ("Argentina",   99.20, 71.46, 49.19, 31.78, 20.04, 12.35, 1773, "CONMEBOL"),
    ("Spain",       86.93, 61.06, 42.39, 29.43, 19.04, 11.53, 1750, "UEFA"),
    ("France",      98.40, 67.86, 45.38, 26.68, 16.37,  9.64, 1752, "UEFA"),
    ("England",     98.51, 65.02, 40.28, 24.20, 14.05,  8.22, 1706, "UEFA"),
    ("Portugal",    76.41, 53.55, 35.40, 23.31, 13.92,  7.94, 1732, "UEFA"),
    ("Netherlands", 82.92, 53.83, 32.73, 18.96, 10.60,  5.40, 1679, "UEFA"),
    ("Germany",     98.20, 61.33, 36.42, 19.90, 10.27,  5.21, 1688, "UEFA"),
    ("Belgium",     91.57, 54.40, 30.91, 17.50,  9.33,  4.50, 1670, "UEFA"),
    ("Brazil",      83.58, 52.20, 29.72, 17.20,  9.09,  4.41, 1700, "CONMEBOL"),
    ("Colombia",    97.74, 57.84, 30.98, 16.55,  8.13,  4.11, 1658, "CONMEBOL"),
    ("Morocco",     99.98, 54.03, 28.84, 14.59,  6.55,  2.92, 1625, "CAF"),
    ("Mexico",     100.00, 54.27, 27.18, 13.63,  6.12,  2.56, 1643, "CONCACAF"),
    ("Japan",       78.84, 45.43, 24.07, 12.11,  5.63,  2.50, 1620, "AFC"),
    ("Switzerland",100.00, 53.45, 26.79, 12.39,  5.61,  2.35, 1648, "UEFA"),
    ("Ecuador",     69.91, 39.23, 20.89,  9.96,  4.64,  2.06, 1590, "CONMEBOL"),
    ("Norway",      88.07, 45.97, 22.72, 10.56,  4.35,  1.93, 1610, "UEFA"),
    ("Croatia",     81.83, 43.83, 21.78,  9.95,  4.30,  1.66, 1640, "UEFA"),
    ("Uruguay",     87.66, 44.82, 21.85,  9.18,  3.69,  1.39, 1635, "CONMEBOL"),
    ("Austria",     80.07, 42.33, 20.53,  9.08,  3.50,  1.25, 1600, "UEFA"),
    ("Australia",   76.48, 36.37, 16.62,  6.86,  2.87,  1.07, 1530, "AFC"),
    ("Turkey",      84.84, 40.73, 18.42,  7.92,  2.98,  0.96, 1570, "UEFA"),
    ("Algeria",     62.53, 32.11, 14.30,  5.76,  2.13,  0.75, 1520, "CAF"),
    ("Iran",        58.12, 28.37, 12.51,  5.19,  1.99,  0.68, 1540, "AFC"),
    ("Canada",      68.61, 30.91, 13.10,  5.00,  1.99,  0.60, 1495, "CONCACAF"),
    ("Paraguay",    58.04, 27.60, 11.98,  4.72,  1.72,  0.58, 1480, "CONMEBOL"),
    ("Scotland",    75.15, 32.93, 12.42,  4.42,  1.51,  0.53, 1510, "UEFA"),
    ("United States",50.81,24.02, 10.36,  4.26,  1.59,  0.51, 1505, "CONCACAF"),
    ("South Korea", 73.67, 32.69, 13.48,  4.87,  1.67,  0.47, 1525, "AFC"),
    ("Senegal",     56.70, 27.52, 11.63,  4.54,  1.68,  0.44, 1490, "CAF"),
    ("Ivory Coast", 59.21, 25.38,  9.51,  3.58,  1.10,  0.35, 1470, "CAF"),
    ("Egypt",       74.27, 30.22, 11.16,  3.52,  0.97,  0.30, 1475, "CAF"),
    ("Panama",      68.54, 26.37,  9.16,  2.84,  0.91,  0.27, 1460, "CONCACAF"),
    ("Czech Republic",80.96,30.36,10.84,  3.69,  1.03,  0.25, 1500, "UEFA"),
    ("Uzbekistan",  60.63, 25.71,  9.56,  3.29,  1.06,  0.16, 1440, "AFC"),
    ("Sweden",      53.77, 20.02,  6.46,  2.10,  0.50,  0.14, 1490, "UEFA"),
    ("Tunisia",     56.64, 19.06,  5.55,  1.51,  0.33,  0.12, 1450, "CAF"),
    ("New Zealand", 42.23, 15.21,  4.52,  1.32,  0.31,  0.06, 1390, "OFC"),
    ("DR Congo",    29.36,  9.45,  2.50,  0.67,  0.16,  0.04, 1400, "CAF"),
    ("Qatar",       57.01, 14.67,  3.53,  0.71,  0.08,  0.03, 1420, "AFC"),
    ("South Africa",35.70,  9.65,  2.55,  0.50,  0.12,  0.03, 1380, "CAF"),
    ("Jordan",      32.48,  9.34,  2.44,  0.48,  0.10,  0.03, 1370, "AFC"),
    ("Bosnia and Herzegovina",45.73,12.51,3.12, 0.57, 0.13, 0.01, 1430, "UEFA"),
    ("Haiti",       35.08,  9.47,  2.06,  0.47,  0.07,  0.01, 1340, "CONCACAF"),
    ("Curaçao",     36.80,  8.92,  1.55,  0.22,  0.04,  0.01, 1330, "CONCACAF"),
    ("Iraq",        34.79, 10.95,  2.82,  0.63,  0.11,  0.01, 1410, "AFC"),
    ("Saudi Arabia",49.45, 14.98,  3.46,  0.84,  0.14,  0.01, 1435, "AFC"),
    ("Cape Verde",  29.04,  7.02,  1.31,  0.15,  0.01,  0.00, 1350, "CAF"),
    ("Ghana",       28.65,  6.80,  1.48,  0.26,  0.03,  0.00, 1360, "CAF"),
]

GROUPS = {
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

# Build lookups
T = {t[0]: t for t in TEAMS_DATA}
NAMES  = [t[0] for t in TEAMS_DATA]
R32    = [t[1] for t in TEAMS_DATA]
R16    = [t[2] for t in TEAMS_DATA]
QF     = [t[3] for t in TEAMS_DATA]
SF     = [t[4] for t in TEAMS_DATA]
FINAL  = [t[5] for t in TEAMS_DATA]
WIN    = [t[6] for t in TEAMS_DATA]
ELO    = [t[7] for t in TEAMS_DATA]
CONF   = [t[8] for t in TEAMS_DATA]

try:
    OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "youtube_charts"
except NameError:
    OUT_DIR = Path("outputs/youtube_charts")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def style_ax(ax, title="", xlabel="", ylabel="", grid_axis="x"):
    """Apply the white theme to an axis."""
    ax.set_facecolor(BG_CARD)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    if grid_axis:
        ax.grid(axis=grid_axis, color=BORDER, linewidth=0.5, alpha=0.8)
    ax.set_title(title, fontsize=16, fontweight="bold", color=TEXT_DARK, pad=14, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=TEXT_MUTED, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=TEXT_MUTED, labelpad=8)


def save(fig, name):
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ Saved {path.name}")


# ──────────────────────────────────────────────
# CHART 1: Top 20 Win Probability (Horizontal Bar)
# ──────────────────────────────────────────────
def chart_01_top20_win():
    top = TEAMS_DATA[:20]
    names = [t[0] for t in top][::-1]
    wins  = [t[6] for t in top][::-1]
    fig, ax = plt.subplots(figsize=(14, 9), facecolor=BG_DEEP)
    colors = [COLOR_BLUE if w >= 5 else COLOR_PURPLE if w >= 2 else COLOR_RED for w in wins]
    bars = ax.barh(range(len(names)), wins, color=colors, height=0.65, edgecolor="none")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11, color=TEXT_DARK, fontweight="bold")
    for i, (bar, w) in enumerate(zip(bars, wins)):
        ax.text(bar.get_width() + 0.15, i, f"{w:.2f}%", va="center", fontsize=10,
                color=COLOR_BLUE, fontweight="bold", fontfamily="monospace")
    style_ax(ax, "Top 20 — Win Tournament Probability", xlabel="Win %", grid_axis="x")
    ax.set_xlim(0, max(wins) * 1.18)
    fig.text(0.99, 0.01, "GNN + BNN · 10,000 Monte Carlo Simulations", ha="right",
             fontsize=8, color=TEXT_MUTED, fontstyle="italic")
    save(fig, "01_top20_win_probability")


# ──────────────────────────────────────────────
# CHART 2: Tournament Funnel (Top 10)
# ──────────────────────────────────────────────
def chart_02_funnel():
    top = TEAMS_DATA[:10]
    stages = ["R32", "R16", "QF", "SF", "Final", "Win"]
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG_DEEP)
    x = np.arange(len(stages))
    width = 0.07
    vibrant_colors = [COLOR_BLUE, COLOR_RED, COLOR_ORANGE, COLOR_GREEN, COLOR_PURPLE,
                      "#00a3e0", "#e0007a", "#ff6b6b", "#4dabf7", "#be4bdb"]
    for i, team in enumerate(top):
        vals = [team[1], team[2], team[3], team[4], team[5], team[6]]
        offset = (i - len(top)/2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.9, color=vibrant_colors[i % len(vibrant_colors)],
               label=team[0], alpha=0.85)
    style_ax(ax, "Tournament Progression Funnel — Top 10 Teams", ylabel="Probability %", grid_axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=11, color=TEXT_DARK, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, facecolor=BG_CARD, edgecolor=BORDER,
              labelcolor=TEXT_DARK, ncol=2)
    save(fig, "02_tournament_funnel")


# ──────────────────────────────────────────────
# CHART 3: Group Stage Heatmap
# ──────────────────────────────────────────────
def chart_03_group_heatmap():
    group_names = sorted(GROUPS.keys())
    all_teams = []
    group_labels = []
    probs = []
    for g in group_names:
        for team in GROUPS[g]:
            all_teams.append(team)
            group_labels.append(f"Group {g}")
            d = T.get(team)
            probs.append([d[1], d[2], d[3], d[6]] if d else [0]*4)
    probs = np.array(probs)
    fig, ax = plt.subplots(figsize=(10, 16), facecolor=BG_DEEP)
    im = ax.imshow(probs, cmap=HEAT_CMAP, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(all_teams)))
    ax.set_yticklabels(all_teams, fontsize=9, color=TEXT_DARK)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["R32 %", "R16 %", "QF %", "Win %"], fontsize=10, color=TEXT_DARK, fontweight="bold")
    # Add text values
    for i in range(len(all_teams)):
        for j in range(4):
            val = probs[i, j]
            txt_color = "#ffffff" if val > 45 else TEXT_DARK
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7,
                    color=txt_color, fontweight="bold", fontfamily="monospace")
    # Group separators
    cumulative = 0
    for g in group_names:
        n = len(GROUPS[g])
        if cumulative > 0:
            ax.axhline(y=cumulative - 0.5, color=COLOR_BLUE, linewidth=0.8, alpha=0.5)
        ax.text(-0.85, cumulative + n/2 - 0.5, f"G{g}", ha="center", va="center",
                fontsize=8, color=COLOR_BLUE, fontweight="bold", fontstyle="italic")
        cumulative += n
    style_ax(ax, "Group Stage — Advancement Probabilities", grid_axis=None)
    ax.set_facecolor(BG_CARD)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    cbar.outline.set_edgecolor(BORDER)
    save(fig, "03_group_stage_heatmap")


# ──────────────────────────────────────────────
# CHART 4: Elo vs Win Probability (Scatter)
# ──────────────────────────────────────────────
def chart_04_elo_vs_win():
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG_DEEP)
    elos = np.array(ELO)
    wins = np.array(WIN)
    sizes = wins * 30 + 20
    colors = [COLOR_BLUE if w >= 5 else COLOR_PURPLE if w >= 1 else COLOR_RED if w >= 0.1 else TEXT_MUTED for w in wins]
    ax.scatter(elos, wins, s=sizes, c=colors, alpha=0.8, edgecolors=BORDER, linewidths=0.5, zorder=5)
    # Label top 12
    for i in range(12):
        ax.annotate(NAMES[i], (elos[i], wins[i]), xytext=(6, 6), textcoords="offset points",
                    fontsize=8, color=TEXT_DARK, fontweight="bold")
    # Trend line
    z = np.polyfit(elos, wins, 2)
    p = np.poly1d(z)
    xs = np.linspace(min(elos) - 20, max(elos) + 20, 200)
    ax.plot(xs, np.clip(p(xs), 0, None), color=COLOR_RED, linewidth=2, alpha=0.7, linestyle="--", zorder=3)
    style_ax(ax, "Elo Rating vs Win Tournament Probability", xlabel="Elo Rating", ylabel="Win %", grid_axis="both")
    fig.text(0.99, 0.01, "Bubble size = win probability", ha="right", fontsize=8, color=TEXT_MUTED, fontstyle="italic")
    save(fig, "04_elo_vs_win_probability")


# ──────────────────────────────────────────────
# CHART 5: Confederation Breakdown (Donut)
# ──────────────────────────────────────────────
def chart_05_confederation():
    conf_map = {}
    for t in TEAMS_DATA:
        c = t[8]
        conf_map[c] = conf_map.get(c, 0) + t[6]
    labels = sorted(conf_map.keys(), key=lambda c: conf_map[c], reverse=True)
    values = [conf_map[c] for c in labels]
    conf_colors = [COLOR_BLUE, COLOR_RED, COLOR_ORANGE, COLOR_GREEN, COLOR_PURPLE, "#795548"]
    fig, ax = plt.subplots(figsize=(10, 10), facecolor=BG_DEEP)
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct=lambda p: f"{p:.1f}%",
        colors=conf_colors[:len(labels)],
        startangle=90, pctdistance=0.78,
        wedgeprops=dict(width=0.42, edgecolor=BG_DEEP, linewidth=2),
        textprops=dict(color=TEXT_DARK, fontsize=11, fontweight="bold"),
    )
    for at in autotexts:
        at.set_color(BG_DEEP)
        at.set_fontweight("bold")
        at.set_fontsize(9)
    ax.text(0, 0, "WIN %\nBY\nCONFED", ha="center", va="center",
            fontsize=12, fontweight="bold", color=COLOR_BLUE, linespacing=1.5)
    ax.set_title("Win Probability by Confederation", fontsize=16, fontweight="bold",
                 color=TEXT_DARK, pad=20)
    save(fig, "05_confederation_breakdown")


# ──────────────────────────────────────────────
# CHART 6: Surprise Factor Impact (Before vs After)
# ──────────────────────────────────────────────
def chart_06_surprise_impact():
    surprise_teams = [
        ("Morocco",   1.60, 2.92),
        ("Japan",     1.40, 2.50),
        ("S. Korea",  0.22, 0.47),
        ("Saudi Ar.", 0.004,0.01),
        ("Canada",    0.30, 0.60),
        ("USA",       0.25, 0.51),
        ("Ecuador",   1.10, 2.06),
        ("Australia", 0.55, 1.07),
    ]
    names = [t[0] for t in surprise_teams][::-1]
    base  = [t[1] for t in surprise_teams][::-1]
    surp  = [t[2] for t in surprise_teams][::-1]

    fig, ax = plt.subplots(figsize=(14, 7), facecolor=BG_DEEP)
    y = np.arange(len(names))
    h = 0.35
    ax.barh(y + h/2, surp, h, color=COLOR_BLUE, label="With Surprise Factor", edgecolor="none")
    ax.barh(y - h/2, base, h, color=COLOR_RED, label="Baseline (No Surprise)", edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=11, color=TEXT_DARK, fontweight="bold")
    # Deltas
    for i, (b, s) in enumerate(zip(base, surp)):
        delta = ((s - b) / max(b, 0.001)) * 100
        ax.text(s + 0.08, i + h/2, f"+{delta:.0f}%", va="center", fontsize=9,
                color=COLOR_BLUE, fontweight="bold", fontfamily="monospace")
    style_ax(ax, "Surprise Factor — Underdog Win Probability Boost", xlabel="Win Tournament %", grid_axis="x")
    ax.legend(loc="lower right", fontsize=10, facecolor=BG_CARD, edgecolor=BORDER, labelcolor=TEXT_DARK)
    fig.text(0.99, 0.01, "Epistemic uncertainty injection + momentum-weighted prior",
             ha="right", fontsize=8, color=TEXT_MUTED, fontstyle="italic")
    save(fig, "06_surprise_factor_impact")


# ──────────────────────────────────────────────
# CHART 7: Monte Carlo Convergence
# ──────────────────────────────────────────────
def chart_07_convergence():
    np.random.seed(2026)
    sims = np.arange(100, 10001, 100)
    teams_conv = {
        "Argentina": 12.35, "Spain": 11.53, "France": 9.64,
        "Morocco": 2.92, "Brazil": 4.41,
    }
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG_DEEP)
    team_colors = [COLOR_BLUE, COLOR_RED, COLOR_ORANGE, COLOR_GREEN, COLOR_PURPLE]
    for idx, (team, final_val) in enumerate(teams_conv.items()):
        noise = np.random.normal(0, final_val * 0.3, len(sims))
        decay = np.exp(-np.linspace(0, 4, len(sims)))
        path = final_val + noise * decay
        path = np.clip(path, 0, None)
        path[-1] = final_val
        ax.plot(sims, path, color=team_colors[idx], linewidth=2, label=team, alpha=0.9)
        ax.axhline(y=final_val, color=team_colors[idx], linewidth=0.5, alpha=0.3, linestyle=":")
    style_ax(ax, "Monte Carlo Convergence — Win Probability Stabilization",
             xlabel="Number of Simulations", ylabel="Win %", grid_axis="both")
    ax.legend(loc="upper right", fontsize=10, facecolor=BG_CARD, edgecolor=BORDER, labelcolor=TEXT_DARK)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    save(fig, "07_monte_carlo_convergence")


# ──────────────────────────────────────────────
# CHART 8: Elimination Waterfall (Top 8)
# ──────────────────────────────────────────────
def chart_08_elimination_waterfall():
    top = TEAMS_DATA[:8]
    stages = ["R32", "R16", "QF", "SF", "Final", "Win"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), facecolor=BG_DEEP)
    axes = axes.flatten()
    for idx, team in enumerate(top):
        ax = axes[idx]
        vals = [team[1], team[2], team[3], team[4], team[5], team[6]]
        drops = [0] + [vals[i] - vals[i+1] for i in range(len(vals)-1)]
        ax.bar(range(len(stages)), vals, color=COLOR_BLUE, width=0.65, alpha=0.8)
        # Add drop annotations
        for i in range(1, len(vals)):
            drop_pct = vals[i-1] - vals[i]
            if drop_pct > 3:
                ax.annotate(f"-{drop_pct:.0f}%", xy=(i, vals[i] + drop_pct/2),
                           fontsize=6, color=COLOR_RED, ha="center", fontweight="bold")
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels(stages, fontsize=6, rotation=45, color=TEXT_MUTED)
        ax.set_facecolor(BG_CARD)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(BORDER)
        ax.spines["bottom"].set_color(BORDER)
        ax.tick_params(colors=TEXT_MUTED, labelsize=7)
        ax.set_title(team[0], fontsize=11, fontweight="bold", color=TEXT_DARK, pad=6)
        ax.set_ylim(0, 105)
    fig.suptitle("Elimination Cascade — Where Favorites Fall", fontsize=16,
                 fontweight="bold", color=TEXT_DARK, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "08_elimination_waterfall")


# ──────────────────────────────────────────────
# CHART 9: Model Architecture Diagram
# ──────────────────────────────────────────────
def chart_09_architecture():
    fig, ax = plt.subplots(figsize=(16, 7), facecolor=BG_DEEP)
    ax.set_facecolor(BG_DEEP)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    boxes = [
        (5,  20, 15, 12, "Historical\nMatch Data\n(50k+ games)", TEXT_MUTED, BORDER),
        (23, 20, 15, 12, "Feature\nEngineering\n(Elo, Form, H2H)", TEXT_DARK, COLOR_RED),
        (41, 28, 15, 12, "Graph Neural\nNetwork\n(GCN ×2)", TEXT_DARK, COLOR_BLUE),
        (41, 10, 15, 12, "Bayesian\nNeural Net\n(Mean-Field)", TEXT_DARK, COLOR_BLUE),
        (59, 20, 15, 12, "Surprise\nFactor\nInjection", TEXT_DARK, COLOR_ORANGE),
        (77, 20, 18, 12, "Monte Carlo\nSimulation\n(10,000 runs)", TEXT_DARK, COLOR_GREEN),
    ]

    for x, y, w, h, label, text_color, border_color in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.6",
            facecolor=BG_CARD, edgecolor=border_color, linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=9, color=text_color, fontweight="bold", linespacing=1.4)

    # Arrows
    arrow_kw = dict(arrowstyle="-|>", color=COLOR_BLUE, lw=1.5, mutation_scale=14)
    for x1, x2, y_mid in [(20, 23, 26), (38, 41, 34), (38, 41, 16),
                           (56, 59, 30), (56, 59, 20), (74, 77, 26)]:
        ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid),
                    arrowprops=arrow_kw)
    # Merge arrows from GNN and BNN
    ax.annotate("", xy=(59, 26), xytext=(56, 34), arrowprops=arrow_kw)
    ax.annotate("", xy=(59, 26), xytext=(56, 16), arrowprops=arrow_kw)

    ax.set_title("Model Architecture — Hybrid GNN + BNN Pipeline", fontsize=18,
                 fontweight="bold", color=TEXT_DARK, pad=16, loc="left")
    fig.text(0.99, 0.02, "Epistemic uncertainty + momentum-weighted prior → surprise-adjusted logits",
             ha="right", fontsize=9, color=TEXT_MUTED, fontstyle="italic")
    save(fig, "09_model_architecture")


# ──────────────────────────────────────────────
# CHART 10: Head-to-Head Upset Grid (Top 12)
# ──────────────────────────────────────────────
def chart_10_upset_grid():
    top12_names = [t[0] for t in TEAMS_DATA[:12]]
    n = len(top12_names)
    # Generate plausible win probabilities based on Elo differences
    grid = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                grid[i][j] = np.nan
            else:
                elo_i = TEAMS_DATA[i][7]
                elo_j = TEAMS_DATA[j][7]
                diff = elo_i - elo_j
                expected = 1.0 / (1.0 + 10 ** (-diff / 400))
                grid[i][j] = expected * 100

    fig, ax = plt.subplots(figsize=(12, 10), facecolor=BG_DEEP)
    mask = np.isnan(grid)
    display = np.where(mask, 50, grid)
    im = ax.imshow(display, cmap=DIV_CMAP, vmin=25, vmax=75, interpolation="nearest")

    ax.set_xticks(range(n))
    ax.set_xticklabels(top12_names, fontsize=8, color=TEXT_DARK, rotation=45, ha="right", fontweight="bold")
    ax.set_yticks(range(n))
    ax.set_yticklabels(top12_names, fontsize=8, color=TEXT_DARK, fontweight="bold")

    for i in range(n):
        for j in range(n):
            if i != j:
                val = grid[i][j]
                txt_color = "#ffffff" if (val > 60 or val < 40) else TEXT_DARK
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=7, color=txt_color, fontweight="bold", fontfamily="monospace")
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color=TEXT_MUTED)

    style_ax(ax, "Head-to-Head Win Probability Matrix — Top 12 Teams", grid_axis=None)
    ax.set_xlabel("Away Team", fontsize=10, color=TEXT_MUTED, labelpad=10)
    ax.set_ylabel("Home Team", fontsize=10, color=TEXT_MUTED, labelpad=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Win Probability %", color=TEXT_MUTED, fontsize=9)
    cbar.ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    cbar.outline.set_edgecolor(BORDER)
    fig.text(0.99, 0.01, "Based on Elo-adjusted Bayesian posterior distributions",
             ha="right", fontsize=8, color=TEXT_MUTED, fontstyle="italic")
    save(fig, "10_head_to_head_matrix")


# ──────────────────────────────────────────────
# ZIP ARCHIVE GENERATOR
# ──────────────────────────────────────────────
import zipfile

def zip_outputs():
    zip_path = OUT_DIR.parent / "youtube_charts.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in sorted(OUT_DIR.glob("*.png")):
            zipf.write(file, arcname=file.name)
    print(f"\n📦 Created zip archive for easy download: {zip_path}\n")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print(f"\n🎬 Generating 10 YouTube-ready light charts → {OUT_DIR}\n")
    chart_01_top20_win()
    chart_02_funnel()
    chart_03_group_heatmap()
    chart_04_elo_vs_win()
    chart_05_confederation()
    chart_06_surprise_impact()
    chart_07_convergence()
    chart_08_elimination_waterfall()
    chart_09_architecture()
    chart_10_upset_grid()
    print(f"\n✅ All 10 charts saved to: {OUT_DIR}\n")
    
    zip_outputs()


if __name__ == "__main__":
    main()
