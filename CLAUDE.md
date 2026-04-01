# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLB prediction model that outputs win probabilities, run lines, and totals for daily MLB games. Two prediction engines run in parallel:

1. **Negative Binomial analytical model** (`src/model.py`) — fast, used in `predict_today.py` and `predict_full.py`
2. **Monte Carlo simulator** (`src/simulator.py`, `src/live_sim.py`) — 10,000 PA-by-PA game simulations, used for raw win probability and live in-game WP

## Commands

```bash
# Install dependencies
pip install numpy scipy requests

# Daily predictions (10-factor NB model with weather)
python predict_full.py --date 2026-03-31

# Raw Monte Carlo projections (10K sims per game, no compression)
# Built inline in scripts — see bottom of predict_full.py usage pattern

# Full box score simulation
python simulate_boxscores.py --date 2026-03-31 --sims 1000
python simulate_boxscores.py --game 824135  # single game

# Live in-game win probability (polls MLB API every second)
python live.py                    # all live games dashboard
python live.py --game 824135      # single game detailed view
python live.py --sims 5000        # more sims per update

# Backtesting
python backtest_2025.py                              # single season
python backtest_multiyear.py --seasons 2023,2024,2025  # multi-year

# Older/simpler model (Poisson-era, no 10-factor)
python predict_today.py --date 2026-03-31
```

## Architecture

### Two Prediction Engines

**Analytical (NB):** `predict_full.py` computes expected runs (lambda) per team by multiplying 10 factors, then uses `src/model.py` (Negative Binomial distribution, r=12) to generate score probability matrices and win probabilities. Fast (~2 seconds for 15 games).

**Monte Carlo:** `src/simulator.py` simulates each game plate-appearance by plate-appearance. Each PA draws from batter-specific outcome probabilities adjusted for the opposing pitcher. Tracks baserunners, batting order, and starter-to-bullpen transitions. Slower (~45 seconds per game at 10K sims) but produces uncompressed win probabilities and full player box scores.

**Live WP:** `src/live_sim.py` + `live.py` — same MC engine but starts from mid-game state (score, inning, outs, runners) and simulates forward. Polls MLB Stats API feed every second.

### 10 Factors (applied in `predict_full.py` and `src/simulator.py`)

| # | Factor | Module | Key Constant |
|---|--------|--------|-------------|
| 1 | Team RS/RA | `src/projections.py` | FanGraphs Depth Charts standings |
| 2 | Lineup offense | `src/lineup_offense.py` | 60% lineup / 40% team blend |
| 3 | L/R platoon | `src/platoon.py` | +3.5-4% platoon advantage |
| 4 | Starter quality | `src/projections.py` | ERA 40% + FIP 60% blend |
| 5 | Bullpen quality | `src/bullpen.py` | IP-weighted reliever ERA/FIP |
| 6 | SP/pen split | `src/bullpen.py` | Dynamic by starter IP/GS |
| 7 | Umpire zone | `src/umpire.py` | Historical run factors |
| 8 | Rest/travel | `src/rest_travel.py` | Schedule API lookback |
| 9 | Weather | `src/weather.py` | Open-Meteo API live |
| 10 | Park + home | `src/features.py` | `HOME_ADVANTAGE = 0.25` |

### Data Flow

```
FanGraphs API → data/projections/{year}/  (Steamer/ZiPS/THE BAT pitcher+batter JSON)
MLB Stats API → schedules, lineups, umpires, pitcher/batter handedness, live game feed
Open-Meteo API → live weather per venue
```

`src/projections.py` loads and blends pitcher projections from all 3 systems with equal weight. `src/lineup_offense.py` does the same for batters, keyed by MLBAM ID.

### Key Tunable Parameters

- `src/model.py`: `NB_SHAPE_R = 12.0` — NB overdispersion. Lower = more upset probability, compresses favorites toward 50%. Higher = closer to Poisson (sharper favorites).
- `src/features.py`: Pitching blend is `starter_factor * 0.65 + team_def * 0.35` in `compute_expected_runs`.
- `src/simulator.py`: Pitcher hit suppression dampened to 40% of raw ERA/FIP ratio (`hit_mult = 1.0 + (raw - 1.0) * 0.40`). K% and BB% use 60/40 batter/pitcher weighting.
- `src/totals_model.py`: `REGRESSION_TO_MEAN = 0.55` default, drops to 0.30-0.40 when multiple factors agree (dynamic regression). Uses additive adjustments, not multiplicative.
- `src/features.py`: `HOME_ADVANTAGE = 0.25` runs.

### Totals Model

`src/totals_model.py` is a **separate model** from the ML engine. Uses additive factors instead of multiplicative to prevent compounding bias. Only bet totals when model disagrees with posted line by 0.5+ runs.

### Market Edge Detection

`src/edge.py` — converts model win% to edge vs posted American odds, computes Kelly Criterion bet sizing. The 52.38% implied probability at -110 is the break-even threshold (`IMPLIED = 110/210`).

### Backtesting

`src/backtest.py` provides log-loss, Brier score, calibration curves, and flat-bet ROI simulations at various edge thresholds. Historical results stored in `data/results_{year}.json`.

**Important caveat:** FanGraphs API returns current projections regardless of season parameter, so 2023/2024 pitcher projections in the backtest are identical to 2026. Team-level stats from MLB API are correctly per-season.

### Lineup Data

Lineup files (`lineups_YYYY_MM_DD.json`) contain 9 batters per team with player ID, name, position, and batting order. Source is either `official_lineup` (from MLB API boxscore) or `depth_chart_projection` (from team depth chart endpoint). The model uses official lineups when available, falls back to depth charts.
