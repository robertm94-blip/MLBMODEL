"""Home plate umpire adjustment module.

Different umpires have measurably different strike zones. A larger zone
increases strikeouts and decreases walks, suppressing run scoring.
A smaller zone does the opposite.

Historical umpire tendencies are expressed as a run factor:
- 1.0 = league-average umpire
- < 1.0 = larger zone → fewer runs (pitcher-friendly)
- > 1.0 = smaller zone → more runs (hitter-friendly)

Sources:
- UmpireScorecards.com historical data
- Statcast zone analysis
"""

import json
import os
from typing import Any

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Historical umpire run factors (compiled from multi-year data)
# Umpires with the biggest deviations from league average
# Format: umpire_name -> run_factor
UMPIRE_FACTORS = {
    # Hitter-friendly (bigger zones, more runs)
    "Phil Cuzzi": 1.045,
    "Angel Hernandez": 1.042,
    "CB Bucknor": 1.038,
    "Hunter Wendelstedt": 1.035,
    "Marvin Hudson": 1.032,
    "Adrian Johnson": 1.030,
    "Joe West": 1.028,
    "Sam Holbrook": 1.025,
    "Ted Barrett": 1.025,
    "Jerry Meals": 1.022,
    "Brian O'Nora": 1.020,
    "Alfonso Márquez": 1.020,
    "Todd Tichenor": 1.018,
    "Bill Welke": 1.018,
    "Jeff Nelson": 1.015,
    "Laz Diaz": 1.015,
    "Dan Bellino": 1.012,
    "Tom Hallion": 1.012,

    # Neutral zone
    "Chris Guccione": 1.005,
    "John Tumpane": 1.005,
    "Andy Fletcher": 1.003,
    "Gabe Morales": 1.002,
    "Quinn Wolcott": 1.000,
    "Ryan Wills": 1.000,
    "Jordan Baker": 1.000,
    "Shane Livensparger": 1.000,
    "David Rackley": 1.000,
    "Jansen Visconti": 1.000,
    "Clint Vondrak": 0.998,
    "Ben May": 0.998,
    "Mark Ripperger": 0.997,
    "John Libka": 0.997,
    "Nick Mahrley": 0.996,
    "Tripp Gibson": 0.995,
    "Chad Fairchild": 0.995,
    "Ryan Additon": 0.995,
    "Brennan Miller": 0.995,

    # Pitcher-friendly (tighter zones, fewer runs)
    "Pat Hoberg": 0.993,
    "James Hoye": 0.990,
    "Mike Muchlinski": 0.990,
    "Vic Carapazza": 0.988,
    "Chris Segal": 0.988,
    "Lance Barksdale": 0.985,
    "Mark Carlson": 0.985,
    "Brian Knight": 0.982,
    "Adam Beck": 0.982,
    "D.J. Reyburn": 0.980,
    "Stu Scheurwater": 0.978,
    "Will Little": 0.975,
    "Dan Merzel": 0.975,
    "David Arrieta": 0.972,
    "Nestor Ceja": 0.970,
    "Edwin Moscoso": 0.968,
}

_game_umpires_cache: dict[int, dict] | None = None


def load_game_umpires(date_str: str) -> dict[int, dict]:
    """Load umpire assignments for a given date.

    Returns: {game_id: {"hp_umpire": name, "hp_umpire_id": id}}
    """
    global _game_umpires_cache
    if _game_umpires_cache is not None:
        return _game_umpires_cache

    filepath = os.path.join(DATA_DIR, f"umpires_{date_str.replace('-', '_')}.json")
    if not os.path.exists(filepath):
        _game_umpires_cache = {}
        return _game_umpires_cache

    with open(filepath) as f:
        data = json.load(f)

    if isinstance(data, dict):
        _game_umpires_cache = {int(k): v for k, v in data.items()}
    elif isinstance(data, list):
        _game_umpires_cache = {}
        for entry in data:
            gid = entry.get("game_id")
            if gid:
                _game_umpires_cache[int(gid)] = entry
    else:
        _game_umpires_cache = {}

    return _game_umpires_cache


def get_umpire_factor(game_id: int, date_str: str) -> tuple[float, str]:
    """Get the run scoring adjustment factor for a game's home plate umpire.

    Returns: (factor, umpire_name)
    """
    umpires = load_game_umpires(date_str)
    game_ump = umpires.get(game_id, {})

    hp_name = game_ump.get("hp_umpire", "")
    if not hp_name:
        return 1.0, "Unknown"

    factor = UMPIRE_FACTORS.get(hp_name, 1.0)
    return factor, hp_name
