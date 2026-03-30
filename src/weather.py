"""Weather impact module for MLB score predictions.

Fetches live weather data for MLB venues and computes run-scoring
adjustments based on atmospheric conditions.

Key effects modeled:
- Temperature: Ball carries ~4.5 ft farther per 10°F above baseline.
  Every ~5 ft of carry ≈ 1-2% more HR. Net effect: ~1.5-2% more runs
  per 10°F above 72°F.
- Wind speed & direction: Outbound wind at 10mph ≈ +5% HR rate.
  Inbound wind at 10mph ≈ -5% HR rate. Crosswinds have ~half effect.
- Humidity: Higher humidity = lower air density = ball carries slightly
  farther. ~0.5% more distance per 10% humidity above 50%.
- Precipitation: Active rain/drizzle suppresses offense (grip issues,
  visibility). Heavy rain = game likely delayed.
- Roof status: Domed/retractable-closed stadiums use neutral weather.

Data source: Open-Meteo API (free, no key required).

References:
- Alan Nathan, "Effect of Weather on Baseball" (2012)
- Baseball Prospectus atmospheric studies
"""

import requests
from typing import Any

# =============================================================================
# VENUE COORDINATES AND METADATA
# =============================================================================
# venue_id -> {name, lat, lon, roof_type, orientation_deg}
# orientation_deg: degrees from home plate to center field (0=N, 90=E, 180=S, 270=W)
# Used to determine if wind is blowing "out" or "in"

VENUE_DATA = {
    # AL East
    3: {"name": "Fenway Park", "lat": 42.3467, "lon": -71.0972, "roof": "open", "cf_bearing": 200},
    2: {"name": "Camden Yards", "lat": 39.2838, "lon": -76.6216, "roof": "open", "cf_bearing": 180},
    14: {"name": "Rogers Centre", "lat": 43.6414, "lon": -79.3894, "roof": "retractable", "cf_bearing": 315},
    3313: {"name": "Yankee Stadium", "lat": 40.8296, "lon": -73.9262, "roof": "open", "cf_bearing": 87},
    12: {"name": "Tropicana Field", "lat": 27.7682, "lon": -82.6534, "roof": "dome", "cf_bearing": 180},
    # AL Central
    4: {"name": "Guaranteed Rate Field", "lat": 41.8299, "lon": -87.6338, "roof": "open", "cf_bearing": 200},
    5: {"name": "Progressive Field", "lat": 41.4962, "lon": -81.6852, "roof": "open", "cf_bearing": 175},
    2394: {"name": "Comerica Park", "lat": 42.3390, "lon": -83.0485, "roof": "open", "cf_bearing": 165},
    7: {"name": "Kauffman Stadium", "lat": 39.0517, "lon": -94.4803, "roof": "open", "cf_bearing": 180},
    3312: {"name": "Target Field", "lat": 44.9817, "lon": -93.2776, "roof": "open", "cf_bearing": 195},
    # AL West
    1: {"name": "Angel Stadium", "lat": 33.8003, "lon": -117.8827, "roof": "open", "cf_bearing": 165},
    2392: {"name": "Minute Maid Park", "lat": 29.7573, "lon": -95.3555, "roof": "retractable", "cf_bearing": 347},
    10: {"name": "Oakland Coliseum", "lat": 37.7516, "lon": -122.2005, "roof": "open", "cf_bearing": 210},
    680: {"name": "T-Mobile Park", "lat": 47.5914, "lon": -122.3325, "roof": "retractable", "cf_bearing": 185},
    5325: {"name": "Globe Life Field", "lat": 32.7473, "lon": -97.0845, "roof": "retractable", "cf_bearing": 180},
    # NL East
    4705: {"name": "Truist Park", "lat": 33.8907, "lon": -84.4677, "roof": "open", "cf_bearing": 185},
    4169: {"name": "loanDepot Park", "lat": 25.7781, "lon": -80.2196, "roof": "retractable", "cf_bearing": 10},
    3289: {"name": "Citi Field", "lat": 40.7571, "lon": -73.8458, "roof": "open", "cf_bearing": 135},
    2681: {"name": "Citizens Bank Park", "lat": 39.9061, "lon": -75.1665, "roof": "open", "cf_bearing": 195},
    3309: {"name": "Nationals Park", "lat": 38.8730, "lon": -77.0074, "roof": "open", "cf_bearing": 175},
    # NL Central
    17: {"name": "Wrigley Field", "lat": 41.9484, "lon": -87.6553, "roof": "open", "cf_bearing": 190},
    2602: {"name": "Great American Ball Park", "lat": 39.0975, "lon": -84.5069, "roof": "open", "cf_bearing": 200},
    32: {"name": "American Family Field", "lat": 43.0280, "lon": -87.9712, "roof": "retractable", "cf_bearing": 180},
    31: {"name": "PNC Park", "lat": 40.4469, "lon": -80.0058, "roof": "open", "cf_bearing": 45},
    2889: {"name": "Busch Stadium", "lat": 38.6226, "lon": -90.1928, "roof": "open", "cf_bearing": 180},
    # NL West
    15: {"name": "Chase Field", "lat": 33.4455, "lon": -112.0667, "roof": "retractable", "cf_bearing": 180},
    19: {"name": "Coors Field", "lat": 39.7559, "lon": -104.9942, "roof": "open", "cf_bearing": 195},
    22: {"name": "Dodger Stadium", "lat": 34.0739, "lon": -118.2400, "roof": "open", "cf_bearing": 175},
    2680: {"name": "Petco Park", "lat": 32.7076, "lon": -117.1570, "roof": "open", "cf_bearing": 185},
    2395: {"name": "Oracle Park", "lat": 37.7786, "lon": -122.3893, "roof": "open", "cf_bearing": 195},
}

# Baseline conditions (all adjustments relative to this)
BASELINE_TEMP_F = 72.0  # Standard baseball temperature
BASELINE_HUMIDITY = 50.0  # 50% relative humidity
BASELINE_WIND_MPH = 5.0  # Light breeze

# Adjustment coefficients (derived from Nathan 2012 + BP research)
TEMP_RUN_FACTOR_PER_10F = 0.018  # +1.8% runs per 10°F above baseline
WIND_OUT_FACTOR_PER_10MPH = 0.045  # +4.5% runs per 10mph wind blowing out
WIND_IN_FACTOR_PER_10MPH = -0.040  # -4.0% runs per 10mph wind blowing in
HUMIDITY_FACTOR_PER_10PCT = 0.004  # +0.4% runs per 10% humidity above baseline
RAIN_SUPPRESSION_FACTOR = 0.92  # -8% runs during active light rain


def fetch_weather(lat: float, lon: float, game_time_utc: str = "") -> dict[str, Any] | None:
    """Fetch current weather from Open-Meteo API.

    Returns: {temp_f, wind_mph, wind_direction_deg, humidity, precip_mm,
              weather_code, weather_desc}
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,"
                       "wind_speed_10m,wind_direction_10m",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "mm",
            "timezone": "America/New_York",
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        current = data.get("current", {})

        # Weather code descriptions (WMO standard)
        weather_codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Depositing rime fog",
            51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            66: "Light freezing rain", 67: "Heavy freezing rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers",
            82: "Violent rain showers", 85: "Slight snow showers", 86: "Heavy snow showers",
            95: "Thunderstorm", 96: "Thunderstorm w/ slight hail",
            99: "Thunderstorm w/ heavy hail",
        }

        code = current.get("weather_code", 0)
        return {
            "temp_f": current.get("temperature_2m", BASELINE_TEMP_F),
            "wind_mph": current.get("wind_speed_10m", 0),
            "wind_direction_deg": current.get("wind_direction_10m", 0),
            "humidity": current.get("relative_humidity_2m", BASELINE_HUMIDITY),
            "precip_mm": current.get("precipitation", 0),
            "weather_code": code,
            "weather_desc": weather_codes.get(code, f"Code {code}"),
        }
    except Exception as e:
        return None


def compute_wind_component(
    wind_speed: float, wind_dir: float, cf_bearing: float
) -> float:
    """Compute the wind component blowing toward center field (outbound).

    Positive = blowing out (favors hitters)
    Negative = blowing in (favors pitchers)
    """
    import math
    # Angle between wind direction (where wind comes FROM) and CF bearing
    # Wind blowing OUT means wind is coming from behind home plate toward CF
    # Wind direction is where wind comes FROM, so wind blowing out means
    # wind_dir ≈ cf_bearing + 180 (wind comes from behind HP)
    angle_diff = math.radians((wind_dir - (cf_bearing + 180)) % 360)
    # cos component: 1.0 = perfectly blowing out, -1.0 = blowing in
    return wind_speed * math.cos(angle_diff)


def compute_weather_factor(
    weather: dict[str, Any],
    venue_id: int | None,
) -> tuple[float, dict[str, float]]:
    """Compute overall weather adjustment factor for run scoring.

    Returns (total_factor, breakdown) where breakdown shows each component.
    Factor of 1.0 = neutral, >1.0 = more runs, <1.0 = fewer runs.
    """
    venue = VENUE_DATA.get(venue_id, {})
    roof = venue.get("roof", "open")

    # Domed stadiums: no weather effect
    if roof == "dome":
        return 1.0, {"temp": 0.0, "wind": 0.0, "humidity": 0.0, "precip": 0.0,
                      "total": 1.0, "roof": "dome"}

    # Retractable roof: check if conditions would close it
    # Typically closed for rain, extreme cold (<50°F), or extreme heat (>95°F)
    if roof == "retractable":
        precip = weather.get("precip_mm", 0)
        temp = weather.get("temp_f", BASELINE_TEMP_F)
        code = weather.get("weather_code", 0)
        if precip > 0.1 or code >= 51 or temp < 50 or temp > 95:
            return 1.0, {"temp": 0.0, "wind": 0.0, "humidity": 0.0, "precip": 0.0,
                          "total": 1.0, "roof": "retractable-closed"}

    # Temperature adjustment
    temp_diff = weather.get("temp_f", BASELINE_TEMP_F) - BASELINE_TEMP_F
    temp_adj = (temp_diff / 10.0) * TEMP_RUN_FACTOR_PER_10F

    # Wind adjustment
    wind_adj = 0.0
    wind_speed = weather.get("wind_mph", 0)
    wind_dir = weather.get("wind_direction_deg", 0)
    cf_bearing = venue.get("cf_bearing", 180)

    if wind_speed > 2:  # Only meaningful above 2 mph
        outbound_component = compute_wind_component(wind_speed, wind_dir, cf_bearing)
        if outbound_component > 0:
            wind_adj = (outbound_component / 10.0) * WIND_OUT_FACTOR_PER_10MPH
        else:
            wind_adj = (abs(outbound_component) / 10.0) * WIND_IN_FACTOR_PER_10MPH

    # Humidity adjustment
    humidity_diff = weather.get("humidity", BASELINE_HUMIDITY) - BASELINE_HUMIDITY
    humidity_adj = (humidity_diff / 10.0) * HUMIDITY_FACTOR_PER_10PCT

    # Precipitation adjustment
    precip_adj = 0.0
    code = weather.get("weather_code", 0)
    if code in (51, 53, 61, 80):  # Light drizzle/rain
        precip_adj = RAIN_SUPPRESSION_FACTOR - 1.0  # -8%
    elif code in (55, 63, 81, 95):  # Moderate rain/storm
        precip_adj = 0.88 - 1.0  # -12%

    # Total factor
    total = 1.0 + temp_adj + wind_adj + humidity_adj + precip_adj

    # Clamp to reasonable range
    total = max(0.80, min(total, 1.25))

    return total, {
        "temp": round(temp_adj, 4),
        "wind": round(wind_adj, 4),
        "humidity": round(humidity_adj, 4),
        "precip": round(precip_adj, 4),
        "total": round(total, 4),
        "roof": roof,
    }


def fetch_game_weather(venue_id: int | None) -> tuple[dict | None, float, dict]:
    """Fetch weather for a venue and compute the adjustment factor.

    Returns (weather_data, factor, breakdown).
    """
    if venue_id is None or venue_id not in VENUE_DATA:
        return None, 1.0, {"total": 1.0, "roof": "unknown"}

    venue = VENUE_DATA[venue_id]
    weather = fetch_weather(venue["lat"], venue["lon"])

    if weather is None:
        return None, 1.0, {"total": 1.0, "roof": venue.get("roof", "open")}

    factor, breakdown = compute_weather_factor(weather, venue_id)
    return weather, factor, breakdown
