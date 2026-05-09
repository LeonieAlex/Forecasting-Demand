"""
fuel.py
-------
Fuel consumption model implementing the ICAO Annex 6 fuel planning framework
using OpenAP (Sun et al., 2020) for all airborne fuel flow computations.

CITATION
--------
Sun, J., Hoekstra, J.M. & Ellerbroek, J. (2020).
OpenAP: An Open-Source Aircraft Performance Model for Air Transportation
Studies and Simulations. Aerospace, 7(8), 104.
https://doi.org/10.3390/aerospace7080104

Engine idle fuel flow (ff_idl) is sourced from the ICAO Aircraft Engine
Emissions Databank, accessed via OpenAP's engine property module.

ICAO ANNEX 6 FUEL STRUCTURE
-----------------------------
Total fuel uplifted (m_total) is the sum of five components (ICAO, 2022):

    m_total = m_trip + m_cont + m_altn + m_final_res + m_taxi     (Eq. 3.7)

    m_trip        -- fuel to fly the planned route (climb+cruise+descent)
    m_cont        -- contingency: 5% of trip fuel (ICAO statistical minimum)
    m_altn        -- fuel to divert to designated alternate airport
    m_final_res   -- 30 min holding at 1,500 ft above alternate aerodrome
    m_taxi        -- ground idle fuel: ff_idl × n_engines × t_taxi

Each airborne component (trip, alternate, final reserve) is computed using
OpenAP's fuelflow.enroute(mass, tas, alt, vs) which returns kg/s.
Taxi fuel uses the ICAO engine idle fuel flow (ff_idl) from the ICAO
Aircraft Engine Emissions Databank, giving a parameterised, aircraft-specific
value rather than a fixed assumption.

AIRCRAFT → OPENAP MAPPING
--------------------------
Config type  →  OpenAP code  →  Aircraft
B777         →  B77W         →  Boeing 777-300ER (GE90-115B)
B787         →  B788         →  Boeing 787-8     (Trent 1000-A2)
B789         →  B789         →  Boeing 787-9     (Trent 1000-K2)
A350         →  A359         →  Airbus A350-900  (Trent XWB-84)
A350ULR      →  A359         →  A350-900ULR modelled as A359 base type
A380         →  A388         →  Airbus A380-800  (GP7270)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

try:
    import openap
    OPENAP_AVAILABLE = True
except ImportError:
    OPENAP_AVAILABLE = False
    raise RuntimeError("openap not installed. Run: pip install openap")


# ── Aircraft configuration ─────────────────────────────────────────────────────

# Maps simulator config code → (OpenAP ICAO designator, default engine name)
AIRCRAFT_MAP: dict[str, tuple[str, str]] = {
    "B777":    ("B77W", "GE90-115B"),
    "B787":    ("B788", "Trent 1000-A2"),
    "B789":    ("B789", "Trent 1000-K2"),
    "A350":    ("A359", "Trent XWB-84"),
    "A350ULR": ("A359", "Trent XWB-84"),   # ULR modelled as A359 base
    "A380":    ("A388", "GP7270"),
}

# Cruise conditions per OpenAP code (TAS in knots, altitude in feet)
# Calibrated to published cruise performance for each type
CRUISE_PARAMS: dict[str, dict] = {
    "B77W": {"tas_kts": 490, "alt_ft": 35_000},   # Mach ~0.84
    "B788": {"tas_kts": 488, "alt_ft": 35_000},
    "B789": {"tas_kts": 488, "alt_ft": 35_000},
    "A359": {"tas_kts": 488, "alt_ft": 37_000},   # A350 typically operates higher
    "A388": {"tas_kts": 486, "alt_ft": 35_000},
}

# Directional wind correction factors applied to cruise duration
# Accounts for climatological jet stream bias on transpacific routes
# Calibrated from ICAO Circular 313 schedule buffer differentials
# (origin_region, dest_region) → cruise time multiplier
WIND_CORRECTION: dict[tuple[str, str], float] = {
    ("asia", "us"):   1.08,   # westbound: ~8% longer, prevailing headwind
    ("us",   "asia"): 0.94,   # eastbound: ~6% shorter, tailwind
    ("au",   "us"):   1.04,   # moderate headwind
    ("us",   "au"):   0.97,   # partial tailwind
    ("me",   "us"):   1.03,   # Middle East → US
    ("us",   "me"):   0.98,
}

# Country → region mapping for wind correction lookup
COUNTRY_REGION: dict[str, str] = {
    "JP": "asia", "CN": "asia", "KR": "asia",
    "HK": "asia", "SG": "asia", "TW": "asia",
    "AU": "au",
    "US": "us", "CA": "us",
    "GB": "us",  # European connecting pax routing as US-bound
    "DE": "us",
    "NL": "us",
    "AE": "me",
}

# Standard flight phase parameters
CLIMB_TAS_KTS     = 350
CLIMB_ALT_FT      = 20_000
CLIMB_VS_FPM      = 1_500
CLIMB_DURATION_H  = 35 / 60      # 35 minutes

DESCENT_TAS_KTS     = 380
DESCENT_ALT_FT      = 15_000
DESCENT_VS_FPM      = -1_500
DESCENT_DURATION_H  = 25 / 60    # 25 minutes

# Alternate airport parameters (200 nm diversion, representative transpacific)
ALTN_DISTANCE_KM  = 370          # 200 nautical miles
ALTN_TAS_KTS      = 420
ALTN_ALT_FT       = 20_000

# Final reserve: 30 min holding at 1,500 ft above alternate
HOLD_TAS_KTS      = 210
HOLD_ALT_FT       = 1_500
HOLD_DURATION_H   = 30 / 60      # 30 minutes

# Taxi parameters
TAXI_TIME_MIN     = 18            # minutes; typical major transpacific hub taxi-out
ICAO_RESERVE_FACTOR = 1.15        # kept for reference; now replaced by explicit ICAO components

JET_A_DENSITY_KG_PER_L = 0.800   # ASTM D1655 standard conditions


# ── Parameters ─────────────────────────────────────────────────────────────────

@dataclass
class FuelConsumptionParams:
    cruise_step_min: float      = 30      # mass iteration step in cruise (minutes)
    cruise_dist_fraction: float = 0.92   # fraction of GCD covered in cruise phase
    taxi_time_min: float        = TAXI_TIME_MIN
    altn_distance_km: float     = ALTN_DISTANCE_KM
    jet_a_density_kg_per_l: float = JET_A_DENSITY_KG_PER_L
    apply_wind_correction: bool = True   # apply directional jet stream correction


# ── Object cache ────────────────────────────────────────────────────────────────

_ff_cache:    dict[str, object] = {}
_props_cache: dict[str, dict]   = {}
_eng_cache:   dict[str, dict]   = {}

def _ff(code: str):
    if code not in _ff_cache:
        _ff_cache[code] = openap.FuelFlow(code)
    return _ff_cache[code]

def _props(code: str) -> dict:
    if code not in _props_cache:
        _props_cache[code] = openap.prop.aircraft(code)
    return _props_cache[code]

def _eng(eng_name: str) -> dict:
    if eng_name not in _eng_cache:
        _eng_cache[eng_name] = openap.prop.engine(eng_name)
    return _eng_cache[eng_name]


# ── Core function ───────────────────────────────────────────────────────────────

def compute_fuel_consumption(
    aircraft_type: str,
    distance_km: float,
    pax_payload_kg: float,
    params: FuelConsumptionParams | None = None,
    market_country: str | None = None,
    dest_region: str = "us",
) -> dict[str, float]:
    """
    Compute total fuel uplifted per flight using the ICAO Annex 6 framework.

    Total fuel = trip + contingency + alternate + final reserve + taxi
    All airborne components computed via OpenAP fuelflow.enroute().
    Taxi fuel computed from ICAO engine idle flow (ff_idl × n_engines × t_taxi).

    Parameters
    ----------
    aircraft_type   : simulator config code e.g. "B777", "B789"
    distance_km     : great-circle route distance (km)
    pax_payload_kg  : passenger payload = seats_sold × 100 kg
    params          : FuelConsumptionParams
    market_country  : ISO-2 origin market (for wind correction)
    dest_region     : destination region code (for wind correction)

    Returns
    -------
    dict with all ICAO Annex 6 components and efficiency metrics
    """
    if params is None:
        params = FuelConsumptionParams()

    openap_code, eng_name = AIRCRAFT_MAP.get(aircraft_type, ("B789", "Trent 1000-K2"))
    ff_model  = _ff(openap_code)
    props     = _props(openap_code)
    eng_props = _eng(eng_name)
    cruise_p  = CRUISE_PARAMS.get(openap_code, {"tas_kts": 488, "alt_ft": 35_000})

    oew     = props["oew"]
    n_eng   = props["engine"]["number"]
    ff_idl  = eng_props["ff_idl"]       # kg/s per engine, from ICAO databank

    # ── Seed TOW for phase calculations ───────────────────────────────────────
    # Initialise with rough fuel estimate; phases update mass sequentially
    fuel_seed  = distance_km * 9.5
    tow        = min(oew + pax_payload_kg + fuel_seed, props["mtow"])
    lnd_mass   = oew + pax_payload_kg * 0.90   # approximate landing mass

    # ── Eq. 3.8: Trip fuel — three phases ─────────────────────────────────────

    # Phase 1: Climb
    ff_climb   = ff_model.enroute(mass=tow, tas=CLIMB_TAS_KTS,
                                  alt=CLIMB_ALT_FT, vs=CLIMB_VS_FPM)
    m_climb    = ff_climb * CLIMB_DURATION_H * 3600

    # Phase 2: Cruise — step-wise mass iteration
    # Apply directional wind correction to cruise time
    wind_mult = 1.0
    if params.apply_wind_correction and market_country:
        origin_region = COUNTRY_REGION.get(market_country, "asia")
        wind_mult = WIND_CORRECTION.get((origin_region, dest_region), 1.0)

    cruise_dist_km  = distance_km * params.cruise_dist_fraction
    cruise_speed_kmh = cruise_p["tas_kts"] * 1.852
    cruise_total_hr  = (cruise_dist_km / cruise_speed_kmh) * wind_mult
    step_hr          = params.cruise_step_min / 60

    current_mass = tow - m_climb
    m_cruise     = 0.0
    elapsed_hr   = 0.0

    while elapsed_hr < cruise_total_hr:
        dt          = min(step_hr, cruise_total_hr - elapsed_hr)
        ff_step     = ff_model.enroute(mass=current_mass,
                                       tas=cruise_p["tas_kts"],
                                       alt=cruise_p["alt_ft"], vs=0)
        step_fuel   = ff_step * dt * 3600
        m_cruise    += step_fuel
        current_mass -= step_fuel
        elapsed_hr  += dt

    # Phase 3: Descent
    ff_desc   = ff_model.enroute(mass=current_mass, tas=DESCENT_TAS_KTS,
                                  alt=DESCENT_ALT_FT, vs=DESCENT_VS_FPM)
    m_descent = ff_desc * DESCENT_DURATION_H * 3600

    m_trip = m_climb + m_cruise + m_descent

    # ── Eq. 3.9: Contingency fuel — 5% of trip fuel ───────────────────────────
    m_cont = m_trip * 0.05

    # ── Alternate fuel — diversion to alternate airport ───────────────────────
    altn_speed_kmh = ALTN_TAS_KTS * 1.852
    altn_hr        = params.altn_distance_km / altn_speed_kmh
    ff_altn        = ff_model.enroute(mass=lnd_mass, tas=ALTN_TAS_KTS,
                                      alt=ALTN_ALT_FT, vs=0)
    m_altn         = ff_altn * altn_hr * 3600

    # ── Final reserve — 30 min holding at 1,500 ft ───────────────────────────
    ff_hold       = ff_model.enroute(mass=lnd_mass, tas=HOLD_TAS_KTS,
                                     alt=HOLD_ALT_FT, vs=0)
    m_final_res   = ff_hold * HOLD_DURATION_H * 3600

    # ── Eq. 3.10: Taxi fuel — ICAO idle flow × n_engines × taxi time ─────────
    m_taxi = ff_idl * n_eng * params.taxi_time_min * 60

    # ── Eq. 3.7: Total fuel uplifted ──────────────────────────────────────────
    m_total = m_trip + m_cont + m_altn + m_final_res + m_taxi

    return {
        # ICAO Annex 6 components (Eq. 3.7)
        "trip_fuel_kg":       round(m_trip, 1),
        "climb_fuel_kg":      round(m_climb, 1),
        "cruise_fuel_kg":     round(m_cruise, 1),
        "descent_fuel_kg":    round(m_descent, 1),
        "contingency_fuel_kg":round(m_cont, 1),
        "alternate_fuel_kg":  round(m_altn, 1),
        "final_reserve_kg":   round(m_final_res, 1),
        "taxi_fuel_kg":       round(m_taxi, 1),
        "total_fuel_uplifted_kg": round(m_total, 1),
        # Conversions
        "fuel_tonnes":        round(m_total / 1_000, 3),
        "fuel_litres":        round(m_total / params.jet_a_density_kg_per_l, 1),
        # Metadata
        "openap_aircraft_code": openap_code,
        "engine_name":          eng_name,
        "wind_correction":      round(wind_mult, 3),
    }


def fuel_efficiency_metrics(
    fuel_data: dict,
    seats_sold: int,
    seats_capacity: int,
    distance_km: float,
) -> dict[str, float]:
    """
    Fuel efficiency KPIs per IATA (2024) sustainability reporting conventions.

    fuel/seat (kg)   — trip fuel per seat sold
    fuel/ASK (kg)    — total uplifted per available seat-km
    fuel/RPK (kg)    — total uplifted per revenue passenger-km
    """
    trip_fuel  = fuel_data["trip_fuel_kg"]
    total_fuel = fuel_data["total_fuel_uplifted_kg"]
    ask        = seats_capacity * distance_km
    rpk        = seats_sold     * distance_km

    return {
        "fuel_per_seat_kg":           round(trip_fuel / seats_sold,    2) if seats_sold > 0    else 0.0,
        "fuel_per_available_seat_kg": round(trip_fuel / seats_capacity, 2) if seats_capacity > 0 else 0.0,
        "fuel_per_ask_kg":            round(total_fuel / ask, 6)           if ask > 0           else 0.0,
        "fuel_per_rpk_kg":            round(total_fuel / rpk, 6)           if rpk > 0           else 0.0,
        "ask_km":                     ask,
        "rpk_proxy_km":               rpk,
    }
