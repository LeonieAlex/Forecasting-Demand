"""
config.py
---------
Static definitions for countries, routes, aircraft types, and hubs.
Updated to reflect current 2025/26 transpacific network.

Sources:
  - OAG Busiest Routes 2025 (Dec 2025)
  - Cirium schedule data (Oct-Dec 2025)
  - Aviation Week / Simple Flying new route announcements (Dec 2025-Apr 2026)
  - Delta A350 full transpacific conversion complete Mar 2026
  - New AU routes: AA LAX-BNE, DL LAX-MEL, UA SFO-ADL, QF DFW-SYD (Dec 2025-Jan 2026)
  - SIN-JFK / SIN-EWR (SQ A350ULR) remain world's longest nonstop routes
"""

from dataclasses import dataclass
from typing import Literal, Optional

SeasonProfile = Literal[
    "japan", "korea", "china", "singapore",
    "australia", "middle_east", "standard"
]


@dataclass
class Country:
    id: str                       # ISO-2 code
    name: str
    season_profile: SeasonProfile
    macro_vol: float              # macro shock volatility (sigma)
    demand_trend: float           # annual demand growth as decimal (e.g. 0.018 = +1.8%/yr)
    currency_sensitivity: float   # FX impact on outbound demand (0=none, 1=high)
    hubs: list[str]               # primary origin IATA airport codes


@dataclass
class Route:
    id: str
    flight_od: str                # physical flight leg label
    origin_airport: str
    dest_airport: str
    market_country: str           # ISO-2 of commercial origin market
    distance_km: int
    is_connecting: bool = False
    connecting_hub: Optional[str] = None
    base_load_factor: float = 0.83
    aircraft_type: str = "B787"


@dataclass
class Aircraft:
    type: str
    total_seats: int
    first_seats: int
    business_seats: int
    premium_eco_seats: int
    economy_seats: int
    max_payload_kg: int


# ==============================================================================
# COUNTRIES
# ==============================================================================

COUNTRIES: dict[str, Country] = {

    # --- Asia-Pacific origin markets ------------------------------------------

    "JP": Country(
        id="JP", name="Japan",
        season_profile="japan",           # quad-modal: GW/Obon/cherry blossom/NYE
        macro_vol=0.08,
        demand_trend=0.012,               # 24% YoY surge summer 2025 (IATA Q1 2025)
        currency_sensitivity=0.65,        # JPY weakness strongly suppresses outbound
        hubs=["HND", "NRT", "KIX", "NGO", "FUK"],
    ),

    "CN": Country(
        id="CN", name="China",
        season_profile="china",           # tri-modal: CNY/May Golden Week/Oct National Day
        macro_vol=0.16,                   # elevated: geopolitical + capacity restrictions
        demand_trend=0.022,               # recovering but below pre-COVID trajectory
        currency_sensitivity=0.40,        # capital controls dampen FX pass-through
        hubs=["PEK", "PKX", "PVG", "CAN", "CTU", "SZX", "HGH"],
    ),

    "KR": Country(
        id="KR", name="South Korea",
        season_profile="korea",           # quad-modal: Seollal/summer/Chuseok/spring
        macro_vol=0.09,
        demand_trend=0.018,               # strong; ICN growing as Asia hub (KE-DL JV)
        currency_sensitivity=0.52,
        hubs=["ICN", "GMP"],
    ),

    "HK": Country(
        id="HK", name="Hong Kong",
        season_profile="standard",
        macro_vol=0.13,                   # political uncertainty; HKG still below 2019
        demand_trend=0.008,
        currency_sensitivity=0.44,
        hubs=["HKG"],
    ),

    "SG": Country(
        id="SG", name="Singapore",
        season_profile="singapore",       # flat with Jun/Dec school holiday peaks
        macro_vol=0.07,
        demand_trend=0.020,               # SIA capacity +44% to KR vs 2019; SIN-SFO 14/wk
        currency_sensitivity=0.33,
        hubs=["SIN"],
    ),

    "TW": Country(
        id="TW", name="Taiwan",
        season_profile="standard",
        macro_vol=0.10,                   # cross-strait geopolitical risk premium
        demand_trend=0.014,
        currency_sensitivity=0.50,
        hubs=["TPE", "KHH"],
    ),

    "AU": Country(
        id="AU", name="Australia",
        season_profile="australia",       # SOUTHERN HEMISPHERE INVERTED: Dec/Jan peak
        macro_vol=0.08,
        demand_trend=0.016,               # US-AU up 8% YoY; US now 3rd largest AU market
        currency_sensitivity=0.55,        # AUD highly volatile vs USD
        hubs=["SYD", "MEL", "BNE", "PER", "ADL"],
    ),

    # --- North America destination markets ------------------------------------

    "US": Country(
        id="US", name="United States",
        season_profile="standard",
        macro_vol=0.08,
        demand_trend=0.012,
        currency_sensitivity=0.30,        # USD reserve currency; low FX sensitivity
        hubs=["LAX", "SFO", "JFK", "ORD", "SEA", "IAH", "DFW", "ATL"],
    ),

    "CA": Country(
        id="CA", name="Canada",
        season_profile="standard",
        macro_vol=0.08,
        demand_trend=0.012,
        currency_sensitivity=0.40,
        hubs=["YVR", "YYZ", "YUL"],
    ),

    # --- Europe markets (primarily connecting via Asia hubs) ------------------

    "GB": Country(
        id="GB", name="United Kingdom",
        season_profile="standard",
        macro_vol=0.09,
        demand_trend=0.010,
        currency_sensitivity=0.45,
        hubs=["LHR", "LGW"],
    ),

    "DE": Country(
        id="DE", name="Germany",
        season_profile="standard",
        macro_vol=0.08,
        demand_trend=0.010,
        currency_sensitivity=0.40,
        hubs=["FRA", "MUC"],
    ),

    "NL": Country(
        id="NL", name="Netherlands",
        season_profile="standard",
        macro_vol=0.07,
        demand_trend=0.010,
        currency_sensitivity=0.40,
        hubs=["AMS"],
    ),

    # --- Middle East (transit hub + outbound leisure origin) ------------------

    "AE": Country(
        id="AE", name="UAE",
        season_profile="middle_east",     # heat-escape summer dominant + Eid spikes
        macro_vol=0.11,                   # geopolitical shock exposure
        demand_trend=0.025,               # Vision 2030 effect; EK/EY massive expansion
        currency_sensitivity=0.20,        # AED pegged to USD
        hubs=["DXB", "AUH"],
    ),
}


# ==============================================================================
# ROUTES
# ==============================================================================
# All distances are great-circle km (GCD).
# Aircraft types reflect dominant fleet config by operator as of 2025/26.
#
# Key structural updates vs prior version:
#   1. HND replaces NRT as primary Tokyo hub for US/EU carriers (slot-limited but premium)
#   2. Delta fully A350-900 on all transpacific from April 2026
#   3. Australia: 4 new nonstop routes added Dec 2025-Jan 2026
#   4. SIN A350ULR routes are premium-only, no economy seats
#   5. ICN growing as dominant 6th-freedom hub for CN and EU connecting pax
#   6. YVR (Vancouver) is Canada's main Pacific gateway -- more relevant than YYZ

ROUTES: list[Route] = [

    # ==========================================================================
    # JAPAN -> NORTH AMERICA
    # ==========================================================================

    # HND (Haneda) -- now the primary slot-premium hub for full-service carriers
    Route("hnd-lax",  "HND->LAX",  "HND", "LAX", "JP", 8815,  aircraft_type="B777",  base_load_factor=0.83),
    Route("hnd-jfk",  "HND->JFK",  "HND", "JFK", "JP", 10838, aircraft_type="B777",  base_load_factor=0.83),
    Route("hnd-ord",  "HND->ORD",  "HND", "ORD", "JP", 10143, aircraft_type="B787",  base_load_factor=0.83),
    Route("hnd-sfo",  "HND->SFO",  "HND", "SFO", "JP", 8280,  aircraft_type="B789",  base_load_factor=0.83),
    Route("hnd-sea",  "HND->SEA",  "HND", "SEA", "JP", 7713,  aircraft_type="B789",  base_load_factor=0.83),

    # NRT (Narita) -- still active for cargo-heavy, LCC, and secondary frequencies
    Route("nrt-lax",  "NRT->LAX",  "NRT", "LAX", "JP", 8750,  aircraft_type="B777",  base_load_factor=0.83),
    Route("nrt-sfo",  "NRT->SFO",  "NRT", "SFO", "JP", 8280,  aircraft_type="B789",  base_load_factor=0.83),

    # Osaka (KIX) and Nagoya (NGO) -- secondary JP markets
    Route("kix-lax",  "KIX->LAX",  "KIX", "LAX", "JP", 9196,  aircraft_type="B789",  base_load_factor=0.83),
    Route("ngo-lax",  "NGO->LAX",  "NGO", "LAX", "JP", 9024,  aircraft_type="B787",  base_load_factor=0.83),

    # Japan -> Canada (YVR dominant Pacific gateway)
    Route("hnd-yvr",  "HND->YVR",  "HND", "YVR", "JP", 7490,  aircraft_type="B789",  base_load_factor=0.83),
    Route("nrt-yvr",  "NRT->YVR",  "NRT", "YVR", "JP", 7565,  aircraft_type="B789",  base_load_factor=0.83),

    # ==========================================================================
    # CHINA -> NORTH AMERICA
    # ==========================================================================

    # Direct routes -- recovering but still below 2019 capacity
    Route("pvg-lax",  "PVG->LAX",  "PVG", "LAX", "CN", 9800,  aircraft_type="B777",  base_load_factor=0.83),
    Route("pek-lax",  "PEK->LAX",  "PEK", "LAX", "CN", 10101, aircraft_type="B777",  base_load_factor=0.83),
    Route("pvg-jfk",  "PVG->JFK",  "PVG", "JFK", "CN", 11385, aircraft_type="B777",  base_load_factor=0.83),
    Route("pvg-sfo",  "PVG->SFO",  "PVG", "SFO", "CN", 9273,  aircraft_type="A350",  base_load_factor=0.83),  # DL resumed 2025
    Route("pvg-sea",  "PVG->SEA",  "PVG", "SEA", "CN", 8996,  aircraft_type="A350",  base_load_factor=0.83),  # DL A350
    Route("can-lax",  "CAN->LAX",  "CAN", "LAX", "CN", 11586, aircraft_type="B787",  base_load_factor=0.83),

    # China via ICN (Korean Air / Asiana 6th freedom -- major flow of CN pax)
    Route("cn-icn-lax", "ICN->LAX (CN)", "ICN", "LAX", "CN", 9370,
          is_connecting=True, connecting_hub="ICN", aircraft_type="B777", base_load_factor=0.83),
    Route("cn-icn-sfo", "ICN->SFO (CN)", "ICN", "SFO", "CN", 9080,
          is_connecting=True, connecting_hub="ICN", aircraft_type="B789", base_load_factor=0.83),
    Route("cn-icn-yvr", "ICN->YVR (CN)", "ICN", "YVR", "CN", 8220,
          is_connecting=True, connecting_hub="ICN", aircraft_type="B789", base_load_factor=0.83),

    # ==========================================================================
    # SOUTH KOREA -> NORTH AMERICA
    # ==========================================================================

    # Korean Air + Asiana; KE-DL joint venture dominant on US routes
    Route("icn-lax",  "ICN->LAX",  "ICN", "LAX", "KR", 9370,  aircraft_type="B777",  base_load_factor=0.83),
    Route("icn-jfk",  "ICN->JFK",  "ICN", "JFK", "KR", 11095, aircraft_type="B777",  base_load_factor=0.83),
    Route("icn-sfo",  "ICN->SFO",  "ICN", "SFO", "KR", 9080,  aircraft_type="B789",  base_load_factor=0.83),
    Route("icn-sea",  "ICN->SEA",  "ICN", "SEA", "KR", 8330,  aircraft_type="B789",  base_load_factor=0.83),
    Route("icn-atl",  "ICN->ATL",  "ICN", "ATL", "KR", 11040, aircraft_type="A350",  base_load_factor=0.83),  # DL A350 hub
    Route("icn-yvr",  "ICN->YVR",  "ICN", "YVR", "KR", 8220,  aircraft_type="B789",  base_load_factor=0.83),

    # ==========================================================================
    # SINGAPORE -> NORTH AMERICA
    # ==========================================================================

    # SQ A350-900ULR: world's longest nonstop commercial flights
    Route("sin-jfk",  "SIN->JFK",  "SIN", "JFK", "SG", 15332, aircraft_type="A350ULR", base_load_factor=0.83),
    Route("sin-ewr",  "SIN->EWR",  "SIN", "EWR", "SG", 15349, aircraft_type="A350ULR", base_load_factor=0.83),
    Route("sin-lax",  "SIN->LAX",  "SIN", "LAX", "SG", 14114, aircraft_type="A350",    base_load_factor=0.83),
    Route("sin-sfo",  "SIN->SFO",  "SIN", "SFO", "SG", 13591, aircraft_type="A350",    base_load_factor=0.83),

    # ==========================================================================
    # HONG KONG -> NORTH AMERICA
    # ==========================================================================

    Route("hkg-lax",  "HKG->LAX",  "HKG", "LAX", "HK", 11636, aircraft_type="B777",  base_load_factor=0.83),
    Route("hkg-jfk",  "HKG->JFK",  "HKG", "JFK", "HK", 13974, aircraft_type="A350",  base_load_factor=0.83),  # CX A350-900
    Route("hkg-yvr",  "HKG->YVR",  "HKG", "YVR", "HK", 10057, aircraft_type="B777",  base_load_factor=0.83),

    # ==========================================================================
    # TAIWAN -> NORTH AMERICA
    # ==========================================================================

    Route("tpe-lax",  "TPE->LAX",  "TPE", "LAX", "TW", 10162, aircraft_type="B777",  base_load_factor=0.83),  # EVA / CI
    Route("tpe-sfo",  "TPE->SFO",  "TPE", "SFO", "TW", 9759,  aircraft_type="B789",  base_load_factor=0.83),
    Route("tpe-jfk",  "TPE->JFK",  "TPE", "JFK", "TW", 12838, aircraft_type="B777",  base_load_factor=0.83),
    Route("tpe-phx",  "TPE->PHX",  "TPE", "PHX", "TW", 10680, aircraft_type="B789",  base_load_factor=0.83),  # CI new Dec 2025

    # ==========================================================================
    # AUSTRALIA -> NORTH AMERICA
    # ==========================================================================

    # Significant route expansion Dec 2025-Jan 2026
    # Qantas flagship + United partnership dominates; AA/DL expanding aggressively
    Route("syd-lax",  "SYD->LAX",  "SYD", "LAX", "AU", 12077, aircraft_type="A380",  base_load_factor=0.83),  # QF A380 flagship
    Route("syd-sfo",  "SYD->SFO",  "SYD", "SFO", "AU", 11942, aircraft_type="B789",  base_load_factor=0.83),  # QF / UA
    Route("syd-dfw",  "SYD->DFW",  "SYD", "DFW", "AU", 13810, aircraft_type="A380",  base_load_factor=0.83),  # QF A380 Jan 2026
    Route("syd-iah",  "SYD->IAH",  "SYD", "IAH", "AU", 13804, aircraft_type="B789",  base_load_factor=0.83),  # UA 787-9 (longest UA route)
    Route("mel-lax",  "MEL->LAX",  "MEL", "LAX", "AU", 12750, aircraft_type="A380",  base_load_factor=0.83),  # QF A380
    Route("mel-sfo",  "MEL->SFO",  "MEL", "SFO", "AU", 12617, aircraft_type="B789",  base_load_factor=0.83),  # UA 787-9
    Route("mel-lax-dl","MEL->LAX (DL)","MEL","LAX","AU", 12750, aircraft_type="A350", base_load_factor=0.83),  # DL new Dec 2025
    Route("bne-lax",  "BNE->LAX",  "BNE", "LAX", "AU", 12981, aircraft_type="B789",  base_load_factor=0.83),  # AA new Dec 2025
    Route("bne-sfo",  "BNE->SFO",  "BNE", "SFO", "AU", 12847, aircraft_type="B789",  base_load_factor=0.83),  # UA
    Route("adl-sfo",  "ADL->SFO",  "ADL", "SFO", "AU", 13019, aircraft_type="B789",  base_load_factor=0.83),  # UA new Dec 2025

    # ==========================================================================
    # CANADA GATEWAY ROUTES
    # ==========================================================================

    Route("pvg-yvr",  "PVG->YVR",  "PVG", "YVR", "CN", 9152,  aircraft_type="B789",  base_load_factor=0.83),
    Route("hkg-yvr2", "HKG->YVR",  "HKG", "YVR", "HK", 10057, aircraft_type="B777",  base_load_factor=0.83),
    Route("icn-yvr2", "ICN->YVR",  "ICN", "YVR", "KR", 8220,  aircraft_type="B789",  base_load_factor=0.83),

    # ==========================================================================
    # UAE (Emirates / Etihad hub routes to US)
    # ==========================================================================

    Route("dxb-jfk",  "DXB->JFK",  "DXB", "JFK", "AE", 11020, aircraft_type="A380",  base_load_factor=0.83),
    Route("dxb-lax",  "DXB->LAX",  "DXB", "LAX", "AE", 13420, aircraft_type="A380",  base_load_factor=0.83),

    # ==========================================================================
    # EUROPE CONNECTING VIA ASIA HUBS (6th freedom flows)
    # ==========================================================================

    Route("eu-icn-lax", "ICN->LAX (EU)", "ICN", "LAX", "GB", 9370,
          is_connecting=True, connecting_hub="ICN", aircraft_type="B777", base_load_factor=0.83),
    Route("eu-sin-lax", "SIN->LAX (EU)", "SIN", "LAX", "GB", 14114,
          is_connecting=True, connecting_hub="SIN", aircraft_type="A350", base_load_factor=0.83),
    Route("eu-hnd-lax", "HND->LAX (EU)", "HND", "LAX", "DE", 8815,
          is_connecting=True, connecting_hub="HND", aircraft_type="B777", base_load_factor=0.83),
]


# ==============================================================================
# AIRCRAFT
# ==============================================================================
# Seat configs based on dominant operator layouts on transpacific as of 2025/26.
# max_payload_kg = structural payload limit (passengers + baggage, no cargo modelled).

AIRCRAFT: dict[str, Aircraft] = {

    "B777": Aircraft(
        # Boeing 777-300ER -- transpacific workhorse (AA, UA, KE, NH, JL, CX)
        type="B777",
        total_seats=396,
        first_seats=8,
        business_seats=52,
        premium_eco_seats=24,
        economy_seats=312,
        max_payload_kg=100_000,
    ),

    "B787": Aircraft(
        # Boeing 787-8 -- shorter-range Dreamliner (NGO-LAX, KIX-LAX, secondary routes)
        type="B787",
        total_seats=246,
        first_seats=0,
        business_seats=28,
        premium_eco_seats=21,
        economy_seats=197,
        max_payload_kg=52_000,
    ),

    "B789": Aircraft(
        # Boeing 787-9 -- dominant on new AU routes + ultra-long-haul (UA, AA, QF)
        # UA premium config: 48J + 21W + 189Y = 258 seats
        type="B789",
        total_seats=258,
        first_seats=0,
        business_seats=48,
        premium_eco_seats=21,
        economy_seats=189,
        max_payload_kg=56_000,
    ),

    "A350": Aircraft(
        # Airbus A350-900 -- Delta transpacific standard (all routes from Apr 2026)
        # DL config: 32J + 48W + 226Y = 306 seats
        type="A350",
        total_seats=306,
        first_seats=0,
        business_seats=32,
        premium_eco_seats=48,
        economy_seats=226,
        max_payload_kg=65_000,
    ),

    "A350ULR": Aircraft(
        # Airbus A350-900ULR -- Singapore Airlines SIN-JFK / SIN-EWR
        # Premium-only; no economy cabin. 161 seats total.
        type="A350ULR",
        total_seats=161,
        first_seats=0,
        business_seats=67,
        premium_eco_seats=94,
        economy_seats=0,
        max_payload_kg=32_000,
    ),

    "A380": Aircraft(
        # Airbus A380-800 -- Qantas SYD/MEL-LAX/DFW; Emirates DXB routes
        # QF config: 14F + 76J + 60W + 341Y = 491 seats
        type="A380",
        total_seats=491,
        first_seats=14,
        business_seats=76,
        premium_eco_seats=60,
        economy_seats=341,
        max_payload_kg=150_000,
    ),
}


# ==============================================================================
# AIRPORT → COUNTRY MAPPING
# ==============================================================================
# Maps destination airport IATA codes to ISO-2 country codes.
# Used to derive true O-D country pairs independent of routing.

AIRPORT_COUNTRY: dict[str, str] = {
    # United States
    "LAX": "US", "SFO": "US", "JFK": "US", "EWR": "US",
    "ORD": "US", "SEA": "US", "IAH": "US", "DFW": "US",
    "ATL": "US", "PHX": "US",
    # Canada
    "YVR": "CA", "YYZ": "CA", "YUL": "CA",
}
