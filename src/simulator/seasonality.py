"""
seasonality.py
--------------
Monthly seasonal index functions per origin market profile.

Profiles implemented:
  - standard_seasonal     : generic Western/business (legacy fallback)
  - japan_seasonal        : quad-modal — Golden Week, Obon, Cherry Blossom, New Year
  - korea_seasonal        : quad-modal — Seollal, Summer, Chuseok, year-end
  - singapore_seasonal    : flat with June/Dec peaks (school hols + expat calendar)
  - australia_seasonal    : SOUTHERN HEMISPHERE — Dec/Jan peak, Jun/Jul secondary
  - middle_east_seasonal  : summer heat-escape dominant, with static Eid proxy
  - china_seasonal        : tri-modal — CNY/Feb, May Golden Week, Oct National Day

NOTE on Middle East / Korea / Japan lunar holidays:
  Eid, Seollal, and Chuseok shift each year by ~11 days (Islamic) or vary
  month to month (lunar). For production use, pass the actual holiday month
  per simulation year rather than using the fixed monthly proxy below.
  A helper function `adjust_for_lunar_shift()` is provided for this purpose.
"""

import numpy as np
from typing import Callable

# Month indices for clarity
JAN, FEB, MAR, APR, MAY, JUN = 0, 1, 2, 3, 4, 5
JUL, AUG, SEP, OCT, NOV, DEC = 6, 7, 8, 9, 10, 11


# ── Profile functions ─────────────────────────────────────────────────────────

def japan_seasonal(month: int, amplitude: float = 0.16) -> float:
    """
    Japan outbound — quad-modal:
      - Apr/May: Golden Week + cherry blossom (strongest single outbound spike)
      - Aug:     Obon mid-August family travel
      - Dec/Jan: Year-end / New Year Oshōgatsu
      - Mar:     Cherry blossom shoulder
    Troughs: Sep (post-Obon), Nov (quiet shoulder), Jun (rainy season)
    """
    base = {
        JAN:  0.60,   # New Year return, quieter
        FEB: -0.10,   # trough — cold, no major holiday
        MAR:  0.40,   # cherry blossom build
        APR:  0.90,   # cherry blossom peak + start of Golden Week
        MAY:  1.00,   # Golden Week — strongest outbound peak
        JUN: -0.20,   # rainy season, school in session
        JUL:  0.50,   # summer build
        AUG:  0.85,   # Obon mid-Aug
        SEP: -0.30,   # post-Obon slump
        OCT:  0.10,   # mild autumn
        NOV: -0.15,   # quiet before year-end
        DEC:  0.70,   # year-end / New Year Oshōgatsu prep
    }
    return 1.0 + amplitude * base[month]


def korea_seasonal(month: int, amplitude: float = 0.16) -> float:
    """
    South Korea outbound — quad-modal:
      - Jan/Feb: Seollal (Lunar New Year) — major outbound spike
      - Jul/Aug: Summer school holidays (6 weeks mid-Jul to late Aug)
      - Sep/Oct: Chuseok (harvest festival) — varies Sep–Oct
      - May:     Children's Day + spring break minor boost
    Troughs: Mar (post-Seollal), Jun, Nov
    """
    base = {
        JAN:  0.50,   # Seollal prep (exact month varies — proxy)
        FEB:  0.85,   # Seollal peak (often Feb) + early spring break
        MAR: -0.25,   # post-Seollal slump
        APR:  0.20,   # spring travel
        MAY:  0.55,   # Children's Day, spring break
        JUN: -0.10,   # pre-summer school in session
        JUL:  0.80,   # summer school holidays begin
        AUG:  1.00,   # peak summer outbound — strongest month
        SEP:  0.60,   # Chuseok (often Sep) outbound spike
        OCT:  0.40,   # Chuseok spillover, autumn travel
        NOV: -0.20,   # quiet
        DEC:  0.30,   # year-end / Christmas for Christian population (~30%)
    }
    return 1.0 + amplitude * base[month]


def singapore_seasonal(month: int, amplitude: float = 0.10) -> float:
    """
    Singapore outbound — relatively flat, two main peaks:
      - Jun:  School holidays (main June break)
      - Dec:  Year-end school holidays + Christmas/New Year for expat community
    Secondary:
      - Mar/Apr: School holidays (March break)
      - Sep:  Small break but also exam season — mixed
    Year-round business travel base suppresses extremes.
    Trough: Jan–Feb (post year-end), Sep (exam period)
    """
    base = {
        JAN: -0.20,   # post year-end quiet
        FEB: -0.10,   # CNY domestic gathering (reduces outbound)
        MAR:  0.40,   # school holiday (Mar break)
        APR:  0.30,   # spring school break
        MAY:  0.10,   # quiet
        JUN:  0.85,   # main school holiday — primary leisure peak
        JUL:  0.70,   # continues into Jul
        AUG:  0.30,   # National Day, moderate
        SEP: -0.25,   # exam season, low leisure
        OCT:  0.10,   # school holiday (late Oct)
        NOV:  0.20,   # pre-year-end build
        DEC:  1.00,   # year-end school + expat Christmas — strongest
    }
    return 1.0 + amplitude * base[month]


def australia_seasonal(month: int, amplitude: float = 0.18) -> float:
    """
    Australia outbound — SOUTHERN HEMISPHERE calendar.
    Primary peaks are INVERTED relative to Northern Hemisphere:
      - Dec/Jan: Australian summer + Christmas school holidays (strongest)
      - Jun/Jul: Australian winter school holidays — secondary peak for long-haul
      - Sep/Oct: Spring school holidays, domestic focus but some international
    Troughs: Feb–Mar (post-summer), Apr–May (autumn shoulder), Aug (quiet)
    
    Key: Dec/Jan also coincides with US/Europe Northern winter, which for
    transpacific routes creates compound demand with Northern Hemisphere holiday
    travelers going the other direction — routes can be very full both ways.
    """
    base = {
        JAN:  0.90,   # Australian summer peak — school hols, post-Christmas travel
        FEB: -0.15,   # post-summer quieter
        MAR: -0.20,   # autumn shoulder, business travel mild
        APR:  0.20,   # Easter school holidays
        MAY: -0.10,   # quiet autumn
        JUN:  0.65,   # winter school holidays — secondary long-haul peak
        JUL:  0.80,   # peak winter school break
        AUG: -0.10,   # mid-winter, between school breaks
        SEP:  0.35,   # spring school break (Sep–Oct)
        OCT:  0.30,   # spring school break continues
        NOV:  0.10,   # pre-summer build
        DEC:  1.00,   # Christmas school break — strongest outbound month
    }
    return 1.0 + amplitude * base[month]


def middle_east_seasonal(month: int, amplitude: float = 0.20) -> float:
    """
    Middle East (UAE / Saudi Arabia) outbound — unique dual driver:

    DRIVER 1 — HEAT ESCAPE (fixed calendar):
      Jun–Sep: Extreme summer heat (45°C+) drives wealthy outbound leisure
      to Europe, Asia, cooler destinations. This is the dominant structural peak.

    DRIVER 2 — ISLAMIC CALENDAR (shifts ~11 days/year):
      Eid al-Fitr (end of Ramadan) and Eid al-Adha (Hajj season) are the
      two major Islamic travel windows. Ramadan itself SUPPRESSES outbound
      leisure (fasting; devout stay home).
      
      This profile uses fixed monthly proxies. For production simulation,
      use the `eid_adjusted_index()` function below which accepts the
      actual Eid month for a given year.

    Fixed monthly proxy (assumes Eid roughly Apr and Jul — 2023-era position):
      Adjust with `eid_adjusted_index()` for accurate year-by-year values.
    """
    base = {
        JAN:  0.10,   # mild winter, moderate leisure
        FEB: -0.05,   # quiet (Ramadan-adjacent in some years)
        MAR: -0.30,   # Ramadan (proxy — suppresses leisure)
        APR:  0.60,   # Eid al-Fitr travel spike (proxy)
        MAY:  0.30,   # post-Eid moderate
        JUN:  0.80,   # heat escape begins
        JUL:  1.00,   # peak summer escape — hottest month, school break
        AUG:  0.90,   # sustained summer escape
        SEP:  0.50,   # summer tapering, Eid al-Adha proxy
        OCT:  0.10,   # cooler, less urgency to travel
        NOV:  0.05,   # quiet
        DEC:  0.35,   # expat Christmas travel + year-end
    }
    return 1.0 + amplitude * base[month]


def china_seasonal(month: int, amplitude: float = 0.18) -> float:
    """
    China outbound — tri-modal:
      - Feb:  Chinese New Year (CNY) — strongest single peak
      - May:  Labour Day Golden Week (May 1–7)
      - Oct:  National Day Golden Week (Oct 1–7)
    Troughs: Mar (post-CNY slump), Jun (exam season), Sep (pre-holiday dip)
    """
    base = {
        JAN:  0.05,   # pre-CNY build
        FEB:  1.00,   # CNY peak
        MAR: -0.30,   # post-CNY return slump
        APR:  0.10,   # mild spring
        MAY:  0.80,   # Labour Day Golden Week
        JUN: -0.20,   # exam season, low leisure
        JUL:  0.20,   # moderate summer
        AUG:  0.15,
        SEP: -0.25,   # pre-National Day dip
        OCT:  0.90,   # National Day Golden Week
        NOV: -0.10,   # quiet
        DEC:  0.10,   # mild year-end
    }
    return 1.0 + amplitude * base[month]


def standard_seasonal(month: int, amplitude: float = 0.14) -> float:
    """Generic fallback — sinusoidal Northern Hemisphere summer + Dec bump."""
    primary  = amplitude * np.sin(2 * np.pi * (month - 1.5) / 12)
    dec_bump = 0.04 * np.exp(-0.5 * ((month - DEC) / 0.8) ** 2)
    sep_dip  = -0.03 * np.exp(-0.5 * ((month - SEP) / 0.7) ** 2)
    return 1.0 + primary + dec_bump + sep_dip


# ── Dispatch table ────────────────────────────────────────────────────────────

SEASONAL_PROFILES: dict[str, Callable] = {
    "japan":          japan_seasonal,
    "korea":          korea_seasonal,
    "singapore":      singapore_seasonal,
    "australia":      australia_seasonal,
    "middle_east":    middle_east_seasonal,
    "china":          china_seasonal,
    "southeast_asia": singapore_seasonal,   # alias
    "standard":       standard_seasonal,
}


# ── Public API ────────────────────────────────────────────────────────────────

DOW_MULTIPLIERS: dict[int, float] = {
    0: 1.05,   # Monday    — business travel ramps up
    1: 0.88,   # Tuesday   — lowest demand
    2: 0.92,   # Wednesday — low (but above Tuesday)
    3: 1.02,   # Thursday  — moderate
    4: 1.15,   # Friday    — highest (business return + leisure start)
    5: 0.92,   # Saturday  — low (mid-trip leisure)
    6: 1.06,   # Sunday    — leisure return + business prep
}

DOW_NAMES: list[str] = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
]


def get_dow_index(dow: int) -> float:
    """
    Day-of-week demand multiplier.
    dow: 0=Monday … 6=Sunday (Python weekday() convention).
    """
    return DOW_MULTIPLIERS[dow]


def get_seasonal_index(profile: str, month: int, amplitude: float | None = None) -> float:
    """
    Returns seasonal demand multiplier for a given profile and month.

    Args:
        profile:   one of the keys in SEASONAL_PROFILES
        month:     0-indexed (0=Jan … 11=Dec)
        amplitude: override default amplitude if provided

    Returns:
        float multiplier, typically 0.70–1.35 depending on profile
    """
    fn = SEASONAL_PROFILES.get(profile, standard_seasonal)
    if amplitude is not None:
        return fn(month, amplitude=amplitude)
    return fn(month)


def seasonal_array(profile: str) -> np.ndarray:
    """Returns length-12 array of monthly indices for a profile (for plotting)."""
    return np.array([get_seasonal_index(profile, m) for m in range(12)])


# ── Lunar holiday adjustment helper ──────────────────────────────────────────

def eid_adjusted_index(
    base_month: int,
    eid_fitr_month: int,
    eid_adha_month: int,
    ramadan_month: int,
    amplitude: float = 0.20,
) -> float:
    """
    Middle East seasonal index adjusted for actual Islamic holiday months in a
    given year. Call this instead of middle_east_seasonal() for year-accurate sims.

    Args:
        base_month:      0-indexed calendar month to query
        eid_fitr_month:  actual month of Eid al-Fitr that year (0-indexed)
        eid_adha_month:  actual month of Eid al-Adha that year (0-indexed)
        ramadan_month:   actual start month of Ramadan that year (0-indexed)

    Returns:
        float seasonal multiplier
    """
    # Start with heat-escape base (structural driver)
    heat_base = {
        JAN: 0.10, FEB: -0.05, MAR: 0.00, APR: 0.10,
        MAY: 0.30, JUN: 0.80,  JUL: 1.00, AUG: 0.90,
        SEP: 0.50, OCT: 0.10,  NOV: 0.05, DEC: 0.35,
    }
    idx = 1.0 + amplitude * heat_base[base_month]

    # Eid spikes
    if base_month == eid_fitr_month:
        idx += amplitude * 0.60
    if base_month == eid_adha_month:
        idx += amplitude * 0.50

    # Ramadan suppression
    ramadan_months = [ramadan_month % 12, (ramadan_month + 1) % 12]
    if base_month in ramadan_months:
        idx -= amplitude * 0.40

    return float(np.clip(idx, 0.5, 1.5))


def seollal_adjusted_index(
    base_month: int,
    seollal_month: int,
    chuseok_month: int,
    amplitude: float = 0.16,
) -> float:
    """
    Korea seasonal index with actual Seollal and Chuseok months injected.
    Use this for year-accurate simulation instead of korea_seasonal().
    """
    base = {
        JAN: 0.10, FEB: 0.10, MAR: -0.25, APR: 0.20,
        MAY: 0.55, JUN: -0.10, JUL: 0.80, AUG: 1.00,
        SEP: 0.20, OCT: 0.20, NOV: -0.20, DEC: 0.30,
    }
    idx = 1.0 + amplitude * base[base_month]

    if base_month == seollal_month:
        idx += amplitude * 0.75
    if base_month == chuseok_month:
        idx += amplitude * 0.60

    return float(np.clip(idx, 0.5, 1.5))


# ── Lookup table: approximate Eid/Seollal months 2015–2027 ───────────────────

# (year -> eid_fitr_month_0indexed, eid_adha_month_0indexed, ramadan_start_0indexed)
EID_CALENDAR: dict[int, tuple[int, int, int]] = {
    2015: (6, 9, 5),   # Jul, Oct, Jun
    2016: (6, 8, 5),   # Jul, Sep, Jun
    2017: (5, 8, 4),   # Jun, Sep, May
    2018: (5, 7, 4),   # Jun, Aug, May
    2019: (5, 7, 4),   # Jun, Aug, May
    2020: (4, 6, 3),   # May, Jul, Apr
    2021: (4, 6, 3),   # May, Jul, Apr
    2022: (4, 6, 3),   # May, Jul, Apr
    2023: (3, 5, 2),   # Apr, Jun, Mar
    2024: (3, 5, 2),   # Apr, Jun, Mar
    2025: (2, 5, 1),   # Mar, Jun, Feb (approx)
    2026: (2, 4, 1),   # Mar, May, Feb (approx)
    2027: (1, 4, 0),   # Feb, May, Jan (approx)
}

# (year -> seollal_month_0indexed, chuseok_month_0indexed)
KOREAN_LUNAR_CALENDAR: dict[int, tuple[int, int]] = {
    2015: (1, 8),   # Feb, Sep
    2016: (1, 8),   # Feb, Sep
    2017: (0, 9),   # Jan, Oct
    2018: (1, 8),   # Feb, Sep
    2019: (1, 8),   # Feb, Sep
    2020: (0, 9),   # Jan, Oct
    2021: (1, 8),   # Feb, Sep
    2022: (1, 8),   # Feb, Sep
    2023: (0, 8),   # Jan, Sep
    2024: (1, 8),   # Feb, Sep
    2025: (0, 9),   # Jan, Oct
    2026: (1, 9),   # Feb, Oct
    2027: (1, 8),   # Feb, Sep
}
