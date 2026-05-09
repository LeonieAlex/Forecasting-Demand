"""
shocks.py
---------
Market shock simulation engine.

Three shock types:
  1. GlobalShock     — affects ALL origin markets simultaneously (pandemic, oil crisis)
  2. CountryShock    — affects ONE country's outbound demand (currency crisis, domestic event)
  3. GeopoliticalShock — bilateral tension suppressing demand on specific country-pair routes

Each shock has:
  - start month
  - severity (negative multiplier, e.g. -0.25 = 25% demand drop at peak)
  - duration (months until full recovery)
  - decay shape (exponential by default)

The ShockEngine pre-generates all shocks for the full simulation horizon,
then lets you query the combined multiplier for any (month, country) pair.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Shock:
    shock_type: str          # 'global' | 'country' | 'geopolitical'
    start_month: int
    severity: float          # negative float, e.g. -0.20
    duration_months: int
    country: Optional[str]   # ISO-2 for country/geopolitical shocks, None for global
    counterpart: Optional[str] = None   # second country for geopolitical
    label: str = ""

    def multiplier_at(self, month: int) -> float:
        """
        Returns the demand multiplier contribution of this shock at `month`.
        Uses exponential decay from peak severity to zero over `duration_months`.

        Returns 0.0 if `month` is outside the shock window.
        """
        if month < self.start_month or month >= self.start_month + self.duration_months:
            return 0.0
        elapsed = month - self.start_month
        # Exponential decay: full severity at start, ~zero at end
        decay = np.exp(-3.0 * elapsed / self.duration_months)
        return self.severity * decay


class ShockEngine:
    """
    Generates and stores all shocks for a simulation run.
    Provides fast multiplier lookup per (month, country_id).
    """

    def __init__(
        self,
        n_months: int,
        country_ids: list[str],
        global_shock_rate: float = 0.8,     # shocks per year
        country_shock_rate: float = 1.5,    # shocks per year per country
        geopolitical_rate: float = 0.4,     # shocks per year
        rng: np.random.Generator | None = None,
    ):
        self.n_months = n_months
        self.country_ids = country_ids
        self.rng = rng or np.random.default_rng()
        self.shocks: list[Shock] = []

        self._generate_global_shocks(global_shock_rate)
        self._generate_country_shocks(country_shock_rate)
        self._generate_geopolitical_shocks(geopolitical_rate)

        # Pre-compute combined multiplier array: shape (n_months, n_countries)
        self._country_index = {c: i for i, c in enumerate(country_ids)}
        self._multiplier_cache = self._build_cache()

    # ── Generation ────────────────────────────────────────────────────────────

    def _generate_global_shocks(self, rate_per_year: float):
        """Poisson-distributed global shocks affecting all markets."""
        prob_per_month = rate_per_year / 12
        for m in range(self.n_months):
            if self.rng.random() < prob_per_month:
                severity = -self.rng.uniform(0.08, 0.35)
                duration = int(self.rng.integers(4, 14))
                self.shocks.append(Shock(
                    shock_type="global",
                    start_month=m,
                    severity=severity,
                    duration_months=duration,
                    country=None,
                    label=f"Global demand shock (sev={severity:.0%}, dur={duration}mo)",
                ))

    def _generate_country_shocks(self, rate_per_year: float):
        """
        Country-specific shocks — currency crisis, political events,
        disease outbreaks, visa restrictions, etc.
        Rate is per country per year.
        """
        prob_per_month = rate_per_year / 12
        for country_id in self.country_ids:
            for m in range(self.n_months):
                if self.rng.random() < prob_per_month:
                    severity = -self.rng.uniform(0.05, 0.40)
                    duration = int(self.rng.integers(2, 15))
                    self.shocks.append(Shock(
                        shock_type="country",
                        start_month=m,
                        severity=severity,
                        duration_months=duration,
                        country=country_id,
                        label=(
                            f"{country_id} macro shock "
                            f"(sev={severity:.0%}, dur={duration}mo)"
                        ),
                    ))

    def _generate_geopolitical_shocks(self, rate_per_year: float):
        """
        Bilateral tension shocks. Defined pairs reflect historically
        plausible friction routes in the transpacific context.
        Geopolitical shocks have partial effect (70% of severity vs country shocks).
        """
        # (origin_market, destination_country) friction pairs
        pairs = [
            ("CN", "US"),
            ("JP", "US"),
            ("KR", "US"),
            ("CN", "JP"),   # e.g. island disputes affecting sentiment
            ("CN", "KR"),   # e.g. THAAD-style events
        ]
        prob_per_month = rate_per_year / 12
        for m in range(self.n_months):
            if self.rng.random() < prob_per_month:
                origin, counterpart = pairs[self.rng.integers(0, len(pairs))]
                if origin not in self.country_ids:
                    continue
                severity = -self.rng.uniform(0.04, 0.22) * 0.70
                duration = int(self.rng.integers(3, 20))
                self.shocks.append(Shock(
                    shock_type="geopolitical",
                    start_month=m,
                    severity=severity,
                    duration_months=duration,
                    country=origin,
                    counterpart=counterpart,
                    label=(
                        f"{origin}<>{counterpart} geopolitical tension "
                        f"(sev={severity:.0%}, dur={duration}mo)"
                    ),
                ))

    # ── Cache & lookup ─────────────────────────────────────────────────────────

    def _build_cache(self) -> np.ndarray:
        """
        Build a (n_months, n_countries) multiplier array.
        Each cell is the PRODUCT of all shock contributions at that month for that country.
        Clipped to [0.25, 1.05] — demand can't go negative or rise from shocks.
        """
        n_c = len(self.country_ids)
        cache = np.zeros((self.n_months, n_c))  # additive contributions

        for shock in self.shocks:
            for m in range(self.n_months):
                contrib = shock.multiplier_at(m)
                if contrib == 0.0:
                    continue
                if shock.shock_type == "global":
                    cache[m, :] += contrib
                elif shock.country in self._country_index:
                    idx = self._country_index[shock.country]
                    cache[m, idx] += contrib

        # Convert additive deltas → multipliers, clip
        result = np.clip(1.0 + cache, 0.25, 1.05)
        return result

    def get_multiplier(self, month: int, country_id: str) -> float:
        """Returns combined shock multiplier for (month, country). 1.0 = no shock."""
        if country_id not in self._country_index:
            return 1.0
        return float(self._multiplier_cache[month, self._country_index[country_id]])

    def get_multiplier_array(self, country_id: str) -> np.ndarray:
        """Returns length-n_months array of multipliers for a country."""
        if country_id not in self._country_index:
            return np.ones(self.n_months)
        return self._multiplier_cache[:, self._country_index[country_id]]

    def shock_summary(self) -> list[dict]:
        """Returns list of shock dicts for logging/export."""
        return [
            {
                "type": s.shock_type,
                "start_month": s.start_month,
                "severity": round(s.severity, 4),
                "duration_months": s.duration_months,
                "country": s.country or "ALL",
                "counterpart": s.counterpart or "",
                "label": s.label,
            }
            for s in sorted(self.shocks, key=lambda x: x.start_month)
        ]
