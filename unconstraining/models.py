import math
import numpy as np
from numba import njit
from abc import ABC, abstractmethod

# ==========================================================
# NUMBA ACCELERATED MATH HELPERS
# ==========================================================
SQRT_2PI = math.sqrt(2.0 * math.pi)
SQRT_2 = math.sqrt(2.0)

@njit(fastmath=True)
def numba_norm_pdf(x):
    return math.exp(-0.5 * x**2) / SQRT_2PI

@njit(fastmath=True)
def numba_norm_cdf(x):
    # The CDF can be perfectly calculated using the error function (erf)
    return 0.5 * (1.0 + math.erf(x / SQRT_2))


# ==========================================================
# COMPILED C-SPEED CORE LOOPS
# ==========================================================

@njit(fastmath=True)
def _em_numba_core(y, cens_idx, cap, mu, sig, max_iter, tol):
    """Core EM loop for standard unconstraining (No Price)."""
    n = len(y)

    for i in range(max_iter):
        prev_mu = mu
        prev_sig = sig

        # ---------- E STEP ----------
        for j in range(len(cens_idx)):
            idx = cens_idx[j]

            # Standardize
            a = (cap[idx] - mu) / sig
            if a > 5.0: a = 5.0
            elif a < -5.0: a = -5.0

            tail = 1.0 - numba_norm_cdf(a)
            if tail < 1e-12: tail = 1e-12

            lam = numba_norm_pdf(a) / tail
            y[idx] = mu + sig * lam

        # ---------- M STEP ----------
        sum_y = 0.0
        for j in range(n):
            sum_y += y[j]
        mu = sum_y / n

        sum_sq_err = 0.0
        for j in range(n):
            resid = y[j] - mu
            sum_sq_err += resid * resid

        sig = math.sqrt(sum_sq_err / n)
        if sig < 1.0: sig = 1.0

        # ---------- Convergence ----------
        diff = abs(mu - prev_mu) + abs(sig - prev_sig)
        if diff < tol:
            break

    return y, mu, sig


@njit(fastmath=True)
def _em_price_numba_core(y, cens_idx, cap, p, b0, b1, sig, max_iter, tol):
    """Core EM loop for price-dependent unconstraining."""
    n = len(y)

    for i in range(max_iter):
        prev_b0 = b0
        prev_b1 = b1
        prev_sig = sig

        # ---------- E STEP ----------
        for j in range(len(cens_idx)):
            idx = cens_idx[j]
            mu = b0 + b1 * p[idx]

            # Standardize
            a = (cap[idx] - mu) / sig
            if a > 5.0: a = 5.0
            elif a < -5.0: a = -5.0

            tail = 1.0 - numba_norm_cdf(a)
            if tail < 1e-12: tail = 1e-12

            lam = numba_norm_pdf(a) / tail
            y[idx] = mu + sig * lam

        # ---------- M STEP ----------
        # Manual Simple Linear Regression is vastly faster than np.linalg.lstsq
        sum_p = 0.0
        sum_y = 0.0
        for j in range(n):
            sum_p += p[j]
            sum_y += y[j]
        mean_p = sum_p / n
        mean_y = sum_y / n

        num = 0.0
        den = 0.0
        for j in range(n):
            dp = p[j] - mean_p
            num += dp * (y[j] - mean_y)
            den += dp * dp

        if den > 1e-6:
            b1 = num / den
        else:
            b1 = 0.0
        b0 = mean_y - b1 * mean_p

        # Calculate Variance (Sigma)
        sum_sq_err = 0.0
        for j in range(n):
            resid = y[j] - (b0 + b1 * p[j])
            sum_sq_err += resid * resid

        sig = math.sqrt(sum_sq_err / n)
        if sig < 1.0: sig = 1.0

        # ---------- Convergence ----------
        diff = abs(b0 - prev_b0) + abs(b1 - prev_b1) + abs(sig - prev_sig)
        if diff < tol:
            break

    return y, b0, b1, sig


# ==========================================================
# CLASS WRAPPERS
# ==========================================================

class BaseUnconstrainer(ABC):
    def __init__(self):
        self.history = []

    @abstractmethod
    def fit(self, observed_bookings, is_censored, capacity, **kwargs):
        pass

    def evaluate(self, true_demand, estimated_demand):
        rmse = np.sqrt(np.mean((true_demand - estimated_demand) ** 2))
        mae = np.mean(np.abs(true_demand - estimated_demand))
        return {"RMSE": round(rmse, 2), "MAE": round(mae, 2)}


class NaiveUnconstrainer(BaseUnconstrainer):
    """
    Baseline Model: Ignores the capacity constraint.
    Assumes observed bookings perfectly represent latent demand.
    """
    def fit(self, observed_bookings, is_censored, capacity=None, price_per_kg=None, max_iter=None, tol=None):
        # We simply return the observations. No "lift" is applied.
        y = observed_bookings.copy().astype(float)
        return y


class EMUnconstrainer(BaseUnconstrainer):
    """
    Standard Truncated Normal EM, optimized with Numba.
    """
    def __init__(self):
        super().__init__()
        self.mu = None
        self.sigma = None

    def fit(self, observed_bookings, is_censored, capacity, **kwargs):
        self.history.clear()

        max_iter = kwargs.get('max_iter', 100)
        tol = kwargs.get('tol', 1e-5)

        # Extract arrays and ensure contiguous memory for C-compilation
        y = np.ascontiguousarray(observed_bookings, dtype=float).copy()
        cens = np.ascontiguousarray(is_censored, dtype=bool)

        if np.isscalar(capacity):
            cap = np.full(len(y), capacity, dtype=float)
        else:
            cap = np.ascontiguousarray(capacity, dtype=float)

        # Step 0: Initialization
        if np.sum(~cens) > 0:
            self.mu = np.mean(y[~cens])
            self.sigma = max(np.std(y[~cens]), 1.0)
        else:
            self.mu = np.mean(y)
            self.sigma = max(np.std(y), 1.0)

        cens_idx = np.where(cens)[0]

        # Call Compiled Engine
        if len(cens_idx) > 0:
            y, self.mu, self.sigma = _em_numba_core(
                y, cens_idx, cap,
                self.mu, self.sigma,
                max_iter, tol
            )

        return y


class EMPriceUnconstrainer(BaseUnconstrainer):
    """
    Price-dependent Truncated Normal EM, optimized with Numba.
    """
    def __init__(self):
        super().__init__()
        self.beta0 = None
        self.beta1 = None
        self.sigma = None

    def fit(self, observed_bookings, is_censored, capacity, **kwargs):
        self.history.clear()

        # Extract strict keyword arguments
        price_per_kg = kwargs.get('price_per_kg')
        if price_per_kg is None:
            raise ValueError("EMPriceUnconstrainer requires 'price_per_kg' to be passed.")

        max_iter = kwargs.get('max_iter', 100)
        tol = kwargs.get('tol', 1e-5)

        # Extract arrays and ensure contiguous memory for C-compilation
        y = np.ascontiguousarray(observed_bookings, dtype=float).copy()
        cens = np.ascontiguousarray(is_censored, dtype=bool)
        p = np.ascontiguousarray(price_per_kg, dtype=float)

        if np.all(y == 0) and not np.any(cens):
            return y

        if np.isscalar(capacity):
            cap = np.full(len(y), capacity, dtype=float)
        else:
            cap = np.ascontiguousarray(capacity, dtype=float)

        # Initial OLS (Run once in standard python using uncensored data)
        mask = ~cens
        if np.sum(mask) < 2 or np.std(p[mask]) < 1e-6:
            self.beta0 = np.mean(y)
            self.beta1 = 0.0
        else:
            X = np.column_stack([np.ones(np.sum(mask)), p[mask]])
            beta = np.linalg.lstsq(X, y[mask], rcond=None)[0]
            self.beta0 = beta[0]
            self.beta1 = beta[1]

        self.sigma = max(np.std(y[mask]) if np.sum(mask) > 1 else 1.0, 1.0)

        cens_idx = np.where(cens)[0]

        # Call Compiled Engine
        if len(cens_idx) > 0:
            y, self.beta0, self.beta1, self.sigma = _em_price_numba_core(
                y, cens_idx, cap, p,
                self.beta0, self.beta1, self.sigma,
                max_iter, tol
            )

        return y

# ==========================================================
# PROJECTION-DETRUNCATION (PD) UNCONSTRAINERS
# ==========================================================

# ==========================================================
# NUMBA INVERSE NORMAL CDF (ACKLAM'S APPROXIMATION)
# ==========================================================

@njit(fastmath=True)
def numba_norm_ppf(p):
    """
    Peter J. Acklam's rational approximation of the inverse standard normal CDF.
    Numba compatible and highly accurate.
    """
    if p <= 0.0: return -5.0
    if p >= 1.0: return 5.0

    # Coefficients
    a1 = -3.969683028665376e+01
    a2 =  2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 =  1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 =  2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 =  1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 =  6.680131188771972e+01
    b5 = -1.328068155288572e+01

    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 =  4.374664141464968e+00
    c6 =  2.938163982698783e+00

    d1 =  7.784695709041462e-03
    d2 =  3.224671290700398e-01
    d3 =  2.445134137142996e+00
    d4 =  3.754408661907416e+00

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        z = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        z = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        z = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
             ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)

    return z


# ==========================================================
# COMPILED PD CORE LOOPS
# ==========================================================

@njit(fastmath=True)
def _pd_numba_core(y, cens_idx, cap, mu, sig, tau, max_iter, tol):
    """Core PD loop for standard unconstraining (No Price)."""
    n = len(y)

    for i in range(max_iter):
        prev_mu = mu
        prev_sig = sig

        # ---------- E STEP (Projection) ----------
        for j in range(len(cens_idx)):
            idx = cens_idx[j]

            # Standardize
            a = (cap[idx] - mu) / sig
            if a > 5.0: a = 5.0
            elif a < -5.0: a = -5.0

            tail_prob = 1.0 - numba_norm_cdf(a)
            if tail_prob < 1e-12: tail_prob = 1e-12

            target_cdf = 1.0 - (tau * tail_prob)
            if target_cdf < 0.0: target_cdf = 0.0
            elif target_cdf > 1.0 - 1e-12: target_cdf = 1.0 - 1e-12

            z_hat_std = numba_norm_ppf(target_cdf)
            y[idx] = mu + sig * z_hat_std

        # ---------- M STEP (Detruncation) ----------
        sum_y = 0.0
        for j in range(n):
            sum_y += y[j]
        mu = sum_y / n

        sum_sq_err = 0.0
        for j in range(n):
            resid = y[j] - mu
            sum_sq_err += resid * resid

        sig = math.sqrt(sum_sq_err / n)
        if sig < 1.0: sig = 1.0

        # ---------- Convergence ----------
        diff = abs(mu - prev_mu) + abs(sig - prev_sig)
        if diff < tol:
            break

    return y, mu, sig


@njit(fastmath=True)
def _pd_price_numba_core(y, cens_idx, cap, p, b0, b1, sig, tau, max_iter, tol):
    """Core PD loop for price-dependent unconstraining."""
    n = len(y)

    for i in range(max_iter):
        prev_b0 = b0
        prev_b1 = b1
        prev_sig = sig

        # ---------- E STEP (Projection) ----------
        for j in range(len(cens_idx)):
            idx = cens_idx[j]
            mu = b0 + b1 * p[idx]

            # Standardize
            a = (cap[idx] - mu) / sig
            if a > 5.0: a = 5.0
            elif a < -5.0: a = -5.0

            tail_prob = 1.0 - numba_norm_cdf(a)
            if tail_prob < 1e-12: tail_prob = 1e-12

            target_cdf = 1.0 - (tau * tail_prob)
            if target_cdf < 0.0: target_cdf = 0.0
            elif target_cdf > 1.0 - 1e-12: target_cdf = 1.0 - 1e-12

            z_hat_std = numba_norm_ppf(target_cdf)
            y[idx] = mu + sig * z_hat_std

        # ---------- M STEP (Detruncation) ----------
        sum_p = 0.0
        sum_y = 0.0
        for j in range(n):
            sum_p += p[j]
            sum_y += y[j]
        mean_p = sum_p / n
        mean_y = sum_y / n

        num = 0.0
        den = 0.0
        for j in range(n):
            dp = p[j] - mean_p
            num += dp * (y[j] - mean_y)
            den += dp * dp

        if den > 1e-6:
            b1 = num / den
        else:
            b1 = 0.0
        b0 = mean_y - b1 * mean_p

        # Variance calculation
        sum_sq_err = 0.0
        for j in range(n):
            resid = y[j] - (b0 + b1 * p[j])
            sum_sq_err += resid * resid

        sig = math.sqrt(sum_sq_err / n)
        if sig < 1.0: sig = 1.0

        # ---------- Convergence ----------
        diff = abs(b0 - prev_b0) + abs(b1 - prev_b1) + abs(sig - prev_sig)
        if diff < tol:
            break

    return y, b0, b1, sig


# ==========================================================
# CLASS WRAPPERS
# ==========================================================

class PDUnconstrainer(BaseUnconstrainer):
    """
    Projection-Detruncation (PD) Algorithm, optimized with Numba.
    """
    def __init__(self):
        super().__init__()
        self.mu = None
        self.sigma = None

    def fit(self, observed_bookings, is_censored, capacity, **kwargs):
        self.history.clear()

        tau = kwargs.get('tau', 0.5)
        max_iter = kwargs.get('max_iter', 100)
        tol = kwargs.get('tol', 1e-5)

        y = np.ascontiguousarray(observed_bookings, dtype=float).copy()
        cens = np.ascontiguousarray(is_censored, dtype=bool)

        if np.isscalar(capacity):
            cap = np.full(len(y), capacity, dtype=float)
        else:
            cap = np.ascontiguousarray(capacity, dtype=float)

        if np.sum(~cens) > 0:
            self.mu = np.mean(y[~cens])
            self.sigma = max(np.std(y[~cens]), 1.0)
        else:
            self.mu = np.mean(y)
            self.sigma = max(np.std(y), 1.0)

        cens_idx = np.where(cens)[0]

        if len(cens_idx) > 0:
            y, self.mu, self.sigma = _pd_numba_core(
                y, cens_idx, cap,
                self.mu, self.sigma,
                tau, max_iter, tol
            )

        return y


class PDPriceUnconstrainer(BaseUnconstrainer):
    """
    Projection-Detruncation (PD) method with price-dependent mean, optimized with Numba.
    """
    def __init__(self):
        super().__init__()
        self.beta0 = None
        self.beta1 = None
        self.sigma = None

    def fit(self, observed_bookings, is_censored, capacity, **kwargs):
        self.history.clear()

        price_per_kg = kwargs.get('price_per_kg')
        if price_per_kg is None:
            raise ValueError("PDPriceUnconstrainer requires 'price_per_kg' to be passed.")

        tau = kwargs.get('tau', 0.5)
        max_iter = kwargs.get('max_iter', 100)
        tol = kwargs.get('tol', 1e-5)

        y = np.ascontiguousarray(observed_bookings, dtype=float).copy()
        cens = np.ascontiguousarray(is_censored, dtype=bool)
        p = np.ascontiguousarray(price_per_kg, dtype=float)

        if np.all(y == 0) and not np.any(cens):
            return y

        if np.isscalar(capacity):
            cap = np.full(len(y), capacity, dtype=float)
        else:
            cap = np.ascontiguousarray(capacity, dtype=float)

        mask = ~cens
        if np.sum(mask) < 2 or np.std(p[mask]) < 1e-6:
            self.beta0 = np.mean(y)
            self.beta1 = 0.0
        else:
            X = np.column_stack([np.ones(np.sum(mask)), p[mask]])
            beta = np.linalg.lstsq(X, y[mask], rcond=None)[0]
            self.beta0 = beta[0]
            self.beta1 = beta[1]

        self.sigma = max(np.std(y[mask]) if np.sum(mask) > 1 else 1.0, 1.0)

        cens_idx = np.where(cens)[0]

        if len(cens_idx) > 0:
            y, self.beta0, self.beta1, self.sigma = _pd_price_numba_core(
                y, cens_idx, cap, p,
                self.beta0, self.beta1, self.sigma,
                tau, max_iter, tol
            )

        return y
