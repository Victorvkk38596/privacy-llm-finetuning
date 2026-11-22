"""Independent RDP accountant for the Poisson-subsampled Gaussian (numpy only, no Opacus).

Integer orders only (Mironov, Talwar & Zhang 2019, exact binomial expansion), converted to
(eps, delta) with the same conversion Opacus' RDP accountant uses
(eps = rdp + log((a-1)/a) - (log delta + log a)/(a-1)). Restricting to integer orders can only
make eps slightly larger than Opacus' value (which also searches fractional orders), so this is
a conservative cross-check of logged epsilon_spent and of noise multipliers for new budgets.

  python scripts/rdp_crosscheck.py
"""
import math

import numpy as np

ORDERS = list(range(2, 257))


def _logsumexp(v):
    m = max(v)
    return m + math.log(sum(math.exp(x - m) for x in v))


def rdp_sampled_gaussian(q: float, sigma: float, alpha: int) -> float:
    terms = [math.lgamma(alpha + 1) - math.lgamma(k + 1) - math.lgamma(alpha - k + 1)
             + (alpha - k) * math.log1p(-q) + (k * math.log(q) if k else 0.0)
             + (k * k - k) / (2 * sigma ** 2) for k in range(alpha + 1)]
    return _logsumexp(terms) / (alpha - 1)


def epsilon(q: float, sigma: float, steps: int, delta: float) -> float:
    best = float("inf")
    for a in ORDERS:
        r = steps * rdp_sampled_gaussian(q, sigma, a)
        eps = r + math.log((a - 1) / a) - (math.log(delta) + math.log(a)) / (a - 1)
        best = min(best, eps)
    return best


def sigma_for(target_eps: float, q: float, steps: int, delta: float) -> float:
    lo, hi = 0.3, 50.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if epsilon(q, mid, steps, delta) > target_eps:
            lo = mid
        else:
            hi = mid
    return hi


if __name__ == "__main__":
    q, delta = 64 / 1024, 1e-5
    print("Cross-check of pilot accounting (logged by Opacus RDP accountant):")
    for sigma, steps, logged in [(1.3171, 300, 4.9934), (1.3684, 300, 4.6946), (2.0557, 1000, 4.9987)]:
        e = epsilon(q, sigma, steps, delta)
        print(f"  sigma={sigma}, T={steps}: eps_int_orders={e:.4f}  opacus_logged={logged}  "
              f"rel.diff={(e - logged) / logged:+.2%}")
    print("Noise multipliers implied for the grid (q=1/16, T=300, delta=1e-5; integer orders, conservative):")
    for t in (1.0, 5.0, 8.0):
        s = sigma_for(t, q, 300, delta)
        print(f"  target eps={t}: sigma ~= {s:.4f}")
