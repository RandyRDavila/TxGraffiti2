#!/usr/bin/env python3
"""
Build a dataset of integer invariants/predicates geared to reproducing or
hinting toward famous number-theoretic theorems and conjectures.

Outputs a CSV with one row per n = 1..N.

Default N = 100_000 (safe on a laptop). You can increase N (e.g., 200k-500k)
if you’re patient. Collatz lengths are optional (memoized; can be slow).

Features include:
- Classical multiplicative functions: tau, sigma, phi, lambda (Carmichael), rad,
  omega, Omega, P_min, P_max, average_divisor, abundancy, etc.
- Ratios & analytic-style proxies: sigma/(n log log n), (phi * log log n)/n,
  sigma*phi/n^2, smoothness ratio, entropy of exponent vector.
- Prime-related: is_prime, prime index, next/prev prime, prime gap, normalized gap.
- Additive/representation: r2(n) (sum of two squares count), is_sum_three_squares,
  feasibility for sum of three cubes (mod 9 obstruction), Goldbach(3) truth flag.
- Structural Booleans: perfect/abundant/deficient, squarefree, semiprime, prime power,
  highly composite, superabundant, Carmichael, “prime without digit 7”, etc.
- Misc: Collatz length (optional, memoized), digit stats.

USAGE:
  python build_integer_dataset.py --N 100000 --out integers_100k.csv --collatz 0

NOTES:
- r2(n) uses the formula r2(n) = 4 * Prod_{p≡1 mod 4} (a_p + 1),
  provided every p≡3 (mod 4) has even exponent; otherwise r2(n)=0.
- is_sum_three_squares uses Legendre’s 3-squares theorem:
  n is representable as x^2 + y^2 + z^2  iff n != 4^a (8b+7).
- Goldbach(3) is a theorem (Helfgott): every odd n>=7 is sum of three primes.
  We include a boolean goldbach3_true accordingly.
- Robin’s inequality proxy columns are provided via sigma/(n log log n) and friends,
  to let a conjecturing pipeline learn thresholds like ~e^gamma empirically.

Author: (you)
"""

import argparse
import csv
import math
from collections import defaultdict
from functools import lru_cache

# ----------------------------
# Utilities: sieve and factors
# ----------------------------

def spf_sieve(N: int):
    """Compute smallest prime factor (spf) for 1..N; also return prime list."""
    spf = [0] * (N + 1)
    primes = []
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
        j = 0
        while j < len(primes):
            p = primes[j]
            v = p * i
            if v > N or p > spf[i]:
                break
            spf[v] = p
            j += 1
    return spf, primes

def factorize(n: int, spf):
    """Return dict {prime: exponent} using spf table."""
    if n <= 1:
        return {}
    f = {}
    while n > 1:
        p = spf[n]
        e = 0
        while n % p == 0:
            n //= p
            e += 1
        f[p] = e
    return f

# ----------------------------------
# Multiplicative/arithmetic features
# ----------------------------------

def tau_from_factors(factors: dict) -> int:
    t = 1
    for e in factors.values():
        t *= (e + 1)
    return t

def sigma_from_factors(factors: dict) -> int:
    # sigma(n) = ∏ (p^(e+1)-1)/(p-1)
    s = 1
    for p, e in factors.items():
        s *= (p**(e + 1) - 1) // (p - 1)
    return s

def phi_from_factors(n: int, factors: dict) -> int:
    # φ(n) = n ∏ (1 - 1/p)
    if n == 0:
        return 0
    res = n
    for p in factors.keys():
        res -= res // p
    return res

def carmichael_lambda_from_factors(factors: dict) -> int:
    """
    λ(n) (Carmichael function): lcm of λ(p^e) across prime powers.
    Rules:
      - For odd p: λ(p^e) = φ(p^e) = p^(e-1) (p-1)
      - For p=2:
         * e=1: λ(2)=1
         * e=2: λ(4)=2
         * e>=3: λ(2^e) = 2^(e-2)
    """
    def lam_pp(p, e):
        if p == 2:
            if e == 1: return 1
            if e == 2: return 2
            return 1 << (e - 2)  # 2^(e-2)
        # odd p
        return (p - 1) * (p ** (e - 1))

    l = 1
    for p, e in factors.items():
        l = lcm(l, lam_pp(p, e))
    return l

def rad_from_factors(factors: dict) -> int:
    r = 1
    for p in factors.keys():
        r *= p
    return r

def omega_from_factors(factors: dict) -> int:
    return len(factors)  # distinct primes

def big_omega_from_factors(factors: dict) -> int:
    return sum(factors.values())  # total prime factors counting multiplicity

def prime_min_max_from_factors(factors: dict):
    if not factors:
        return (None, None)
    ps = sorted(factors.keys())
    return ps[0], ps[-1]

def entropy_exponents(factors: dict) -> float:
    """
    Shannon entropy of exponent multiset normalized by Omega:
      H = - sum_i (a_i / Ω) * log(a_i / Ω)
    """
    Ω = big_omega_from_factors(factors)
    if Ω == 0:
        return 0.0
    h = 0.0
    for a in factors.values():
        p = a / Ω
        h -= p * math.log(p)
    return h

def smoothness_ratio(n: int, pmax: int | None) -> float:
    """Return log(Pmax)/log(n) (0 for n<=1)."""
    if n <= 1 or pmax in (None, 1):
        return 0.0
    return math.log(pmax) / math.log(n)

# --------------------------
# Prime-related computations
# --------------------------

def prime_index_map(primes):
    """Map prime -> its 1-based index π(p)."""
    return {p: i + 1 for i, p in enumerate(primes)}

def prime_gaps(primes):
    """Return dicts: gap_next[p], gap_prev[p], next_prime[p], prev_prime[p]."""
    gap_next = {}
    gap_prev = {}
    nxt = {}
    prv = {}
    for i, p in enumerate(primes):
        if i + 1 < len(primes):
            q = primes[i + 1]
            gap_next[p] = q - p
            nxt[p] = q
        else:
            gap_next[p] = None
            nxt[p] = None
        if i - 1 >= 0:
            r = primes[i - 1]
            gap_prev[p] = p - r
            prv[p] = r
        else:
            gap_prev[p] = None
            prv[p] = None
    return gap_next, gap_prev, nxt, prv

# -----------------------
# Additive/representation
# -----------------------

def r2_sum_two_squares_count(factors: dict) -> int:
    """
    r2(n) = number of integer pairs (x,y) with x^2 + y^2 = n,
    counting order and sign. Formula:
      r2(n) = 4 * ∏_{p≡1 mod 4} (a_p + 1),
    if all primes p≡3 (mod 4) have even exponent; otherwise r2(n)=0.
    """
    # check 4k+3 parity condition
    for p, e in factors.items():
        if p % 4 == 3 and (e % 2 == 1):
            return 0
    # product over p≡1 mod 4
    prod = 1
    for p, e in factors.items():
        if p % 4 == 1:
            prod *= (e + 1)
    return 4 * prod

def is_sum_three_squares(n: int) -> bool:
    """
    Legendre's three-square theorem:
    n is representable as x^2 + y^2 + z^2 iff n != 4^a (8b + 7).
    """
    m = n
    while m % 4 == 0:
        m //= 4
    return (m % 8) != 7

def three_cubes_feasible_mod9(n: int) -> bool:
    """
    Sum of three cubes modulo 9 obstruction:
    n ≡ 4 or 5 (mod 9) is impossible; others are potentially feasible.
    """
    r = n % 9
    return r not in (4, 5)

# -------------
# Classifiers
# -------------

def is_perfect(sigma_n: int, n: int) -> bool:
    return sigma_n == 2 * n

def is_abundant(sigma_n: int, n: int) -> bool:
    return sigma_n > 2 * n

def is_deficient(sigma_n: int, n: int) -> bool:
    return sigma_n < 2 * n

def is_semiprime(factors: dict) -> bool:
    return big_omega_from_factors(factors) == 2

def is_prime_power(factors: dict) -> bool:
    return len(factors) == 1

def is_squarefree(factors: dict) -> bool:
    return all(e == 1 for e in factors.values())

def is_carmichael(n: int, factors: dict) -> bool:
    """
    Korselt's criterion: n is Carmichael iff
    - n is composite
    - n is squarefree
    - for every prime p|n, (p-1) | (n-1)
    """
    if n < 3:
        return False
    if len(factors) == 0 or len(factors) == 1:
        return False  # prime or prime power
    if not is_squarefree(factors):
        return False
    for p in factors:
        if (n - 1) % (p - 1) != 0:
            return False
    return True

def digitsum(n: int) -> int:
    return sum(int(ch) for ch in str(n))

def has_digit(n: int, d: str) -> bool:
    return d in str(n)

# ----------------
# Collatz (opt-in)
# ----------------

@lru_cache(maxsize=None)
def collatz_length(n: int) -> int:
    if n <= 1:
        return 0
    if n % 2 == 0:
        return 1 + collatz_length(n // 2)
    else:
        return 1 + collatz_length(3 * n + 1)

# -----------
# Math utils
# -----------

def safe_log(x: float) -> float:
    return math.log(x) if x > 0 else float("nan")

def safe_loglog(x: float) -> float:
    return math.log(math.log(x)) if x > math.e else float("nan")

def lcm(a: int, b: int) -> int:
    return a // math.gcd(a, b) * b

# -----------------------------------
# Highly composite / superabundant
# -----------------------------------

def tag_highly_composite(tau_list):
    """
    highly_composite[n] = True if τ(n) > τ(m) for all m < n.
    """
    res = [False] * len(tau_list)
    best = 0
    for n in range(1, len(tau_list)):
        if tau_list[n] > best:
            res[n] = True
            best = tau_list[n]
    return res

def tag_superabundant(sigma_over_n):
    """
    superabundant[n] = True if σ(n)/n > σ(m)/m for all m < n.
    """
    res = [False] * len(sigma_over_n)
    best = 0.0
    for n in range(1, len(sigma_over_n)):
        if sigma_over_n[n] > best:
            res[n] = True
            best = sigma_over_n[n]
    return res

# -----------
# Main build
# -----------

def build_dataset(N: int, out_path: str, include_collatz: bool):
    spf, primes = spf_sieve(N)
    prime_set = set(primes)
    pi_index = prime_index_map(primes)
    gap_next, gap_prev, next_p, prev_p = prime_gaps(primes)

    # Precompute multiplicative functions
    tau_list = [0] * (N + 1)
    sigma_list = [0] * (N + 1)
    phi_list = [0] * (N + 1)
    lam_list = [0] * (N + 1)
    rad_list = [0] * (N + 1)
    omega_list = [0] * (N + 1)
    bigOmega_list = [0] * (N + 1)
    pmin_list = [None] * (N + 1)
    pmax_list = [None] * (N + 1)
    ent_list = [0.0] * (N + 1)
    r2_list = [0] * (N + 1)
    is3sq_list = [False] * (N + 1)
    feasible3cubes = [False] * (N + 1)

    # populate
    for n in range(1, N + 1):
        f = factorize(n, spf)
        tau_list[n] = tau_from_factors(f)
        sigma_list[n] = sigma_from_factors(f)
        phi_list[n] = phi_from_factors(n, f) if n > 0 else 0
        lam_list[n] = carmichael_lambda_from_factors(f) if n > 1 else 1
        rad_list[n] = rad_from_factors(f) if n > 1 else 1
        omega_list[n] = omega_from_factors(f)
        bigOmega_list[n] = big_omega_from_factors(f)
        pmin, pmax = prime_min_max_from_factors(f)
        pmin_list[n] = pmin
        pmax_list[n] = pmax
        ent_list[n] = entropy_exponents(f)
        r2_list[n] = r2_sum_two_squares_count(f)
        is3sq_list[n] = is_sum_three_squares(n)
        feasible3cubes[n] = three_cubes_feasible_mod9(n)

    # global tags
    abundancy = [sigma_list[n] / n if n > 0 else float("nan") for n in range(N + 1)]
    sigma_over_n = abundancy
    highly_comp = tag_highly_composite(tau_list)
    superabund = tag_superabundant(sigma_over_n)

    # write CSV
    fields = [
        "n",
        # basic transforms
        "log_n", "loglog_n", "sqrt_n",
        # multiplicative core
        "tau", "sigma", "phi", "lambda_carmichael", "rad",
        "omega", "Omega",
        "prime_min", "prime_max",
        "avg_divisor", "abundancy_index",
        "entropy_exponents", "smoothness_ratio",
        # prime info
        "is_prime", "prime_index", "prev_prime", "next_prime",
        "gap_prev", "gap_next", "normalized_gap",  # gap/log(p)
        "prime_without_digit7",
        # additive/representation
        "r2_sum_two_squares",
        "is_sum_three_squares",
        "three_cubes_feasible_mod9",
        # classes
        "is_perfect", "is_abundant", "is_deficient",
        "is_squarefree", "is_semiprime", "is_prime_power",
        "is_carmichael",
        "is_highly_composite", "is_superabundant",
        # proxies toward famous inequalities
        "sigma_over_n_loglog",          # sigma / (n * log log n)
        "phi_times_loglog_over_n",      # (phi * log log n) / n
        "sigma_times_phi_over_n2",      # sigma * phi / n^2
        # goldbach(3) truth flag (Helfgott)
        "goldbach3_true",
        # digits / misc
        "digit_sum", "has_digit_7",
    ]
    if include_collatz:
        fields.append("collatz_length")

    with open(out_path, "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=fields)
        w.writeheader()
        for n in range(1, N + 1):
            is_prime = (n in prime_set)
            pidx = pi_index[n] if is_prime else None
            pp = prev_p[n] if is_prime else None
            np_ = next_p[n] if is_prime else None
            gp = gap_prev[n] if is_prime else None
            gn = gap_next[n] if is_prime else None
            norm_gap = (gn / math.log(n)) if (is_prime and gn is not None and n > 1) else None

            log_n = safe_log(n)
            loglog_n = safe_loglog(n)
            sqrt_n = math.sqrt(n)

            # ratios & proxies
            sigma_over_n_loglog = (sigma_list[n] / (n * loglog_n)) if (n > math.e and loglog_n == loglog_n) else float("nan")
            phi_times_loglog_over_n = (phi_list[n] * loglog_n / n) if (n > math.e and loglog_n == loglog_n) else float("nan")
            sigma_times_phi_over_n2 = (sigma_list[n] * phi_list[n]) / (n * n)

            # classes
            perf = is_perfect(sigma_list[n], n)
            abund = is_abundant(sigma_list[n], n)
            defic = is_deficient(sigma_list[n], n)
            f = factorize(n, spf)  # (cheap: n is small)
            sqfree = is_squarefree(f)
            semip = is_semiprime(f)
            ppow = is_prime_power(f)
            carm = is_carmichael(n, f)

            pmax = pmax_list[n]
            smooth = smoothness_ratio(n, pmax)

            row = {
                "n": n,
                "log_n": log_n,
                "loglog_n": loglog_n,
                "sqrt_n": sqrt_n,
                "tau": tau_list[n],
                "sigma": sigma_list[n],
                "phi": phi_list[n],
                "lambda_carmichael": lam_list[n],
                "rad": rad_list[n],
                "omega": omega_list[n],
                "Omega": bigOmega_list[n],
                "prime_min": pmin_list[n] if pmin_list[n] is not None else "",
                "prime_max": pmax if pmax is not None else "",
                "avg_divisor": sigma_list[n] / tau_list[n],
                "abundancy_index": abundancy[n],
                "entropy_exponents": ent_list[n],
                "smoothness_ratio": smooth,
                "is_prime": int(is_prime),
                "prime_index": pidx if pidx is not None else "",
                "prev_prime": pp if pp is not None else "",
                "next_prime": np_ if np_ is not None else "",
                "gap_prev": gp if gp is not None else "",
                "gap_next": gn if gn is not None else "",
                "normalized_gap": norm_gap if norm_gap is not None else "",
                "prime_without_digit7": int(is_prime and not has_digit(n, "7")),
                "r2_sum_two_squares": r2_list[n],
                "is_sum_three_squares": int(is3sq_list[n]),
                "three_cubes_feasible_mod9": int(feasible3cubes[n]),
                "is_perfect": int(perf),
                "is_abundant": int(abund),
                "is_deficient": int(defic),
                "is_squarefree": int(sqfree),
                "is_semiprime": int(semip),
                "is_prime_power": int(ppow),
                "is_carmichael": int(carm),
                "is_highly_composite": int(highly_comp[n]),
                "is_superabundant": int(superabund[n]),
                "sigma_over_n_loglog": sigma_over_n_loglog,
                "phi_times_loglog_over_n": phi_times_loglog_over_n,
                "sigma_times_phi_over_n2": sigma_times_phi_over_n2,
                "goldbach3_true": int((n % 2 == 1) and (n >= 7)),
                "digit_sum": digitsum(n),
                "has_digit_7": int(has_digit(n, "7")),
            }

            if include_collatz:
                row["collatz_length"] = collatz_length(n)

            w.writerow(row)

    print(f"Wrote {out_path} with N={N} rows.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=100_000, help="Max integer n to include (default 100000).")
    ap.add_argument("--out", type=str, default="integers_dataset.csv", help="Output CSV path.")
    ap.add_argument("--collatz", type=int, default=0, help="Include Collatz length (0/1).")
    args = ap.parse_args()

    build_dataset(N=args.N, out_path=args.out, include_collatz=bool(args.collatz))



if __name__ == "__main__":
    main()
