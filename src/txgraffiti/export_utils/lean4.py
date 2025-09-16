# txgraffiti/lean/export.py
"""
Module for exporting TxGraffiti conjectures to Lean 4 syntax.

This module converts symbolic TxGraffiti `Conjecture` objects into
Lean-compatible theorems. It keeps hypotheses in `Prop`/`ℕ` (adding
`connected G` and `order G ≥ (2 : ℕ)` if missing), and makes the conclusion
type-correct by casting fractions to `ℚ`, typing all remaining integers,
and coercing invariants to a single ambient type in the conclusion.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping, List
import pandas as pd  # only for auto_var_map’s signature

from txgraffiti.logic import Conjecture

__all__ = [
    "conjecture_to_lean4",
    "necessary_conjecture_to_lean",
    "auto_var_map",
    "conjectures_to_lean",
]

# ─────────────────────────────────────────────────────────────────────────────
# 0) Optional: build a quick var-map from a DataFrame (not required by pipeline)
# ─────────────────────────────────────────────────────────────────────────────
def auto_var_map(df: pd.DataFrame, *, skip: tuple[str, ...] = ("name",)) -> dict[str, str]:
    """Return {column -> 'col G'} for Lean-friendly substitution."""
    return {c: f"{c} G" for c in df.columns if c not in skip}


# ─────────────────────────────────────────────────────────────────────────────
# 1) Header builder: renders hypotheses cleanly (Prop), raw inequality payload
# ─────────────────────────────────────────────────────────────────────────────
_REL_MAP = {"<=": "≤", "<": "<", ">=": "≥", ">": ">", "==": "=", "!=": "≠"}

def conjecture_to_lean4(
    conj: Conjecture,
    name: str,
    *,
    object_symbol: str = "G",
    object_decl: str = "SimpleGraph V",
) -> str:
    """
    Build a Lean theorem skeleton:
      theorem <name> (G : SimpleGraph V)
        (h1 : P1 G) (h2 : P2 G) ... :
        <raw-conclusion> :=
      sorry
    """
    # 1) Hypotheses (flatten only ANDs; necessary-conjectures typically use ∧)
    and_terms = getattr(conj.hypothesis, "_and_terms", [conj.hypothesis])
    binds = [f"(h{i} : {p.name} {object_symbol})" for i, p in enumerate(and_terms, 1)]

    # 2) Conclusion payload (symbolic, fix later passes)
    ineq = conj.conclusion
    lhs, op, rhs = ineq.lhs.name, _REL_MAP[ineq.op], ineq.rhs.name

    bind_block = "\n    ".join(binds) if binds else ""
    return (
        f"theorem {name} ({object_symbol} : {object_decl})\n"
        f"    {bind_block} : {lhs} {object_symbol} {op} {rhs} {object_symbol} :=\n"
        f"sorry\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2) Header hygiene: ensure (connected G) and (order G ≥ (2 : ℕ)) hypotheses
# ─────────────────────────────────────────────────────────────────────────────
RE_CONNECTED = re.compile(r"\(\s*h\d+\s*:\s*connected\s+G\s*\)")
RE_ORDER = re.compile(
    r"\(\s*h\d+\s*:\s*order\s+G\s*(?:>=|≥|<=|≤)\s*(?:2|\(2\s*:\s*ℕ\))\s*\)"
)

def _next_h_index(hyp_block: str) -> int:
    idxs = [int(m.group(1)) for m in re.finditer(r"\(\s*h(\d+)\s*:", hyp_block)]
    return (max(idxs) + 1) if idxs else 1

def _insert_missing_hypotheses(hdr_and_g: str, hyp_block: str) -> str:
    need_connected = RE_CONNECTED.search(hyp_block) is None
    need_order = RE_ORDER.search(hyp_block) is None
    if not (need_connected or need_order):
        return hdr_and_g + hyp_block

    indent_match = re.search(r"\n(\s*)\(", hyp_block)
    indent = indent_match.group(1) if indent_match else "    "
    n = _next_h_index(hyp_block)

    pieces = [hyp_block]
    if need_connected:
        pieces.append(f"\n{indent}(h{n} : connected G) ")
        n += 1
    if need_order:
        pieces.append(f"\n{indent}(h{n} : order G ≥ (2 : ℕ)) ")
    return hdr_and_g + "".join(pieces)

def _fix_header_and_hyps(text: str, func_names: Iterable[str]) -> str:
    """
    Add missing (connected G) and (order G ≥ (2 : ℕ)) in the header only.
    Do NOT touch the conclusion here.
    """
    theorems_pat = re.compile(
        r"(?P<before>theorem\b.*?\(G\s*:\s*SimpleGraph\s+V\))"
        r"(?P<hyps>(?:.*?\(h\d+\s*:.*?\)\s*)*)"
        r"(?P<colon>:\s)"
        r"(?P<expr>.*?)"
        r"(?P<assign>\s*:=)",
        flags=re.DOTALL,
    )

    def add_hyps(m: re.Match) -> str:
        before, hyps, colon, expr, assign = (
            m.group("before"),
            m.group("hyps") or "",
            m.group("colon"),
            m.group("expr"),
            m.group("assign"),
        )
        header_plus_hyps = _insert_missing_hypotheses(before, hyps)
        return f"{header_plus_hyps}{colon}{expr}{assign}"

    return theorems_pat.sub(add_hyps, text)


# ─────────────────────────────────────────────────────────────────────────────
# 3) Safe “conclusion only” editor
# ─────────────────────────────────────────────────────────────────────────────
def _edit_theorem_conclusions(lean_text: str, edit_fn) -> str:
    """
    For each 'theorem ... :=', apply edit_fn ONLY to the conclusion between
    the final top-level ':' and the ':='.
    """
    out: list[str] = []
    i = 0
    theorempat = re.compile(r"\btheorem\b")
    assign_pat = re.compile(r":=")

    while True:
        m = theorempat.search(lean_text, i)
        if not m:
            out.append(lean_text[i:])
            break

        out.append(lean_text[i:m.start()])
        j = m.start()

        a = assign_pat.search(lean_text, j)
        if not a:
            out.append(lean_text[j:])
            break
        k_assign = a.start()

        # find last ':' at paren-depth 0 before ':='
        depth, k_colon = 0, -1
        p = j
        while p < k_assign:
            c = lean_text[p]
            if c == "(":
                depth += 1
            elif c == ")":
                depth = max(0, depth - 1)
            elif c == ":" and depth == 0:
                k_colon = p
            p += 1

        if k_colon == -1:
            out.append(lean_text[j:k_assign])
            i = k_assign
            continue

        header = lean_text[j:k_colon]
        conclusion = lean_text[k_colon + 1 : k_assign]
        fixed = edit_fn(conclusion)

        out.append(header)
        out.append(": ")
        out.append(fixed.strip())
        i = k_assign  # keep ':=' and beyond for next round

    return "".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# 4) Conclusion passes: fractions → ℚ; then integers; then invariant coercions
# ─────────────────────────────────────────────────────────────────────────────
# Fractions: a/b -> (a/b : ℚ)
_FRAC = re.compile(
    r"""
    (?<![\w.])          # not preceded by name/num/dot
    (-?\d+)             # numerator
    \s*/\s*
    (\d+)               # denominator
    (?!\s*:\s*ℚ)        # not already cast
    """,
    re.VERBOSE,
)
def _cast_fractions_to_q(expr: str) -> str:
    return _FRAC.sub(lambda m: f"({m.group(1)}/{m.group(2)} : ℚ)", expr)

def fix_fractions_in_conclusions(lean_text: str) -> str:
    return _edit_theorem_conclusions(lean_text, _cast_fractions_to_q)

# Integers: stand-alone integer literal (not numerator of a/b), typed to ambient
_INT_TYPE = r"[\wℕℤℚℝα-ωΑ-Ω]+"
_INT = re.compile(
    r"""
    (?<![\w.])          # not preceded by name/num/dot
    (-?\s*\d+)          # integer (allow '-  3')
    (?!\s*/\s*\d)       # not a fraction numerator
    (?=(?:\s|[(){}\[\]+\-*^<>=≠≤≥,:]|$))  # next is a delimiter (prevents '12'→'(1)2')
    (?!\s*:\s*%s)       # not already ':\ TYPE'
    """
    % _INT_TYPE,
    re.VERBOSE,
)

def _choose_target_type(expr: str) -> str:
    if "ℚ" in expr:
        return "ℚ"
    if re.search(r"(?<![\w.])-\s*\d+(?=(?:\s|[(){}\[\]+\-*^<>=≠≤≥,:]|$))", expr):
        return "ℤ"
    return "ℕ"

def _cast_all_ints(expr: str, *, to_type: str) -> str:
    def repl(m: re.Match) -> str:
        s = re.sub(r"-\s*(\d+)", r"-\1", m.group(1))  # '-  12' → '-12'
        # Skip exponents (immediate non-space left is '^')
        j = m.start() - 1
        while j >= 0 and expr[j].isspace():
            j -= 1
        if j >= 0 and expr[j] == "^":
            return m.group(0)
        return f"({s} : {to_type})"
    return _INT.sub(repl, expr)

def _normalize_rel_spacing(expr: str) -> str:
    return re.sub(r"(≥|≤|=|<|>|≠)\(", r"\1 (", expr)

def _cast_invariants(expr: str, func_names: Iterable[str], *, to_type: str) -> str:
    """Wrap 'fn G' as '(fn G : to_type)' when to_type ≠ ℕ."""
    if to_type == "ℕ":
        return expr
    for fn in sorted(func_names, key=len, reverse=True):
        expr = re.sub(
            rf"(?<![\w.]){re.escape(fn)}\s+G(?!\s*:\s*(?:ℕ|ℤ|ℚ))",
            rf"({fn} G : {to_type})",
            expr,
        )
    return expr

def retarget_and_cast_conclusions(lean_text: str, func_names: Iterable[str]) -> str:
    def edit(expr: str) -> str:
        expr = _normalize_rel_spacing(expr)
        tgt = _choose_target_type(expr)
        expr = _cast_all_ints(expr, to_type=tgt)
        expr = _cast_invariants(expr, func_names, to_type=tgt)
        return expr
    return _edit_theorem_conclusions(lean_text, edit)


# ─────────────────────────────────────────────────────────────────────────────
# 5) Utility: ensure '... G' forms inside the conclusion before typing
# ─────────────────────────────────────────────────────────────────────────────
def _push_paren_G(expr: str, func_names: Iterable[str]) -> str:
    """Turn '... <fn>) G ...' → '... <fn> G) ...' for each fn in func_names."""
    for fn in func_names:
        expr = re.sub(rf"\b({re.escape(fn)})\s*\)\s*G\b", rf"\1 G)", expr)
    return expr

def _ensure_G(expr: str, func_names: Iterable[str]) -> str:
    """Ensure each listed function token is followed by ' G' unless already."""
    for fn in func_names:
        expr = re.sub(rf"\b({re.escape(fn)})\b(?!\s*G\b|\s*\()", rf"\1 G", expr)
    return expr

def _prep_conclusion_invariants(lean_text: str, func_names: Iterable[str]) -> str:
    # Operate only inside the conclusion slice
    def edit(expr: str) -> str:
        expr = _push_paren_G(expr, func_names)
        expr = _ensure_G(expr, func_names)
        return expr
    return _edit_theorem_conclusions(lean_text, edit)


# ─────────────────────────────────────────────────────────────────────────────
# 6) Public API: end-to-end translator for NECESSARY conjectures
# ─────────────────────────────────────────────────────────────────────────────
def necessary_conjecture_to_lean(
    conjectures: List[Conjecture],
    func_names: Iterable[str],
    *,
    name_prefix: str = "TxGraffitiBench",
) -> List[str]:
    """
    Render a batch of necessary conjectures (hypothesis ⇒ numeric inequality)
    into Lean theorems, fully typed.

    Parameters
    ----------
    conjectures : list[Conjecture]
        TxGraffiti conjectures whose conclusions are `Inequality`s.
    func_names : iterable[str]
        Names of invariants/properties that must appear as '<fn> G' in Lean.
        e.g., ["independence_number","matching_number","zero_forcing_number",...]
    name_prefix : str
        Prefix for theorem names (theorems are numbered 1..n).

    Returns
    -------
    list[str] : Lean theorems with typed conclusions.
    """
    outs: list[str] = []
    for i, conj in enumerate(conjectures, 1):
        t = conjecture_to_lean4(conj, f"{name_prefix}_{i}")
        t = _fix_header_and_hyps(t, func_names)         # add connected/order if missing
        t = _prep_conclusion_invariants(t, func_names)  # ensure '... G' in conclusion
        t = fix_fractions_in_conclusions(t)             # (a/b) -> (a/b : ℚ)
        t = retarget_and_cast_conclusions(t, func_names)# ints + invariant coercions
        outs.append(t)
    return outs


from fractions import Fraction
from typing import Iterable, List, Dict, Tuple

# ---- configuration -----------------------------------------------------------

# Map invariant/property tokens (as they appear in __repr__) to Lean identifiers.
LEAN_INV = {
    'order': 'order',
    'size': 'size',
    'maximum_degree': 'maximum_degree',
    'minimum_degree': 'minimum_degree',
    'diameter': 'diameter',
    'radius': 'radius',
    'clique_number': 'clique_number',
    'chromatic_number'  : 'chromatic_number',
    'independence_number': 'independence_number',
    'vertex_cover_number': 'vertex_cover_number',
    'matching_number': 'matching_number',
    'triameter': 'triameter',
    'slater': 'slater',
    'annihilation_number': 'annihilation_number',
    'residue': 'residue',
    'harmonic_index': 'harmonic_index',
    'domination_number': 'domination_number',
    'total_domination_number': 'total_domination_number',
    'independent_domination_number': 'independent_domination_number',
    'min_maximal_matching_number': 'min_maximal_matching_number',
    'spectral_radius': 'spectral_radius',
    'largest_laplacian_eigenvalue': 'largest_laplacian_eigenvalue',
    'second_largest_adjacency_eigenvalue': 'second_largest_adjacency_eigenvalue',
    'zero_forcing_number': 'zero_forcing_number',
}

# NEW: the native codomain of each invariant. Adjust as needed.
# Defaults to ℕ-like counts unless flagged.
LEAN_TYPE: Dict[str, str] = {
    # naturals (counts)
    'order': 'N', 'size': 'N', 'maximum_degree': 'N', 'minimum_degree': 'N',
    'diameter': 'N', 'radius': 'N', 'clique_number': 'N', 'chromatic_number': 'N',
    'independence_number': 'N', 'vertex_cover_number': 'N', 'matching_number': 'N',
    'triameter': 'N', 'slater': 'N', 'annihilation_number': 'N', 'residue': 'N',
    'domination_number': 'N', 'total_domination_number': 'N',
    'independent_domination_number': 'N', 'min_maximal_matching_number': 'N',
    'zero_forcing_number': 'N',
    # reals
    'harmonic_index': 'Q',
    'spectral_radius': 'R',
    'largest_laplacian_eigenvalue': 'R',
    'second_largest_adjacency_eigenvalue': 'R',
    # if you ever introduce ratios like "ratio_alpha", set them to 'Q' or 'R'
    # 'ratio_alpha': 'Q',
}

LEAN_PROP = {
    'connected': 'connected',
    'bipartite': 'bipartite',
    'chordal': 'chordal',
    'cubic': 'cubic',
    'eulerian': 'eulerian',
    'planar': 'planar',
    'regular': 'regular',
    'subcubic': 'subcubic',
    'tree': 'tree',
    'K_4_free': 'K_4_free',
    'triangle_free': 'triangle_free',
    'claw_free': 'claw_free',
    'cograph': 'cograph',
}

# ---- tiny parser utilities ---------------------------------------------------

def _strip_outer_parens(s: str) -> str:
    s = s.strip()
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        ok = True
        for i, ch in enumerate(s):
            if ch == "(": depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0: ok = False; break
            if i < len(s)-1 and depth == 0:
                ok = False
        if ok and depth == 0:
            s = s[1:-1].strip()
        else:
            break
    return s

def _tokenize_sum(expr: str) -> List[Tuple[int, str]]:
    e = _strip_outer_parens(expr.replace(" ", ""))
    e = e.replace("(", "").replace(")", "")
    if not e: return []
    if e[0] not in "+-": e = "+" + e
    out: List[Tuple[int,str]] = []
    i = 0
    while i < len(e):
        sign = +1
        if e[i] == "+": sign = +1; i += 1
        elif e[i] == "-": sign = -1; i += 1
        j = i
        while j < len(e) and e[j] not in "+-": j += 1
        term = e[i:j]
        if term: out.append((sign, term))
        i = j
    return out

def _parse_linear_comb(expr: str, allowed_vars: Dict[str,str]) -> Tuple[Dict[str, Fraction], Fraction]:
    coeffs: Dict[str, Fraction] = {}
    const = Fraction(0)
    for sign, raw in _tokenize_sum(expr):
        if "*" in raw:
            coef_str, var = raw.split("*", 1)
            coef = Fraction(coef_str)
            var = var.strip()
            if var not in allowed_vars:
                raise ValueError(f"Unknown variable '{var}' in term '{raw}'")
            coeffs[var] = coeffs.get(var, Fraction(0)) + sign * coef
        else:
            if raw in allowed_vars:
                coeffs[raw] = coeffs.get(raw, Fraction(0)) + sign * Fraction(1)
            else:
                const += sign * Fraction(raw)
    return coeffs, const

def _lcm(a: int, b: int) -> int:
    from math import gcd
    return abs(a*b) // gcd(a, b) if a and b else abs(a or b)

def _denom_lcm(fracs) -> int:
    d = 1
    for f in fracs: d = _lcm(d, f.denominator)
    return d or 1

def _detect_ineq(lhs_rhs: str) -> Tuple[str, str, str]:
    for op in [">=", "<=", "==", "=", ">", "<"]:
        parts = lhs_rhs.split(op)
        if len(parts) == 2:
            return parts[0].strip(), op, parts[1].strip()
    raise ValueError("No inequality/equality operator found.")

def _flip_op(op: str) -> str:
    return {">=":"<=", "<=":">=", ">":"<", "<":">", "=":"=", "==":"="}[op]

# ---- NEW: type lattice + casts ----------------------------------------------

_ORDER = {'N': 0, 'Z': 1, 'Q': 2, 'R': 3}
_ALPHA = {'N': 'ℕ', 'Z': 'ℤ', 'Q': 'ℚ', 'R': 'ℝ'}

def _join_types(types: Iterable[str]) -> str:
    lvl = max((_ORDER[t] for t in types), default=1)
    for k,v in _ORDER.items():
        if v == lvl: return k
    return 'Z'

def _var_type(tok: str) -> str:
    return LEAN_TYPE.get(tok, 'N')

def _alpha_str(t: str) -> str:
    return _ALPHA[t]

def _lean_lit(n: int, target: str) -> str:
    # literal n cast into α
    return f"({n} : {_alpha_str(target)})"

def _lean_var(var: str, target: str) -> str:
    # (invariant G) cast/annotated into α
    return f"({LEAN_INV[var]} G : {_alpha_str(target)})"

def conjectures_to_lean(
    conjs: Iterable[object],
    *,
    name_prefix: str = "Conj"
) -> List[str]:
    """
    Convert repr strings like
        "<Conj (chordal) → (independence_number >= 5)>"
        "<Conj (tree) → (independence_number = 2 + residue)>"
        "<Conj (planar) → (independence_number <= (-1/2 * total_domination_number) + 7)>"
        "<Conj (subcubic) → (7 + residue <= independence_number)>"
    into Lean theorems with standard hypotheses and clean types.

    Pipeline per conjecture:
      1) Parse hypothesis 'prop' and inequality 'lhs op rhs'.
      2) If the invariant is on the RHS and not on LHS, swap sides and flip op.
      3) Parse RHS as a linear combination over allowed invariants with a rational constant.
      4) Determine a common α ∈ {ℕ, ℤ, ℚ, ℝ} via the type join of LHS and all RHS vars.
      5) Multiply both sides by positive integer L = lcm(denominators) to clear fractions.
      6) **Normalize signs**: move all negative RHS terms (including negative constant) to LHS.
         After this step, the RHS has only nonnegative coefficients (and possibly 0).
      7) Render both sides in Lean with casts to α, preserving the original inequality operator.

    Note:
      - Because we only *add the same nonnegative expression to both sides* in step (6),
        the inequality direction is preserved (no flipping after normalization).
      - When α = ℕ, RHS being a sum of nonnegative terms avoids writing negative naturals.
    """
    results: List[str] = []

    for i, obj in enumerate(conjs, start=1):
        # ---- unwrap and split on hypothesis/implication ----
        s = repr(obj).strip()
        try:
            inner = s[s.index("<Conj ") + 6 : s.rindex(">")].strip()
        except Exception:
            raise ValueError(f"Unrecognized repr format: {s}")

        try:
            hyp_part, ineq_part = inner.split("→")
        except ValueError:
            raise ValueError(f"Missing '→' in: {s}")

        prop = _strip_outer_parens(hyp_part).strip()
        if prop not in LEAN_PROP:
            raise ValueError(f"Unknown property '{prop}' in: {s}")

        # ---- detect inequality ----
        ineq_body = _strip_outer_parens(ineq_part).strip()
        lhs_raw, op, rhs_raw = _detect_ineq(ineq_body)
        lhs_raw, rhs_raw = _strip_outer_parens(lhs_raw), _strip_outer_parens(rhs_raw)

        # ---- if bare invariant is on RHS, swap sides ----
        def _is_bare_invariant(tok: str) -> bool:
            return tok.strip() in LEAN_INV

        if not _is_bare_invariant(lhs_raw) and _is_bare_invariant(rhs_raw):
            lhs_raw, rhs_raw, op = rhs_raw, lhs_raw, _flip_op(op)

        if lhs_raw not in LEAN_INV:
            raise ValueError(f"LHS must be an invariant name; got '{lhs_raw}' in: {s}")

        # ---- parse RHS linear combo as rationals ----
        coeffs, const = _parse_linear_comb(rhs_raw, LEAN_INV)  # Dict[var] -> Fraction, const -> Fraction

        # ---- choose α by joining native types (LHS var + RHS vars) ----
        rhs_var_types = [_var_type(v) for v in coeffs.keys()]
        target = _join_types([_var_type(lhs_raw)] + rhs_var_types)
        alpha = _alpha_str(target)

        # ---- clear denominators with positive L ----
        denoms = [const] + list(coeffs.values()) + [Fraction(1)]
        L = _denom_lcm(denoms)  # positive integer

        # Left side base (possibly scaled by L)
        lhs_base = (
            _lean_var(lhs_raw, target)
            if L == 1 else
            f"{_lean_lit(L, target)} * {_lean_var(lhs_raw, target)}"
        )

        # ---- turn RHS rationals into integers (after * L) ----
        int_coeffs: Dict[str, int] = {}
        for var, q in coeffs.items():
            k = int(q * L)
            if k != 0:
                int_coeffs[var] = int_coeffs.get(var, 0) + k
        k0 = int(const * L)  # integer constant on RHS

        # ---- SIGN NORMALIZATION: move all negatives on RHS to LHS ----
        # Split variable terms by sign
        pos_coeffs: Dict[str, int] = {v: k for v, k in int_coeffs.items() if k > 0}
        neg_coeffs: Dict[str, int] = {v: -k for v, k in int_coeffs.items() if k < 0}  # magnitudes > 0

        rhs_const_pos = k0 if k0 > 0 else 0
        lhs_const_from_rhs = -k0 if k0 < 0 else 0  # if RHS const was negative, move its magnitude to LHS

        # Build LHS = lhs_base + (moved negative var terms) + (moved negative const)
        lhs_terms: List[str] = [lhs_base]
        for var, mag in neg_coeffs.items():  # mag > 0
            if mag == 1:
                lhs_terms.append(_lean_var(var, target))
            else:
                lhs_terms.append(f"{_lean_lit(mag, target)} * {_lean_var(var, target)}")
        if lhs_const_from_rhs > 0:
            lhs_terms.append(_lean_lit(lhs_const_from_rhs, target))
        lhs_lean = " + ".join(lhs_terms)

        # Build RHS = (only nonnegative var terms) + (nonnegative const); ensure at least 0 literal
        rhs_terms: List[str] = []
        for var, mag in pos_coeffs.items():  # mag > 0
            if mag == 1:
                rhs_terms.append(_lean_var(var, target))
            else:
                rhs_terms.append(f"{_lean_lit(mag, target)} * {_lean_var(var, target)}")
        if rhs_const_pos > 0 or not rhs_terms:
            rhs_terms.append(_lean_lit(rhs_const_pos, target))
        rhs_lean = " + ".join(rhs_terms)

        # ---- hypotheses block ----
        hyp_lines = [
            "    (h_conn : connected G)",
            "    (h_ord  : order G ≥ (2 : ℕ))",
        ]
        if prop != "connected":
            hyp_lines.append(f"    (h_{prop} : {LEAN_PROP[prop]} G)")
        hyps_block = "\n".join(hyp_lines)

        # ---- render theorem ----
        op_lean = {"==":"=", "=":"=", ">=":"≥", "<=":"≤", ">":">", "<":"<"}[op]
        thm_name = f"{name_prefix}_{i}"

        theorem = (
f"theorem {thm_name} (G : SimpleGraph V)\n"
f"{hyps_block} : {lhs_lean} {op_lean} {rhs_lean} :=\n"
"sorry\n"
        )
        results.append(theorem)

    return results


# # ---------- main ----------
# def conjectures_to_lean(
#     conjs: Iterable[object],
#     *,
#     name_prefix: str = "Conj"
# ) -> List[str]:
#     """
#     Accepts repr strings like:
#       "<Conj (chordal) → (independence_number >= 5)>"
#       "<Conj (tree) → (independence_number = 2 + residue)>"
#       "<Conj (planar) → (independence_number <= (-1/2 * total_domination_number) + 7)>"
#       "<Conj (subcubic) → (7 + residue <= independence_number)>"
#     Emits Lean theorems with hypotheses (connected G) and (order G ≥ 2).
#     Handles >=, <=, =/==, >, <; auto-flips sides; and UPCASTS to a common α ∈ {ℕ, ℤ, ℚ, ℝ}.
#     Denominators are cleared by multiplying both sides by (L : α), L > 0.
#     """
#     results: List[str] = []
#     for i, obj in enumerate(conjs, start=1):
#         s = repr(obj).strip()
#         try:
#             inner = s[s.index("<Conj ") + 6 : s.rindex(">")].strip()
#         except Exception:
#             raise ValueError(f"Unrecognized repr format: {s}")

#         try:
#             hyp_part, ineq_part = inner.split("→")
#         except ValueError:
#             raise ValueError(f"Missing '→' in: {s}")

#         prop = _strip_outer_parens(hyp_part).strip()
#         if prop not in LEAN_PROP:
#             raise ValueError(f"Unknown property '{prop}' in: {s}")

#         ineq_body = _strip_outer_parens(ineq_part).strip()
#         lhs_raw, op, rhs_raw = _detect_ineq(ineq_body)
#         lhs_raw, rhs_raw = _strip_outer_parens(lhs_raw), _strip_outer_parens(rhs_raw)

#         # If invariant is on the right (e.g., "… <= independence_number"), swap sides.
#         def _is_bare_invariant(tok: str) -> bool:
#             return tok.strip() in LEAN_INV

#         if not _is_bare_invariant(lhs_raw) and _is_bare_invariant(rhs_raw):
#             lhs_raw, rhs_raw, op = rhs_raw, lhs_raw, _flip_op(op)

#         if lhs_raw not in LEAN_INV:
#             raise ValueError(f"LHS must be an invariant name; got '{lhs_raw}' in: {s}")

#         # Parse RHS linear combo (coefficients as Fractions)
#         coeffs, const = _parse_linear_comb(rhs_raw, LEAN_INV)

#         # Compute the common target α by joining native types
#         rhs_var_types = [_var_type(v) for v in coeffs.keys()]
#         target = _join_types([_var_type(lhs_raw)] + rhs_var_types)
#         alpha = _alpha_str(target)

#         # Denominator LCM across RHS and implicit LHS coefficient 1
#         denoms = [const] + list(coeffs.values()) + [Fraction(1)]
#         L = _denom_lcm(denoms)  # positive

#         # Scale sides by L and render in α with integer literals cast into α
#         lhs_lean = (
#             _lean_var(lhs_raw, target)
#             if L == 1 else
#             f"{_lean_lit(L, target)} * {_lean_var(lhs_raw, target)}"
#         )

#         rhs_terms: List[str] = []
#         for var, q in coeffs.items():
#             k = int(q * L)
#             if k == 0: continue
#             if k == 1:
#                 rhs_terms.append(f"{_lean_var(var, target)}")
#             elif k == -1:
#                 rhs_terms.append(f"-{_lean_var(var, target)}")
#             else:
#                 rhs_terms.append(f"{_lean_lit(k, target)} * {_lean_var(var, target)}")
#         k0 = int(const * L)
#         if k0 != 0 or not rhs_terms:
#             rhs_terms.append(_lean_lit(k0, target))
#         rhs_lean = " + ".join(rhs_terms).replace("+ -", "- ")

#         # Hypotheses
#         hyp_lines = [
#             "    (h_conn : connected G)",
#             "    (h_ord  : order G ≥ (2 : ℕ))",
#         ]
#         if prop != "connected":
#             hyp_lines.append(f"    (h_{prop} : {LEAN_PROP[prop]} G)")
#         hyps_block = "\n".join(hyp_lines)

#         op_lean = {"==":"=", "=":"=", ">=":"≥", "<=":"≤", ">":">", "<":"<"}[op]
#         thm_name = f"{name_prefix}_{i}"

#         theorem = (
# f"theorem {thm_name} (G : SimpleGraph V)\n"
# f"{hyps_block} : {lhs_lean} {op_lean} {rhs_lean} :=\n"
# "sorry\n"
#         )
#         results.append(theorem)

#     return results
