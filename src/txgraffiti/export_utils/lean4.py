"""
Module for exporting TxGraffiti conjectures to Lean 4 syntax.

This module provides functions to convert symbolic conjectures into
Lean-compatible theorems or propositions. It includes Lean-friendly
symbol mappings, automatic variable mappings, and translators from
Conjecture objects to Lean 4 strings.
"""

from __future__ import annotations
import re
from typing import Iterable
from collections.abc import Mapping
import pandas as pd
from typing import Iterable, List


from txgraffiti.logic import *

__all__ = [
    "conjecture_to_lean4",
    # "auto_var_map",
    "LEAN_SYMBOLS",
    "LEAN_SYMBOLS",
    "necessary_conjecture_to_lean",
]

# ---------------------------------------------------------------------------
# 1. Lean-friendly replacements for operators & symbols
# ---------------------------------------------------------------------------
LEAN_SYMBOLS: Mapping[str, str] = {
    "∧": "∧",
    "∨": "∨",
    "¬": "¬",
    "→": "→",
    "≥": "≥",
    "<=": "≤",
    ">=": "≥",
    "==": "=",
    "=": "=",
    "!=": "≠",
    "<": "<",
    ">": ">",
    "/": "/",
    "**": "^",
}

# ---------------------------------------------------------------------------
# 2. Automatic variable-map builder
# ---------------------------------------------------------------------------
def auto_var_map(df: pd.DataFrame, *, skip: tuple[str, ...] = ("name",)) -> dict[str, str]:
    """
    Build a variable mapping for Lean 4 translation.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe from which to extract column names.
    skip : tuple of str, optional
        Column names to skip in the output (default is ('name',)).

    Returns
    -------
    dict of str to str
        A mapping from column names to Lean variable expressions.
    """
    return {c: f"{c} G" for c in df.columns if c not in skip}


# ---------------------------------------------------------------------------
# 3. The main translator
# ---------------------------------------------------------------------------
def _translate(expr: str, var_map: Mapping[str, str]) -> str:
    # 3a. longest variable names first so 'order' doesn't clobber 'total_order'
    for var in sorted(var_map, key=len, reverse=True):
        expr = re.sub(rf"\b{re.escape(var)}\b", var_map[var], expr)

    # 3b. symbolic replacements (do ** after >= / <= replacements)
    for sym, lean_sym in LEAN_SYMBOLS.items():
        expr = expr.replace(sym, lean_sym)

    # tidy whitespace
    expr = re.sub(r"\s+", " ", expr).strip()
    return expr

def conjecture_to_lean4(
    conj: Conjecture,
    name: str,
    object_symbol: str = "G",
    object_decl: str = "SimpleGraph V"
) -> str:
    """
    Convert a Conjecture object into a Lean 4 theorem with explicit hypotheses.

    Parameters
    ----------
    conj : Conjecture
        The conjecture object to convert.
    name : str
        Name of the theorem in Lean.
    object_symbol : str, optional
        Symbol representing the graph (default is 'G').
    object_decl : str, optional
        Lean type declaration for the object (default is 'SimpleGraph V').

    Returns
    -------
    str
        A Lean 4 theorem string with bound hypotheses and a conclusion.
    """

    # 1) extract hypothesis Predicates
    terms = getattr(conj.hypothesis, "_and_terms", [conj.hypothesis])
    binds = []
    for idx, p in enumerate(terms, start=1):
        lean_pred = p.name
        binds.append(f"(h{idx} : {lean_pred} {object_symbol})")

    # 2) extract conclusion
    ineq = conj.conclusion
    lhs, op, rhs = ineq.lhs.name, ineq.op, ineq.rhs.name
    lean_rel = {"<=":"≤", "<":"<", ">=":"≥", ">":">", "==":"=", "!=":"≠"}[op]

    # 3) assemble
    bind_str = "\n    ".join(binds)
    return (
        f"theorem {name} ({object_symbol} : {object_decl})\n"
        f"    {bind_str} : {lhs} {object_symbol} {lean_rel} {rhs} {object_symbol} :=\n"
        f"sorry \n"
    )



# Recognize existing required hypotheses (supports >=, ≥, <=, ≤)
RE_CONNECTED = re.compile(r"\(\s*h\d+\s*:\s*connected\s+G\s*\)")
RE_ORDER     = re.compile(r"\(\s*h\d+\s*:\s*order\s+G\s*(?:>=|≥|<=|≤)\s*2\s*\)")

def _next_h_index(hyp_block: str) -> int:
    idxs = [int(m.group(1)) for m in re.finditer(r"\(\s*h(\d+)\s*:", hyp_block)]
    return (max(idxs) + 1) if idxs else 1

def _insert_missing_hypotheses(hdr_and_g: str, hyp_block: str) -> str:
    need_connected = (RE_CONNECTED.search(hyp_block) is None)
    need_order     = (RE_ORDER.search(hyp_block) is None)
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
        pieces.append(f"\n{indent}(h{n} : order G ≥ 2) ")  # change to ≥ if you prefer
    return hdr_and_g + "".join(pieces)

def fix_lean_conjectures(text: str, func_names: Iterable[str]) -> str:
    """
    Fix Lean4 conjecture theorems by:
      • Pushing ') G' onto the function inside parentheses.
      • Ensuring each named invariant is written with ' G' (unless already ' G' or '(' call).
      • Removing ONLY the stray ') G' right before ':=' (keeps 'residue G :=', etc.).
      • Adding missing hypotheses: (connected G) and (order G ≥ 2).
    Note: Operator-agnostic w.r.t. '==', '≥'/'>=', '≤'/'<=' in the conclusion.
    """

    # A) Add missing hypotheses per theorem
    def add_hyps(match: re.Match) -> str:
        before = match.group("before")
        hyps   = match.group("hyps") or ""
        colon  = match.group("colon")
        expr   = match.group("expr")
        assign = match.group("assign")
        header_plus_hyps = _insert_missing_hypotheses(before, hyps)
        return f"{header_plus_hyps}{colon}{expr}{assign}"

    theorems_pat = re.compile(
        r"(?P<before>theorem\b.*?\(G\s*:\s*SimpleGraph\s+V\))"
        r"(?P<hyps>(?:.*?\(h\d+\s*:.*?\)\s*)*)"
        r"(?P<colon>:\s)"
        r"(?P<expr>.*?)"
        r"(?P<assign>\s*:=)",
        flags=re.DOTALL
    )
    text = theorems_pat.sub(add_hyps, text)

    # B) Expression-level fixes (between ':' and ':='), independent of comparator
    concl_pat = re.compile(r":\s*(?P<expr>.*?)\s*:=", flags=re.DOTALL)

    push_inside = [
        (re.compile(rf"\b({re.escape(fn)})\s*\)\s*G\b"), rf"\1 G)")
        for fn in func_names
    ]
    add_missing_G = [
        (re.compile(rf"\b({re.escape(fn)})\b(?!\s*G\b|\s*\()"), rf"\1 G")
        for fn in func_names
    ]

    def fix_expr(m: re.Match) -> str:
        expr = m.group("expr")
        for rx, repl in push_inside:
            expr = rx.sub(repl, expr)
        for rx, repl in add_missing_G:
            expr = rx.sub(repl, expr)
        return f": {expr} :="

    text = concl_pat.sub(fix_expr, text)

    # C) ONLY remove ') G' right before ':=' (don’t touch 'func G :=')
    text = re.sub(r"\)\s*G(\s*:=)", r")\1", text)

    return text

def cast_numeric_fractions_to_q(expr: str) -> str:
    """
    Inside a Lean expression, cast any integer-literal fraction a/b to (a/b : ℚ).
    Examples: 5/4 -> (5/4 : ℚ), -53/2 -> (-53/2 : ℚ), 1637/2403 -> (1637/2403 : ℚ)
    Skips if it's already followed by ': ℚ'.
    Only matches when BOTH numerator and denominator are integer literals.
    """
    # Match integer literal fraction possibly with leading minus on numerator.
    # Ensure it's not already followed by ': ℚ' (no double casting).
    frac_pat = re.compile(
        r"""
        (?<![\w.])          # not preceded by word char or dot (avoid part of name or float)
        (-?\d+)             # numerator (int, optional leading '-')
        \s*/\s*             # slash with optional spaces
        (\d+)               # denominator (positive int)
        (?!\s*:\s*ℚ)        # NOT already cast to ℚ
        (?!\s*[\w(])        # don't match if immediately followed by a name or '(' (rare false positives)
        """,
        re.VERBOSE
    )

    def repl(m: re.Match) -> str:
        num, den = m.group(1), m.group(2)
        return f"({num}/{den} : ℚ)"

    return frac_pat.sub(repl, expr)


def fix_fractions_in_conclusions(lean_text: str) -> str:
    """
    Find every conclusion segment ': <expr> :=' and cast numeric fractions inside <expr> to ℚ.
    Leaves everything else (hypotheses, headers, proofs) unchanged.
    """
    concl_pat = re.compile(r":\s*(?P<expr>.*?)\s*:=", flags=re.DOTALL)

    def _one(m: re.Match) -> str:
        expr = m.group("expr")
        expr_fixed = cast_numeric_fractions_to_q(expr)
        return f": {expr_fixed} :="

    return concl_pat.sub(_one, lean_text)

_INT_OR_FRAC_TYPE = r"[\wℕℤℚℝα-ωΑ-Ω]+"  # for robust '(: TYPE)' checks

def cast_all_int_literals(expr: str, *, to_type: str) -> str:
    """
    Wrap *all* standalone integer literals as (k : to_type), including positives and zero.
    Skips:
      • numerators/denominators in fractions (handled elsewhere as ℚ)
      • anything already typed '... : TYPE'
      • numerals used as exponents right after '^'
    """
    pat = re.compile(
        r"""
        (?<![\w.])          # not preceded by name/number/dot
        (-?\s*\d+)          # integer literal (allows '-   3' -> normalized to '-3')
        (?!\s*/\s*\d)       # not a fraction's numerator (a/b)
        (?!\s*:\s*%s)       # not already annotated ':\ TYPE'
        """ % _INT_OR_FRAC_TYPE,
        re.VERBOSE
    )

    def repl(m: re.Match) -> str:
        s = m.group(1)
        # normalize '-  12' -> '-12'
        s = re.sub(r"-\s*(\d+)", r"-\1", s)

        # Skip if this literal is used as an exponent: look left for the last non-space char
        i = m.start()
        j = i - 1
        while j >= 0 and expr[j].isspace():
            j -= 1
        if j >= 0 and expr[j] == "^":
            return m.group(0)  # leave exponent numerals alone (they live in ℕ)

        return f"({s} : {to_type})"

    return pat.sub(repl, expr)


def fix_integers_in_conclusions(lean_text: str) -> str:
    """
    In every conclusion ': <expr> :=':
      • If '<expr>' mentions 'ℚ' (e.g., due to 'a/b : ℚ'), cast ALL integer literals to ℚ.
      • Otherwise, cast ALL integer literals to ℤ.
    Fractions should already be handled by 'fix_fractions_in_conclusions'.
    """
    concl_pat = re.compile(r":\s*(?P<expr>.*?)\s*:=", flags=re.DOTALL)

    def _one(m: re.Match) -> str:
        expr = m.group("expr")
        tgt = "ℚ" if "ℚ" in expr else "ℤ"
        expr2 = cast_all_int_literals(expr, to_type=tgt)
        return f": {expr2} :="

    return concl_pat.sub(_one, lean_text)

def necessary_conjecture_to_lean(conjectures: list, keys: list, name="TxGraffitiBench") -> list:
    lean_conjectures = []
    for i, conj in enumerate(conjectures):
        conj = conjecture_to_lean4(conj, f"{name}_{i+1}")
        conj = fix_lean_conjectures(conj, keys)
        conj = fix_fractions_in_conclusions(conj)   # (a/b) -> (a/b : ℚ)
        conj = fix_integers_in_conclusions(conj)    #  -1, 0, 2, 17 -> (: ℤ/ℚ)
        lean_conjectures.append(conj)
    return lean_conjectures

# --- Core token handling ------------------------------------------------------

def _push_paren_G(expr: str, func_names: Iterable[str]) -> str:
    """Turn '... <func>) G ...' into '... <func> G) ...' for each func in func_names."""
    for fn in func_names:
        expr = re.sub(rf"\b({re.escape(fn)})\s*\)\s*G\b", rf"\1 G)", expr)
    return expr

def _ensure_G(expr: str, func_names: Iterable[str]) -> str:
    """Ensure each listed function/invariant in expr is followed by ' G' unless already."""
    for fn in func_names:
        expr = re.sub(rf"\b({re.escape(fn)})\b(?!\s*G\b|\s*\()", rf"\1 G", expr)
    return expr

def _normalize_comparator(s: str) -> str:
    """Map ASCII comparators to Lean-friendly forms (keep unicode if present)."""
    # keep ≤ ≥ if already there
    s = s.replace("==", "=")
    s = re.sub(r"(?<!<)<>", "≠", s)  # (rare) if you ever use <>
    s = s.replace("<=", "≤").replace(">=", "≥")
    return s

# --- Parser for a single '<Conj (...) → (...)>' line -------------------------

LINE_RE = re.compile(
    r"^\s*<\s*Conj\s*\(\s*(?P<cond>.+?)\s*\)\s*→\s*\(\s*(?P<prop>[A-Za-z0-9_]+)\s*\)\s*>$"
)

def parse_sufficient_line(line: str):
    m = LINE_RE.match(line)
    if not m:
        raise ValueError(f"Not a valid <Conj (...) → (... )> line: {line}")
    return m.group("cond"), m.group("prop")


# --- Public API ---------------------------------------------------------------

def sufficient_conjs_to_lean(
    lines: List[str],
    func_names: Iterable[str],
    start_index: int = 1,
    theorem_prefix: str = "TxSufficient",
) -> List[str]:
    """
    Convert sufficient conjecture lines like
        <Conj (independence_number < (1 + radius)) → (claw_free)>
    into Lean4 theorems that assume the inequality and conclude the property.

    Parameters
    ----------
    lines : list of str
        Each a single-line conjecture in the angle-bracket format shown above.
    func_names : iterable of str
        All invariant / function identifiers that should apply to ' G'.
        e.g. ['independence_number','radius','slater','size', ...].
    start_index : int
        Starting theorem index.
    theorem_prefix : str
        Base name for theorems, e.g. 'TxSufficient' -> 'theorem TxSufficient 1 ...'

    Returns
    -------
    list of str
        Lean4 theorem strings.
    """
    out = []
    k = start_index
    for raw in lines:
        cond, prop = parse_sufficient_line(raw)

        # Normalize comparator spelling first
        cond = _normalize_comparator(cond)

        # Push ') G' inside, then ensure each listed function token has ' G'
        cond = _push_paren_G(cond, func_names)
        cond = _ensure_G(cond, func_names)

        # Build theorem text
        thm = (
f"""theorem {theorem_prefix} {k} (G : SimpleGraph V)
    (h1 : connected G)
    (h2 : order G >= 2)
    (hcond : {cond}) :
    {prop} G := by
  sorry
"""
        )
        out.append(thm)
        k += 1
    return out
