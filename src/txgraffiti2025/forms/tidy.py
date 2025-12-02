# src/txgraffiti2025/forms/tidy.py
from __future__ import annotations
import re
from typing import Dict, Optional

# --- core helpers -------------------------------------------------------------

def _humanize_token(token: str, *, unicode: bool = True, alias_map: Optional[Dict[str, str]] = None) -> str:
    """Turn a raw feature token into a human-friendly string, generically."""
    # allow external renaming without domain assumptions
    if alias_map and token in alias_map:
        return alias_map[token]

    # pair constructs
    m = re.fullmatch(r"(.+)__absdiff__(.+)", token)
    if m:
        a = _humanize_token(m.group(1), unicode=unicode, alias_map=alias_map)
        b = _humanize_token(m.group(2), unicode=unicode, alias_map=alias_map)
        return f"|{a} - {b}|"

    m = re.fullmatch(r"min__(.+)__(.+)", token)
    if m:
        a = _humanize_token(m.group(1), unicode=unicode, alias_map=alias_map)
        b = _humanize_token(m.group(2), unicode=unicode, alias_map=alias_map)
        return f"min({a}, {b})"

    m = re.fullmatch(r"max__(.+)__(.+)", token)
    if m:
        a = _humanize_token(m.group(1), unicode=unicode, alias_map=alias_map)
        b = _humanize_token(m.group(2), unicode=unicode, alias_map=alias_map)
        return f"max({a}, {b})"

    # unary transforms
    if token.endswith("__square"):
        base = _humanize_token(token[:-9], unicode=unicode, alias_map=alias_map)
        return f"{base}²" if unicode else f"{base}^2"

    if token.endswith("__sqrt"):
        base = _humanize_token(token[:-7], unicode=unicode, alias_map=alias_map)
        return f"√{base}" if unicode else f"sqrt({base})"

    if token.endswith("__log"):
        base = _humanize_token(token[:-6], unicode=unicode, alias_map=alias_map)
        return f"log {base}"

    # default: return as-is (or alias applied above)
    return token

def tidy_expression(expr_str: str, *, unicode: bool = True, alias_map: Optional[Dict[str, str]] = None) -> str:
    """
    Tidy a linear-ish expression string (RHS/LHS), *domain-agnostic*.
    - remove leading '0 +'
    - normalize '+/-' spacing
    - hide '1·x' and '-1·x'
    - humanize transformed/paired tokens
    """
    s = expr_str.strip()

    # nuke leading "0 + " chains and leading "+"
    s = re.sub(r"^(?:0\s*\+\s*)+", "", s)
    s = re.sub(r"^\+\s*", "", s)

    # find candidate tokens to humanize (names w/ letters/underscores, allowing our __patterns__)
    token_re = r"[A-Za-z][A-Za-z0-9_]*(?:__(?:absdiff__|sqrt|square|log)[A-Za-z0-9_]+)*(?:__[A-Za-z0-9_]+)*"
    tokens = sorted(set(re.findall(token_re, s)), key=len, reverse=True)

    for t in tokens:
        human = _humanize_token(t, unicode=unicode, alias_map=alias_map)
        s = re.sub(rf"\b{re.escape(t)}\b", human, s)

    # hide 1· and -1· coefficients
    s = re.sub(r"(?<![\w])1\s*·\s*", "", s)
    s = re.sub(r"(?<![\w])-1\s*·\s*", "- ", s)

    # clean '+ -' → '- '
    s = s.replace("+ -", "- ")

    # spacing around +/-
    s = re.sub(r"\s*\+\s*", " + ", s)
    s = re.sub(r"\s*-\s*", " - ", s)

    # compress whitespace
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def tidy_hypothesis(h_str: str) -> str:
    """
    Clean hypothesis text:
    - drop redundant outer parentheses
    - normalize '∧' spacing
    - collapse double parens around atoms
    """
    s = h_str.strip()
    # collapse ((A)) → (A)
    s = s.replace("((", "(").replace("))", ")")
    # `(A) ∧ (B)` → `A ∧ B`
    s = re.sub(r"\(\s*", "(", s)
    s = re.sub(r"\s*\)", ")", s)
    if s.startswith("(") and s.endswith(")"):
        # only strip if it looks like a top-level wrap
        s = s[1:-1]
    s = re.sub(r"\)\s*∧\s*\(", ") ∧ (", s)
    # remove single parens around words: (A) → A, but keep if contains '∧' or '∨'
    def drop_single_parens(m):
        inner = m.group(1)
        return inner if ("∧" not in inner and "∨" not in inner) else f"({inner})"
    s = re.sub(r"\(([^()]+)\)", drop_single_parens, s)
    return s

def tidy_conjecture(raw: str, *, arrow: str = "⇒", unicode: bool = True, alias_map: Optional[Dict[str, str]] = None) -> str:
    """
    Tidy a full conjecture line like:
      '((A) ∧ (B)) ⇒ y ≥ 0 + 1·x + 1/2·x__square'
    into a cleaner, domain-agnostic rendering.
    """
    parts = raw.split(arrow)
    if len(parts) != 2:
        return tidy_expression(raw, unicode=unicode, alias_map=alias_map)

    left, right = parts[0].strip(), parts[1].strip()
    left = tidy_hypothesis(left)
    right = tidy_expression(right, unicode=unicode, alias_map=alias_map)
    return f"{left} {arrow} {right}"
