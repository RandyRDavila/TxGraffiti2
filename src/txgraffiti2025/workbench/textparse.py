from __future__ import annotations

def strip_redundant_parens(s: str) -> str:
    """Peel *all* redundant outer parens that wrap the whole expression."""
    s = s.strip()
    while len(s) >= 2 and s[0] == '(' and s[-1] == ')':
        depth = 0
        encloses_all = True
        for i, ch in enumerate(s):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            if depth == 0 and i < len(s) - 1:
                encloses_all = False
                break
        if encloses_all:
            s = s[1:-1].strip()
        else:
            break
    return s

import re
from typing import List, Set
from txgraffiti2025.forms.predicates import Predicate

IFTHEN_RE = re.compile(r"^\[\s*(?P<hyp>.+?)\s*\]\s*::\s*(?P<phi>.+)$")

def strip_outer_parens_once(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        inner = s[1:-1].strip()
        return inner if (inner.startswith("(") and inner.endswith(")")) else f"({inner})"
    return s

def split_top_level_conj(s: str) -> List[str]:
    s = strip_redundant_parens(s)
    parts, buf, depth = [], [], 0
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch == "(":
            depth += 1; buf.append(ch); i += 1; continue
        if ch == ")":
            if depth > 0: depth -= 1
            buf.append(ch); i += 1; continue
        if depth == 0:
            if ch == "∧":
                part = "".join(buf).strip()
                if part: parts.append(part)
                buf = []; i += 1; continue
            if s.startswith(r"\land", i):
                part = "".join(buf).strip()
                if part: parts.append(part)
                buf = []; i += len(r"\land"); continue
        buf.append(ch); i += 1
    tail = "".join(buf).strip()
    if tail: parts.append(tail)
    return parts or [s]

def canon_atom(s: str) -> str:
    s = " ".join(strip_outer_parens_once(s).split())
    if not (s.startswith("(") and s.endswith(")")):
        s = f"({s})"
    return s

def atoms_for_pred(p: Predicate) -> Set[str]:
    hyp_obj = getattr(p, "_derived_hypothesis", None)
    phi_obj = getattr(p, "_derived_conclusion", None)
    if hyp_obj is not None and phi_obj is not None:
        hyp_txt = getattr(hyp_obj, "pretty", lambda: repr(hyp_obj))()
        phi_txt = getattr(phi_obj, "pretty", lambda: repr(phi_obj))()
        return {canon_atom(x) for x in split_top_level_conj(hyp_txt)} | {canon_atom(phi_txt)}
    name = getattr(p, "name", None)
    if name:
        m = IFTHEN_RE.match(name)
        if m:
            hyp_txt = m.group("hyp").strip()
            phi_txt = m.group("phi").strip()
            return {canon_atom(x) for x in split_top_level_conj(hyp_txt)} | {canon_atom(phi_txt)}
    s = (p.pretty() if hasattr(p, "pretty") else repr(p)).strip()
    return {canon_atom(x) for x in split_top_level_conj(s)}

def predicate_to_conjunction(p: Predicate, *, ascii_ops: bool = False) -> str:
    land = r"\land" if ascii_ops else "∧"
    raw = getattr(p, "name", None)
    if raw:
        m = IFTHEN_RE.match(raw)
        if m:
            hyp = strip_outer_parens_once(m.group("hyp"))
            phi = m.group("phi").strip()
            if not (hyp.startswith("(") and hyp.endswith(")")): hyp = f"({hyp})"
            if not (phi.startswith("(") and phi.endswith(")")): phi = f"({phi})"
            return f"({hyp} {land} {phi})"
    s = p.pretty() if hasattr(p, "pretty") else repr(p)
    s = strip_outer_parens_once(s)
    if not (s.startswith("(") and s.endswith(")")): s = f"({s})"
    return s

def predicate_to_if_then(p: Predicate) -> str:
    name = getattr(p, "name", None) or (p.pretty() if hasattr(p, "pretty") else repr(p))
    m = IFTHEN_RE.match(name)
    if not m: return name
    hyp = strip_outer_parens_once(m.group("hyp"))
    phi = m.group("phi").strip()
    if not (hyp.startswith("(") and hyp.endswith(")")): hyp = f"({hyp})"
    return f"If {hyp} ⇒ {phi}"

def pretty_class_relations_conj(title, eqs, incs, df, ascii_ops=False, show_violations=False):
    arrow_eq  = "<=>" if ascii_ops else "⇔"
    arrow_inc = "<="  if ascii_ops else "⊆"
    print(f"=== {title} ===")
    if eqs:
        print("-- Equivalences --")
        for i, e in enumerate(eqs, 1):
            try:
                print(f"{i:2d}. {e.A.name} {arrow_eq} {e.B.name}")
            except Exception:
                print(f"{i:2d}. {e}")
    if incs:
        print("-- Inclusions --")
        for i, inc in enumerate(incs, 1):
            try:
                print(f"{i:2d}. {inc.A.name} {arrow_inc} {inc.B.name}")
            except Exception:
                print(f"{i:2d}. {inc}")
    if not eqs and not incs:
        print("(none)")
