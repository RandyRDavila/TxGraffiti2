# src/txgraffiti2025/workbench/class_relations.py

from __future__ import annotations
from typing import Iterable, List, Set, Tuple
import numpy as np
import pandas as pd

from txgraffiti2025.forms.predicates import Predicate
from txgraffiti2025.forms.class_relations import ClassInclusion, ClassEquivalence
from .textparse import atoms_for_pred, canon_atom
from .arrays import support as mask_support, same_mask

"""
Discovery of boolean class relations among predicates.

Given a collection of boolean predicates evaluated on a dataset, this module
infers:

- **Equivalences**:        (A) ⇔ (B) when masks are identical (optionally skipping
  trivial always-true/false cases and subset-by-syntax tautologies).
- **Inclusions**:          (A) ⇒ (B) when ``mask(A) ⊆ mask(B)``, with optional
  filtering to avoid shared-atom artifacts unless they belong to a supplied
  ambient set (e.g., foundational atoms you permit to overlap).

Design
------
- Masks are derived via ``Predicate.mask(df)`` and reindexed to ``df.index``.
- All masks are coerced to boolean arrays with missing values treated as ``False``.
- Deduplication is performed via each relation object’s ``.signature()``.

Examples
--------
>>> import pandas as pd
>>> from txgraffiti2025.forms.predicates import Predicate   # doctest: +SKIP
>>> from txgraffiti2025.workbench.class_relations import discover_class_relations
>>> df = pd.DataFrame({"x":[1,2,3], "y":[1,1,0]})
>>> A = Predicate(lambda df: df["x"] >= 2)     # doctest: +SKIP
>>> B = Predicate(lambda df: df["x"] >  1)     # doctest: +SKIP
>>> C = Predicate(lambda df: df["y"] == 1)     # doctest: +SKIP
>>> equivs, incls = discover_class_relations(df, predicates=[A,B,C])   # doctest: +SKIP
>>> len(equivs) >= 0 and len(incls) >= 1      # doctest: +SKIP
True
"""

def _pred_key(p: Predicate) -> str:
    """
    Stable string key for a predicate.

    Prefers ``p.pretty()`` if available; otherwise uses ``repr(p)``.

    Parameters
    ----------
    p : Predicate
        Predicate object.

    Returns
    -------
    str
        A stable string identifier for dictionary keys and caches.

    Notes
    -----
    The stability of this key affects cache hits only; it does not alter
    logical results. If your Predicates define a canonical ``pretty()``,
    prefer that for reproducibility across runs.
    """
    return p.pretty() if hasattr(p, "pretty") else repr(p)

def discover_class_relations(
    df: pd.DataFrame,
    *,
    predicates: Iterable[Predicate],
    min_support_A: int = 3,
    skip_trivial_equiv: bool = True,
    disallow_shared_atoms: bool = True,
    ambient_atoms: Set[str] | None = None,
) -> Tuple[List[ClassEquivalence], List[ClassInclusion]]:
    """
    Infer predicate **equivalences** and **inclusions** from masks on ``df``.

    This routine evaluates each predicate on ``df`` (after reindexing to
    ``df.index`` and coercing to boolean), then:

    1) **Equivalences (A ⇔ B):**
       - Declared when masks are exactly equal (``same_mask``).
       - Optionally skip trivial equivalences where masks are all-True or all-False
         (``skip_trivial_equiv=True``).
       - Optionally skip syntax-driven tautologies where the set of atoms in one
         predicate is a **strict** subset of the other (helps avoid “A ⇒ A ∧ B”
         style artifacts masquerading as equivalence).

    2) **Inclusions (A ⇒ B):**
       - Declared when ``mask(A) ⊆ mask(B)`` (i.e., ``np.any(mi & ~mj) == False``),
         and ``support(A) ≥ min_support_A``, and neither mask is degenerate as filtered
         below.
       - Skip inclusions into all-True masks (too weak to be interesting).
       - Skip when atoms(B) ⊆ atoms(A) (prevents trivial syntactic nesting).
       - Optionally disallow shared atoms between A and B unless those atoms are
         explicitly allowed via ``ambient_atoms`` (canonicalized with ``canon_atom``).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset on which predicates are evaluated.
    predicates : Iterable[Predicate]
        Collection of predicates to analyze.
    min_support_A : int, default=3
        Minimum number of rows satisfying A required to consider any A ⇒ B relation.
    skip_trivial_equiv : bool, default=True
        If ``True``, do not report equivalences whose mask is all-True or all-False.
    disallow_shared_atoms : bool, default=True
        If ``True``, exclude A ⇒ B when A and B share any atom **outside** of
        ``ambient_atoms`` (mitigates tautological statements tied to shared syntax).
    ambient_atoms : set of str or None, default=None
        Atoms allowed to be shared without disqualifying inclusion. Each atom is
        normalized with :func:`canon_atom` before comparison.

    Returns
    -------
    equivalences, inclusions : tuple[list[ClassEquivalence], list[ClassInclusion]]
        Deduplicated lists of discovered relations.

    Notes
    -----
    - **Atom sets.** ``atoms_for_pred`` extracts the set of atomic symbols appearing
      in a predicate (implementation-dependent). These are used to filter trivial
      equivalences and inclusions driven by syntax rather than data.
    - **Complexity.** The algorithm is O(k^2 · n) in the number of predicates ``k``
      and rows ``n`` (pairwise mask comparisons). It is typically fast for tens to
      hundreds of predicates; for thousands, consider blocking or hashing masks.
    - **Mask semantics.** All masks are aligned to ``df.index`` with missing values
      treated as False; dtype is coerced to boolean.

    Examples
    --------
    >>> import pandas as pd
    >>> from txgraffiti2025.forms.predicates import Predicate   # doctest: +SKIP
    >>> df = pd.DataFrame({"u":[0,1,2,3], "v":[0,1,1,1]})
    >>> A = Predicate(lambda df: df["u"] >= 1)    # doctest: +SKIP
    >>> B = Predicate(lambda df: df["u"] >  0)    # doctest: +SKIP
    >>> C = Predicate(lambda df: df["v"] == 1)    # doctest: +SKIP
    >>> equivs, incls = discover_class_relations(df, predicates=[A,B,C], min_support_A=2)  # doctest: +SKIP
    >>> isinstance(equivs, list) and isinstance(incls, list)   # doctest: +SKIP
    True
    """
    cand = list(predicates)

    # Caches
    mask_cache: dict[str, np.ndarray] = {}
    atoms_cache: dict[str, set[str]] = {}
    obj_cache: dict[str, Predicate] = {}

    for p in cand:
        k = _pred_key(p)
        s = p.mask(df).reindex(df.index, fill_value=False)
        m = s.fillna(False).astype(bool, copy=False).to_numpy()
        mask_cache[k] = m
        obj_cache[k] = p
        atoms_cache[k] = atoms_for_pred(p)

    keys = list(mask_cache.keys())
    n = len(df)
    all_true = np.ones(n, dtype=bool)
    all_false = np.zeros(n, dtype=bool)
    ambient = set() if ambient_atoms is None else {canon_atom(a) for a in ambient_atoms}

    equivalences: List[ClassEquivalence] = []
    inclusions: List[ClassInclusion] = []

    # --- Equivalences ---
    for i, ki in enumerate(keys):
        mi = mask_cache[ki]
        Ai = obj_cache[ki]
        atoms_i = atoms_cache[ki]
        for kj in keys[i + 1 :]:
            mj = mask_cache[kj]
            Aj = obj_cache[kj]
            atoms_j = atoms_cache[kj]
            if not same_mask(mi, mj):
                continue
            if skip_trivial_equiv and (np.array_equal(mi, all_true) or np.array_equal(mi, all_false)):
                continue
            # Avoid reporting equivalence when atom sets are strictly nested
            if (atoms_i.issubset(atoms_j) and atoms_i != atoms_j) or (atoms_j.issubset(atoms_i) and atoms_i != atoms_j):
                continue
            equivalences.append(ClassEquivalence(Ai, Aj))

    # --- Inclusions ---
    for ki in keys:
        Ai = obj_cache[ki]
        mi = mask_cache[ki]
        atoms_i = atoms_cache[ki]
        suppA = mask_support(mi)
        if suppA < int(min_support_A) or np.array_equal(mi, all_false):
            continue
        for kj in keys:
            if kj == ki:
                continue
            Aj = obj_cache[kj]
            mj = mask_cache[kj]
            atoms_j = atoms_cache[kj]
            if np.array_equal(mj, all_true):
                continue
            # Check mask(A) subset of mask(B)
            if np.any(mi & ~mj):
                continue
            # Exclude trivial syntactic nesting B ⊆ A
            if atoms_j.issubset(atoms_i):
                continue
            # Disallow shared atoms except those whitelisted in ambient
            if disallow_shared_atoms and ((atoms_i & atoms_j) - ambient):
                continue
            inclusions.append(ClassInclusion(Ai, Aj))

    # De-duplicate by relation signature
    def _uniq(items):
        seen, out = set(), []
        for x in items:
            sig = x.signature()
            if sig in seen:
                continue
            seen.add(sig)
            out.append(x)
        return out

    equivalences = _uniq(equivalences)
    inclusions = _uniq(inclusions)
    return equivalences, inclusions
