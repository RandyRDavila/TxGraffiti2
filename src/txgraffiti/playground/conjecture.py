import pandas as pd
import numpy as np
import itertools
from typing import Union, Iterator, List, Callable, Sequence, Optional

from txgraffiti.logic import *
from txgraffiti.playground.registry import list_playgrounds
from txgraffiti.processing.registry import get_post
from txgraffiti.export_utils.lean4 import conjecture_to_lean4
from txgraffiti.generators import list_gens


__all__ = [
    'ForAll',
    'Exists',
    'ConjecturePlayground',
]

from collections import defaultdict

def find_strengthened_equalities(conjs):
    """
    Given a list of Conjecture, return new Conjecture of the form
      (H1 ∧ H2) → (LHS == RHS)
    whenever you have both
      H1 → (LHS <= RHS)
    and
      H2 → (LHS >= RHS)
    with identical LHS and identical RHS.name.
    """
    # bucket:  (lhs_prop, rhs_name) → { op_symbol: hypothesis_predicate }
    buckets = defaultdict(dict)

    for c in conjs:
        ineq = c.conclusion
        if not isinstance(ineq, Inequality):
            continue
        key = (ineq.lhs, ineq.rhs.name)
        buckets[key][ineq.op] = c.hypothesis

    equalities = []
    for (lhs_prop, rhs_name), op2hyp in buckets.items():
        if "<=" in op2hyp and ">=" in op2hyp:
            H1 = op2hyp["<="]
            H2 = op2hyp[">="]
            # build the joint hypothesis
            H = H1 & H2
            # build equality Predicate: lhs == rhs
            eq_pred = lhs_prop == Constant(0)  # dummy, we'll override
            # better: directly use Inequality
            eq_pred = Inequality(lhs_prop, "==",
                                 Property(rhs_name,
                                          lambda df, p=rhs_name: pd.Series(
                                              df.eval(p), index=df.index)))
            # but easier: reuse one of the originals:
            # pick the <= version to supply the rhs Property
            any_le_conj = next(c for c in conjs
                               if isinstance(c.conclusion, Inequality)
                               and c.hypothesis is H1
                               and c.conclusion.op in ("<=", "<"))
            rhs_prop = any_le_conj.conclusion.rhs
            eq_pred = Inequality(lhs_prop, "==", rhs_prop)

            # and finally the new Conjecture
            equalities.append(Conjecture(H, eq_pred))
    return equalities


class ForAll:
    def __init__(self, conj: Predicate, df: pd.DataFrame, object_symbol="G"):
        self.pred = conj; self.df = df
        self.name = f"∀ {object_symbol}: {conj.name}"
    def is_true(self) -> bool:
        return bool(self.pred(self.df).all())
    def counterexamples(self) -> pd.DataFrame:
        return self.conj.counterexamples(self.df)
    def __repr__(self):
        return self.name

class Exists:
    def __init__(self, pred: Predicate, df: pd.DataFrame, object_symbol="G"):
        self.pred = pred; self.df = df
        self.name = f"∃ {object_symbol}: {pred.name}"
    def is_true(self) -> bool:
        return bool(self.pred(self.df).any())
    def witness(self) -> pd.DataFrame:
        return self.df[self.pred(self.df)]
    def __repr__(self):
        return self.name

class ConjecturePlayground:
    def __init__(self, df: pd.DataFrame, object_symbol="G", base: Optional[Union[str, Predicate]] = None):
        self.df = df
        self.conjectures: list[Conjecture] = []   # cache slot
        self.object_symbol = object_symbol
        # Set up the base predicate
        if base is None:
            # default to “always true”
            self.base: Predicate = TRUE
        elif isinstance(base, Predicate):
            self.base = base
        elif isinstance(base, str):
            # lift via getattr; assumes you have a boolean column or existing predicate
            self.base = getattr(self, base)
        else:
            raise TypeError("`base` must be None, a column-name (str), or a Predicate")

        self.conjectures: List[Conjecture] = []
        self.custom_hypotheses: dict[str, Predicate] = {}

    def prop(self, name: str) -> Property:
        return Property(name, lambda df, c=name: df[c])

    def __getattr__(self, attr):
        if attr in self.df.columns:
            col = self.df[attr]
            if pd.api.types.is_bool_dtype(col):
                return Predicate(attr, lambda df, c=attr: df[c])
            return self.prop(attr)
        return super().__getattribute__(attr)

    def forall(self, pred: Predicate) -> bool:
        # return bool(pred(self.df).all())
        return ForAll(pred, self.df, object_symbol=self.object_symbol)

    def exists(self, pred: Predicate) -> bool:
        # return bool(pred(self.df).any())
        return Exists(pred, self.df, object_symbol=self.object_symbol)

    def generate(
        self,
        *,
        methods:      List[Callable[..., Iterator[Conjecture]]] = None,
        features:     Optional[List[Union[str, Property]]]       = None,
        target:       Optional[Union[str, Property]]              = None,
        hypothesis:   Optional[
                          Union[str, Predicate, Sequence[Union[str, Predicate]]]
                      ] = None,
        heuristics:      Optional[List[Callable[[Conjecture, List[Conjecture], pd.DataFrame], bool]]] = None,
        post_processors: Optional[List[Union[str, Callable[[List[Conjecture], pd.DataFrame], List[Conjecture]]]]] = None,
        **kwargs
    ) -> Iterator[Conjecture]:
        gens = methods or list_playgrounds()

        # Build hyp_list by conjoining each user hypothesis with self.base
        if hypothesis is None:
            hyp_list = [self.base]
        elif isinstance(hypothesis, (list, tuple)):
            lifted = [
                (getattr(self, h) if isinstance(h, str) else h)
                for h in hypothesis
            ]
            hyp_list = [self.base & h for h in lifted]
        else:
            h = (getattr(self, hypothesis) if isinstance(hypothesis, str)
                 else hypothesis)
            hyp_list = [self.base & h]

        # Lift features & target
        feat_list = [] if features is None else features
        targ_arg  = None if target is None else target

        feat_props = [
            (self.prop(f) if isinstance(f, str) else f)
            for f in feat_list
        ]
        targ_prop = (self.prop(targ_arg) if isinstance(targ_arg, str) else targ_arg)

        # 4) raw iterator
        def raw_all():
            for gen_fn in gens:
                for hyp in hyp_list:
                    yield from gen_fn(
                        self.df,
                        features   = feat_props,
                        target     = targ_prop,
                        hypothesis = hyp,
                        **kwargs
                    )

        # … + heuristics & post‐processing as before …
        stream = raw_all()

        # ———————————————————————————————
        # 2) apply heuristics into a new iterator
        # ———————————————————————————————

        if heuristics:
            kept: List[Conjecture] = []
            def filtered() -> Iterator[Conjecture]:
                for conj in raw_all():
                    if all(h(conj, kept, self.df) for h in heuristics):
                        kept.append(conj)
                        yield conj
            stream = filtered()
        else:
            stream = raw_all()

        # ———————————————————————————————
        # 3) post-processing if requested
        # ———————————————————————————————

        if post_processors:
            # collect into list
            all_kept = list(stream)
            # apply each post-processor in order
            for proc in post_processors:
                fn = get_post(proc) if isinstance(proc, str) else proc
                all_kept = fn(all_kept, self.df)
            # yield final
            for c in all_kept:
                yield c
        else:
            # no post-processing → just stream
            yield from stream

    def discover(self, *args, **kwargs) -> list[Conjecture]:
        """
        Run generate(...) with the given parameters, cache, and return the list.
        """
        conjs = list(self.generate(*args, **kwargs))
        self.conjectures = find_strengthened_equalities(conjs)
        self.conjectures.extend(conjs)
        return self.conjectures

    def discover_equalities(
        self,
        *,
        generators:    list[Callable[..., Iterator[Conjecture]]] = None,
        min_fraction:  float = 0.15
    ) -> list[Predicate]:
        """
        Find all single‐feature inequalities of the form L≤R or L≥R
        (with hypothesis TRUE) that are *tight* (slack==0) on at least
        `min_fraction` of the rows.  For each, register & return
        a Predicate `L_eq_R`.
        """
        df   = self.df
        gens = generators or list_gens()
        N    = len(df)
        found: list[tuple[Predicate, float]] = []

        # 1) numeric columns as Properties
        num_cols = df.select_dtypes(include=[np.number]).columns
        props = {c: self.prop(c) for c in num_cols}

        # 2) for each generator × each (P,Q) pair
        for gen_fn in gens:
            for P_name, Q_name in itertools.permutations(num_cols, 2):
                P, Q = props[P_name], props[Q_name]
                # run under TRUE, but safely
                try:
                    conj_iter = gen_fn(
                        df,
                        features   = [P],
                        target     = Q,
                        hypothesis = TRUE
                    )
                except Exception:
                    # skip any generator that errors out on this (P,Q)
                    continue

                for conj in conj_iter:
                    if not isinstance(conj.conclusion, Inequality):
                        continue
                    ineq = conj.conclusion

                    support = ineq.touch_count(df) / N
                    if support < min_fraction:
                        continue

                    L, R = ineq.lhs, ineq.rhs

                    pred = L == R
                    if pred.name not in self.custom_hypotheses:
                        self.custom_hypotheses[pred.name] = pred
                        found.append((pred, support))

        # 3) return just the Predicates, sorted by descending support
        found.sort(key=lambda pr: pr[1], reverse=True)
        return [pr[0] for pr in found]

    def append_row(self, row: dict):
        """
        Append a new row (dict of col:value) to self.df,
        then clear any cached conjectures.
        """
        # Append in-place
        self.df.loc[len(self.df)] = row
        # Clear cached results
        self.conjectures = []

    def reset(self, new_df: pd.DataFrame = None):
        """
        If new_df is given, replace self.df; otherwise rebind the same DataFrame
        (e.g. after in-place edits).  In either case, clear cached conjectures.
        """
        if new_df is not None:
            self.df = new_df
        # Invalidate cache
        self.conjectures = []

    def export_conjectures(self, path: str, format: str = "json"):
        """
        Write self.conjectures to disk in JSON or CSV form.
        """
        if not hasattr(self, "conjectures"):
            raise RuntimeError("No conjectures to export; run discover() first.")
        rows = []
        for c in self.conjectures:
            rows.append({
                "hypothesis":   c.hypothesis.name,
                "conclusion":   c.conclusion.name,
                "accuracy":     c.accuracy(self.df),
                # you could even serialize masks or touch_counts
            })
        dfc = pd.DataFrame(rows)
        if format == "json":
            dfc.to_json(path, orient="records", lines=True)
        else:
            dfc.to_csv(path, index=False)


    def convert_columns(self, convert_dict : dict):
        for key, func in convert_dict.items():
            self.df[key] = self.df[key].map(func)

    def export_to_lean(
        self,
        path: str,
        name_prefix: str = "conjecture",
        object_symbol: Optional[str] = None
    ):
        """
        Write all cached conjectures to `path` in Lean theorem‐stub format.
        Each theorem will be named `{name_prefix}_{i}`.
        """
        symbol = object_symbol or self.object_symbol
        lines = []
        for i, conj in enumerate(self.conjectures, start=1):
            thm_name = f"{name_prefix}_{i}"
            # render using your helper, passing through the object symbol
            src = conjecture_to_lean4(conj, thm_name, object_symbol=symbol)
            lines.append(src)

        # Write out to file
        with open(path, "w") as f:
            f.write("\n".join(lines))

        print(f"Wrote {len(lines)} Lean theorems to {path}")
