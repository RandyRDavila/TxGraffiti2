ConjecturePlayground
=====================

`ConjecturePlayground` is your main entry point for automated conjecture discovery.
It wraps a `pandas.DataFrame` and provides methods to:

- **Generate** conjectures via multiple generators (`convex_hull`, `ratios`, `lp_model`, …)
- **Filter** them with heuristics (`morgan`, `dalmatian`, …)
- **Post‐process** (sort, dedupe, strengthen equalities)
- **Quantify** them (`forall`, `exists`)
- **Persist** new hypotheses (`discover_equalities`)
- **Export** results to JSON/CSV or Lean theorem stubs

Class Signature
---------------

.. code-block:: python

   class ConjecturePlayground:
       def __init__(
           self,
           df: pandas.DataFrame,
           object_symbol: str = "G",
           *,
           base: Optional[Union[str, Predicate]] = None
       ):
           ...

       def discover(
           self, *args, **kwargs
       ) -> List[Conjecture]:
           ...

       def generate(
           self, *,
           methods: Optional[List[Callable]] = None,
           features: Optional[List[Union[str,Property]]] = None,
           target: Optional[Union[str,Property]] = None,
           hypothesis: Optional[
               Union[str, Predicate, Sequence[Union[str,Predicate]]]
           ] = None,
           heuristics: Optional[List[Callable]] = None,
           post_processors: Optional[List[Union[str,Callable]]] = None,
           **kwargs
       ) -> Iterator[Conjecture]:
           ...

       def discover_equalities(
           self, *,
           generators: Optional[List[Callable]] = None,
           min_fraction: float = 0.15
       ) -> List[Predicate]:
           ...

       def append_row(self, row: dict) -> None: ...
       def reset(self, new_df: Optional[pd.DataFrame] = None) -> None: ...
       def export_conjectures(self, path: str, format: str = "json") -> None: ...
       def export_to_lean(
           self, path: str,
           name_prefix: str = "conjecture",
           object_symbol: Optional[str] = None
       ) -> None: ...
       def convert_columns(self, convert_dict: dict) -> None: ...

Initialization (`__init__`)
---------------------------

- **df**: the `DataFrame` to explore
- **object_symbol**: a name used by `.forall`/`.exists` in pretty printing
- **base**: an optional `Predicate` or column name to conjoin with every hypothesis
  (defaults to `TRUE`, i.e. no restriction)

Discovery (`discover`)
-----------------------

A one‐stop method that runs:

1. **Generators** (all registered or user‐supplied) × **Hypotheses** (base ∧ each given)
2. **Heuristics** in sequence (global kept list)
3. **Post‐processors** on the final list (e.g. `remove_duplicates`, `sort_by_touch_count`)

Returns the full list of accepted `Conjecture` objects and caches them in `.conjectures`.

Generation (`generate`)
------------------------

The streaming equivalent of `discover`.  Yields `Conjecture` objects one by one,
in generator order, before any post‐processing.  Signature mirrors `discover` but
returns an iterator.

Quantifiers (`forall`, `exists`)
--------------------------------

- `pg.forall(pred)` wraps a `Predicate` in universal notation (`∀ G: …`)
- `pg.exists(pred)` wraps in existential notation (`∃ G: …`)

Hypothesis Mining (`discover_equalities`)
-----------------------------------------

Automatically finds any single-feature inequalities `L ≤ R` or `L ≥ R` (under `TRUE`)
that are **tight** on at least `min_fraction` of the rows, registers them as new
`Predicate`s (`L_eq_R`), and returns them sorted by support.

Dataframe Updates
-----------------

- `append_row(row: dict)` adds a new row in place and invalidates the cache
- `reset(new_df: DataFrame = None)` replaces or rebinds the DataFrame, clearing cached conjectures

Exporting
----------

- `def conjecture_to_lean4(conj: Conjecture, name: str, object_symbol: str = "G", object_decl: str = "SimpleGraph V") -> str` writes Lean 4
  theorem stubs for all cached conjectures

Column Conversions
------------------

- `convert_columns(convert_dict)` applies a mapping function to existing columns,
  e.g. to preprocess raw values into Booleans or enums.

Illustrative Example
--------------------

.. code-block:: python

   import pandas as pd
   from txgraffiti.playground import ConjecturePlayground
   from txgraffiti.generators import convex_hull, ratios, lp_model
   from txgraffiti.heuristics import morgan, dalmatian
   from txgraffiti.postprocessing import remove_duplicates, sort_by_touch_count

   # 1) Load your DataFrame
   df = pd.read_csv("graph_data.csv")

   # 2) Create a playground with base hypothesis “connected”
   pg = ConjecturePlayground(df, object_symbol="G", base="connected")

   # 3) Auto-mine any equality hypotheses holding ≥20% of rows
   new_hyps = pg.discover_equalities(min_fraction=0.2)

   # 4) Run full discovery over two hypotheses:
   conjs = pg.discover(
       methods         = [convex_hull, ratios, lp_model],
       features        = ['order','matching_number','min_degree'],
       target          = 'independence_number',
       hypothesis      = new_hyps,          # includes base ∧ each discovered
       heuristics      = [morgan, dalmatian],
       post_processors = [remove_duplicates, sort_by_touch_count],
       round_decimals  = 2,
       drop_coeff_below= 0.05,
   )

   # 5) Export to Lean
   pg.conjecture_to_lean4(
    "graph_conjectures.lean",
    name="graph_conj",
    object_symbol: str = "G",
    object_decl: str = "SimpleGraph V",
    )

   # 6) Wrap and print top 3
   for i, conj in enumerate(pg.conjectures[:3], start=1):
       print(f"Conjecture {i}.", pg.forall(conj))
