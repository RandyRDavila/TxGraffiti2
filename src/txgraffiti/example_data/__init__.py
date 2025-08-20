import pandas as pd
import importlib.resources as pkg_resources

def _load_csv(name: str) -> pd.DataFrame:
    # Zip/wheel safe: open text resource from this package
    with pkg_resources.open_text(__name__, name) as fp:
        return pd.read_csv(fp)

# Expose CSVs at import time
graph_data = _load_csv("graph_data.csv")
integer_data = _load_csv("integer_data.csv")
nba_game_data = _load_csv("nba_game_data.csv")


# ---- Example graphs loader (zip-safe) ----
import networkx as nx
from importlib.resources import files
import graphcalc as gc  # if this creates a circular import, see note below

def load_example_graphs() -> dict:
    """
    Load all example graphs from the ``graph-edgelists`` resource folder.

    Each ``.txt`` file is a plain-text edge list (one edge per line).
    Graphs are returned as :class:`graphcalc.SimpleGraph` with names taken
    from the filename (without extension).

    Returns
    -------
    dict
        Mapping from graph name (str) to :class:`graphcalc.SimpleGraph`.
    """
    graphs = []
    # Use importlib.resources for zip/wheel-safe resource access
    base_dir = files(__package__).joinpath("graph-edgelists")
    for file in base_dir.iterdir():
        if file.suffix != ".txt":
            continue
        # NetworkX can read from a path-like object
        G_nx = nx.read_edgelist(file, nodetype=int)
        # Wrap into your SimpleGraph and preserve the filename as the name
        graphs.append(gc.SimpleGraph(G_nx.edges, name=file.stem))
    return graphs
