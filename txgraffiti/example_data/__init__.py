import pandas as pd
import importlib.resources as pkg_resources

def _load_csv(name: str) -> pd.DataFrame:
    # this will read the CSV bundled in this package
    with pkg_resources.open_text(__name__, name) as fp:
        return pd.read_csv(fp)

# expose it at module level:
graph_data = _load_csv("graph_data.csv")

integer_data = _load_csv("integer_data.csv")

nba_game_data = _load_csv("nba_game_data.csv")
