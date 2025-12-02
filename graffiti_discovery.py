from txgraffiti2025.math_discovery import MathDiscovery, MathDiscoveryConfig
from txgraffiti.example_data import graph_data as df

df["nontrivial"] = df["connected"]
df.drop(columns=['vertex_cover_number'], inplace=True)
TARGET = "independence_number"

config = MathDiscoveryConfig(
    touch_atol=0.0,
    touch_rtol=0.0,
    top_k_per_bucket=100,
)

md = MathDiscovery(df, target=TARGET, config=config)

final_bank = md.run_full_pipeline(
    lp_direction="both",
    lp_max_denom=20,
    lp_min_support=3,
    k_values=(1, ),
    k_hypotheses_limit=25,
    k_min_touch=3,
    k_max_denom=20,
    k_top_m_by_variance=None,
    intricate_weight=0.5,
    intricate_min_touch=3,
    enable_integer_lift=True,
)

md.print_bank(final_bank, k_per_bucket=16)
