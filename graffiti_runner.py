

from txgraffiti2025.runner import TxGraffitiConfig, TxGraffitiRunner, print_full_result
from txgraffiti.example_data import graph_data as df

df["nontrivial"] = df["connected"]
df.drop(
    columns=[
        "vertex_cover_number",
        "cograph",
        "cubic",
        "chordal",
        "tree",
        "size",
        "triameter",
    ],
    inplace=True,
)

cfg = TxGraffitiConfig(
    target="independence_number",
    enable_eq_bootstrap=False,
    k_affine_values=(1, 2),
)

runner = TxGraffitiRunner(
    config=cfg,
    report_path="reports/txgraffiti_run.txt",
)

result = runner.run(df)

print_full_result(
    result,
    k_per_bucket=20,
    top_sophie=20,
    max_atomic=20,
    max_asymptotic=20,
    max_eq_classes=20,
)
