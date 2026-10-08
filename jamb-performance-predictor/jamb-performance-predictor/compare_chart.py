"""Chart the model comparison (thesis Figure 4.2).

Uses results/model_comparison.csv if you have run compare.py,
otherwise the figures reported in Table 4.1 of the thesis.

Run:  python compare_chart.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plots import comparison_figure
from project_info import THESIS_METRICS

pd.set_option("display.float_format", "{:.4f}".format)

csv = Path("results/model_comparison.csv")
df = pd.read_csv(csv) if csv.exists() else pd.DataFrame(THESIS_METRICS)
print("=== Model Evaluation Metrics (Test Set) ===")
print(df.to_string(index=False))

best = df.loc[df["R2"].idxmax()]
print(f"\nBest model by R2: {best['Model']} (R2 {best['R2']:.4f}, "
      f"RMSE {best['RMSE']:.4f}, MAE {best['MAE']:.4f})")

fig = comparison_figure(df, facecolor="white")
Path("results").mkdir(exist_ok=True)
fig.savefig("results/model_comparison.png", dpi=160, facecolor="white")
print("Saved results/model_comparison.png")
plt.show()
