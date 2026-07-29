"""
analysis.py

Statistical analysis for benchmark_results.csv.
"""

import os
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests


INPUT_FILE = "benchmark_results1.csv"
OUTPUT_DIR = "analysis_fourier"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)
df["Extractor"] = df["Extractor"].fillna("None").astype(str)
df["Selector"] = df["Selector"].fillna("None").astype(str)
df["Combination"] = df["Extractor"] + " + " + df["Selector"]

metrics = [
    "Prediction Accuracy",
    "Extraction Time",
    "Selection Time",
    "Prediction Time",
    "Total Time",
]

summary = (
    df.groupby(["Extractor","Selector"])[metrics]
      .agg(["mean","std","median","min","max"])
)
summary.to_csv(os.path.join(OUTPUT_DIR,"summary_statistics.csv"))

def boxplot(metric, group):

    width = 12
    if group == "Combination":
        width = max(16, len(df[group].unique()) * 0.6)

    plt.figure(figsize=(width, 6))

    df.boxplot(column=metric, by=group)

    plt.title(f"{metric} by {group}")
    plt.suptitle("")

    plt.xticks(
        rotation=60,
        ha="right",
        fontsize=9
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{metric.replace(' ','_').lower()}_by_{group.lower()}.png"
        ),
        dpi=300
    )

    plt.close()

for g in ["Extractor","Selector","Combination"]:
    for m in metrics:
        boxplot(m,g)

def cohens_d(x,y):
    d=x-y
    s=d.std(ddof=1)
    return np.nan if s==0 else d.mean()/s

def pairwise(group_col, metric):
    methods=sorted(df[group_col].unique())
    rows=[]
    for a,b in combinations(methods,2):
        x=df[df[group_col]==a].sort_values("Dataset")[metric].to_numpy()
        y=df[df[group_col]==b].sort_values("Dataset")[metric].to_numpy()
        n=min(len(x),len(y))
        x=x[:n]; y=y[:n]
        t,p=ttest_rel(x,y)
        d=cohens_d(x,y)
        m1=x.mean(); m2=y.mean()
        if "Accuracy" in metric:
            better=a if m1>m2 else b
            if p>=0.05:
                concl="No significant difference"
            else:
                concl=f"{better} is significantly more accurate"
        else:
            faster=a if m1<m2 else b
            if p>=0.05:
                concl="No significant difference"
            else:
                concl=f"{faster} is significantly faster"
        rows.append({
            "Method 1":a,
            "Method 2":b,
            "Mean 1":m1,
            "Mean 2":m2,
            "Difference":m1-m2,
            "t statistic":t,
            "p value":p,
            "Cohens d":d,
            "Conclusion":concl
        })
    out=pd.DataFrame(rows)
    if not out.empty:
        out["Adjusted p"]=multipletests(out["p value"],method="fdr_bh")[1]
        out["Significant"]=out["Adjusted p"]<0.05
    return out

for grp,name in [("Extractor","extractor"),("Selector","selector"),("Combination","combination")]:
    for metric,filetag in [
        ("Prediction Accuracy","accuracy"),
        ("Extraction Time","extraction_time"),
        ("Selection Time","selection_time"),
        ("Prediction Time","prediction_time"),
        ("Total Time","total_time"),
    ]:
        pairwise(grp,metric).to_csv(
            os.path.join(OUTPUT_DIR,f"{name}_{filetag}_ttests.csv"),
            index=False
        )

print("Analysis complete.")
