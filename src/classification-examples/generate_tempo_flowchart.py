"""
Script to generate clean, minimal TEMPO 5-stage pipeline flowchart.
Matches clean, basic 2-column connector flowchart styling with perfect bounding box fits and routing.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.environ["MPLCONFIGDIR"] = str(Path("presentation_figures/.cache").resolve())

OUTPUT_DIR = Path("presentation_figures")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

DOCS_DIR = Path("docs/Documents/Deliverable 2 - Mid-year Report/presentation_figures")
DOCS_DIR.mkdir(exist_ok=True, parents=True)


def draw_flowchart():
    fig, ax = plt.subplots(figsize=(17, 15), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 15)

    box_edge = "#0F172A"
    box_fill = "#FFFFFF"
    pill_fill = "#F8FAFC"
    circle_fill = "#F1F5F9"
    text_color = "#0F172A"
    subtext_color = "#334155"
    arrow_color = "#0F172A"
    font_main = "sans-serif"

    # Title
    ax.text(
        8.5, 14.4, "TEMPO 5-Stage Benchmarking Pipeline",
        ha="center", va="center", fontsize=20, fontweight="bold",
        color=text_color, fontfamily=font_main
    )

    def draw_box(x, y, w, h, title, items=None):
        rect = patches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.08,rounding_size=0.08",
            facecolor=box_fill, edgecolor=box_edge, linewidth=1.5, zorder=3
        )
        ax.add_patch(rect)
        
        title_y = y + h / 2 - 0.32 if items else y
        ax.text(
            x, title_y, title, ha="center", va="center",
            fontsize=12.0, fontweight="bold", color=text_color, fontfamily=font_main, zorder=4
        )
        
        if items:
            item_text = "\n".join(items)
            ax.text(
                x - w / 2 + 0.25, title_y - 0.28, item_text, ha="left", va="top",
                fontsize=9.0, color=subtext_color, fontfamily=font_main, linespacing=1.40, zorder=4
            )
        return rect

    def draw_pill(x, y, w, h, text):
        pill = patches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.1,rounding_size=0.35",
            facecolor=pill_fill, edgecolor=box_edge, linewidth=1.5, zorder=3
        )
        ax.add_patch(pill)
        ax.text(
            x, y, text, ha="center", va="center",
            fontsize=12.5, fontweight="bold", color=text_color, fontfamily=font_main, zorder=4
        )
        return pill

    def draw_connector(x, y, radius, label):
        circle = patches.Circle(
            (x, y), radius,
            facecolor=circle_fill, edgecolor=box_edge, linewidth=1.5, zorder=3
        )
        ax.add_patch(circle)
        ax.text(
            x, y, label, ha="center", va="center",
            fontsize=12.5, fontweight="bold", color=text_color, fontfamily=font_main, zorder=4
        )
        return circle

    def draw_diamond(x, y, w, h, text):
        pts = [
            [x, y + h / 2],      # top
            [x + w / 2, y],      # right
            [x, y - h / 2],      # bottom
            [x - w / 2, y],      # left
        ]
        poly = patches.Polygon(
            pts, closed=True,
            facecolor=pill_fill, edgecolor=box_edge, linewidth=1.5, zorder=3
        )
        ax.add_patch(poly)
        ax.text(
            x, y, text, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=text_color, fontfamily=font_main, zorder=4
        )
        return poly

    def draw_arrow(x1, y1, x2, y2, label=None, label_pos="right", label_offset=(0.15, 0)):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>", color=arrow_color, lw=1.5,
                mutation_scale=14, shrinkA=0, shrinkB=0
            ), zorder=2
        )
        if label:
            lx = (x1 + x2) / 2 + label_offset[0]
            ly = (y1 + y2) / 2 + label_offset[1]
            ha = "left" if label_pos == "right" else "right"
            ax.text(
                lx, ly, label, ha=ha, va="center",
                fontsize=9.2, fontweight="bold", color=text_color, fontfamily=font_main, zorder=4
            )

    # --------------------------------------------------------------------------
    # COLUMN 1: Ingestion to Feature Extraction & Caching
    # --------------------------------------------------------------------------
    col1_x = 3.8
    w_main = 5.6
    
    # 1. Start Pill
    y_start = 13.4
    draw_pill(col1_x, y_start, 3.2, 0.65, "Start Pipeline")
    draw_arrow(col1_x, y_start - 0.33, col1_x, 12.45)

    # 2. Process Step 1: Ingestion & Storage
    y_p1 = 11.0
    h_p1 = 2.6
    draw_box(
        col1_x, y_p1, w_main, h_p1,
        "1. Data Ingestion & Storage",
        items=[
            "• Datasets: BEED, AI4I, HAR, Gas Drift, Energy, PM2.5",
            "• Window segmentation: window_size, stride parameters",
            "• Label strategy: majority_vote, last, any_positive",
            "• Parquet schema: time_series.parquet, targets.parquet",
            "• Internal formats: Polars DataFrame, 3D NumPy Tensor"
        ]
    )
    draw_arrow(col1_x, y_p1 - h_p1 / 2, col1_x, 9.15)

    # 3. Decision Diamond: In Feature Cache?
    y_dec = 8.35
    w_dec = 2.6
    h_dec = 1.4
    draw_diamond(col1_x, y_dec, w_dec, h_dec, "Features\nCached?")
    draw_arrow(col1_x, y_dec - h_dec / 2, col1_x, 7.05, label="No (Compute)", label_pos="right", label_offset=(0.15, 0))

    # 4. Process Step 2: Feature Extraction
    y_p2 = 5.45
    h_p2 = 2.9
    draw_box(
        col1_x, y_p2, w_main, h_p2,
        "2. Feature Extraction",
        items=[
            "• Numba Efficient: ~780 features/channel (JIT C-speed)",
            "• Polars / NumPy Stats: 5 summary statistical features/channel",
            "• TSFresh: Minimal (8 feat), Efficient (~780 feat), Comprehensive",
            "• TSFEL: Statistical, Spectral, Temporal domains",
            "• Hardware Telemetry: Process RSS RAM, CPU%, GPU%, Time",
            "• Persistent Store: Save matrix to Parquet / HDF5 / Memory"
        ]
    )

    # Choice "Yes" -> Right bypass branch
    x_bypass = col1_x + w_main / 2 + 1.4
    w_cache = 2.1
    h_cache = 1.3
    y_cache = 5.7

    # Line right from diamond to bypass x
    ax.plot([col1_x + w_dec / 2, x_bypass], [y_dec, y_dec], color=arrow_color, lw=1.5, zorder=2)
    ax.text(col1_x + w_dec / 2 + 0.15, y_dec + 0.22, "Yes (Cache Hit)", fontsize=9.2, fontweight="bold", color=text_color, fontfamily=font_main, zorder=4)
    
    # Arrow down from decision branch to top of Load Cache box
    draw_arrow(x_bypass, y_dec, x_bypass, y_cache + h_cache / 2)
    
    # Bypass box
    draw_box(
        x_bypass, y_cache, w_cache, h_cache,
        "Load Cache",
        items=["• Memory / Parquet", "• Zero-compute bypass"]
    )

    # Arrow down from bottom of Load Cache box to corner y_conn1
    y_conn1 = 1.5
    ax.plot([x_bypass, x_bypass], [y_cache - h_cache / 2, y_conn1], color=arrow_color, lw=1.5, zorder=2)
    
    # Arrow from Step 2 down to bottom connector
    draw_arrow(col1_x, y_p2 - h_p2 / 2, col1_x, y_conn1 + 0.38)
    
    # Line from bypass to connector A
    draw_arrow(x_bypass, y_conn1, col1_x + 0.38, y_conn1)

    # Connector circle A at bottom of Col 1
    draw_connector(col1_x, y_conn1, 0.38, "A")


    # --------------------------------------------------------------------------
    # COLUMN 2: Selection to Model Fits & Statistical Assessment
    # --------------------------------------------------------------------------
    col2_x = 12.3

    # Connector circle A at top of Col 2
    y_conn2 = 13.4
    draw_connector(col2_x, y_conn2, 0.38, "A")
    draw_arrow(col2_x, y_conn2 - 0.38, col2_x, 12.25)

    # 5. Process Step 3: CV Splitting
    y_p3 = 11.25
    h_p3 = 1.75
    draw_box(
        col2_x, y_p3, w_main, h_p3,
        "3. Cross-Validation Splitting",
        items=[
            "• K-Fold (Regression) / Stratified K-Fold (Classification)",
            "• n_splits = 5, multiple random seed configurations",
            "• Enforces strict partition isolation across folds"
        ]
    )
    draw_arrow(col2_x, y_p3 - h_p3 / 2, col2_x, 9.75)

    # 6. Process Step 4: Feature Selection
    y_p4 = 8.1
    h_p4 = 3.0
    draw_box(
        col2_x, y_p4, w_main, h_p4,
        "4. Feature Selection (Train Split)",
        items=[
            "• Filters: None (Passthrough), TSFresh FDR (q < 0.05)",
            "• SelectKBest: ANOVA F-statistic, Mutual Information (top-k)",
            "• Distribution tests: Cramér-von Mises, Kolmogorov-Smirnov",
            "• Wrappers: Boruta (Random Forest shadow permutations)",
            "• Subsampled: SubsampledFeatureSelector (10% ratio wrapper)",
            "• Telemetry: Duration, % reduction, Jaccard fold stability"
        ]
    )
    draw_arrow(col2_x, y_p4 - h_p4 / 2, col2_x, 6.0)

    # 7. Process Step 5: Model Fit & Inference
    y_p5 = 4.7
    h_p5 = 2.3
    draw_box(
        col2_x, y_p5, w_main, h_p5,
        "5. Model Training & Evaluation",
        items=[
            "• Classification: RandomForestClassifier, LogisticRegression",
            "• Regression: RandomForestRegressor, Ridge",
            "• Pipeline: SimpleImputer, StandardScaler transformation",
            "• Metrics: Accuracy, RMSE, MAE, R², Prediction Latency"
        ]
    )
    draw_arrow(col2_x, y_p5 - h_p5 / 2, col2_x, 3.05)

    # 8. Process Step 6: Telemetry & Assessment
    y_p6 = 1.95
    h_p6 = 2.0
    draw_box(
        col2_x, y_p6, w_main, h_p6,
        "6. Telemetry & Statistical Assessment",
        items=[
            "• Consolidated benchmark results exported to structured CSV",
            "• Paired Student's t-tests + Benjamini-Hochberg FDR (q < 0.05)",
            "• Cohen's d effect sizes across extractors and selectors",
            "• Automated boxplot generation (linear & log-scale)"
        ]
    )
    draw_arrow(col2_x, y_p6 - h_p6 / 2, col2_x, 0.75)

    # 9. End Pill
    y_end = 0.45
    draw_pill(col2_x, y_end, 3.4, 0.60, "End / Bakeoff Complete")

    for d in [OUTPUT_DIR, DOCS_DIR]:
        svg_p = d / "tempo_pipeline_flowchart.svg"
        png_p = d / "tempo_pipeline_flowchart.png"
        fig.savefig(svg_p, bbox_inches="tight", format="svg")
        fig.savefig(png_p, bbox_inches="tight", dpi=300, format="png")
        print(f"Saved: {png_p}")

    plt.close(fig)


if __name__ == "__main__":
    draw_flowchart()
