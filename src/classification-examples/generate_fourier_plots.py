"""
Interactive Matplotlib subplot script for Fourier coefficient truncation experiments.

Generates a combined figure with two subplots:
  - Subplot 1 (Left): Prediction Accuracy vs FFT Coefficients (n)
  - Subplot 2 (Right): Extraction Time vs FFT Coefficients (n)

Opens in an interactive window so you can adjust aspect ratio before saving.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Exact telemetry dataset corresponding to Table 2 in Mid-Year Report
data = [
    {"Method": "Minimal", "FFT coefficients": 0, "Features extracted": 40, "Features selected": 31, "Extraction Time": 3.205604, "Selection Time": 2.922331, "Accuracy (%)": 78.723404},
    {"Method": "Efficient", "FFT coefficients": 5, "Features extracted": 1510, "Features selected": 657, "Extraction Time": 18.101483, "Selection Time": 3.938444, "Accuracy (%)": 78.723404},
    {"Method": "Efficient", "FFT coefficients": 10, "Features extracted": 1590, "Features selected": 673, "Extraction Time": 19.191631, "Selection Time": 3.986827, "Accuracy (%)": 79.787234},
    {"Method": "Efficient", "FFT coefficients": 25, "Features extracted": 1798, "Features selected": 723, "Extraction Time": 22.097346, "Selection Time": 4.631882, "Accuracy (%)": 77.659574},
    {"Method": "Efficient", "FFT coefficients": 50, "Features extracted": 1798, "Features selected": 723, "Extraction Time": 22.722418, "Selection Time": 4.228905, "Accuracy (%)": 77.659574},
    {"Method": "Efficient", "FFT coefficients": 75, "Features extracted": 1798, "Features selected": 723, "Extraction Time": 18.918715, "Selection Time": 4.126958, "Accuracy (%)": 77.659574},
    {"Method": "Efficient", "FFT coefficients": 100, "Features extracted": 1798, "Features selected": 723, "Extraction Time": 19.234950, "Selection Time": 4.067573, "Accuracy (%)": 77.659574},
    {"Method": "Comprehensive", "FFT coefficients": 5, "Features extracted": 1534, "Features selected": 671, "Extraction Time": 20.836962, "Selection Time": 4.007044, "Accuracy (%)": 79.797234},
    {"Method": "Comprehensive", "FFT coefficients": 10, "Features extracted": 1614, "Features selected": 687, "Extraction Time": 22.073055, "Selection Time": 4.097040, "Accuracy (%)": 77.659574},
    {"Method": "Comprehensive", "FFT coefficients": 25, "Features extracted": 1822, "Features selected": 737, "Extraction Time": 21.094772, "Selection Time": 4.221230, "Accuracy (%)": 80.851064},
    {"Method": "Comprehensive", "FFT coefficients": 50, "Features extracted": 1822, "Features selected": 737, "Extraction Time": 21.934420, "Selection Time": 4.332901, "Accuracy (%)": 80.851064},
    {"Method": "Comprehensive", "FFT coefficients": 75, "Features extracted": 1822, "Features selected": 737, "Extraction Time": 21.389142, "Selection Time": 4.219188, "Accuracy (%)": 80.851064},
    {"Method": "Comprehensive", "FFT coefficients": 100, "Features extracted": 1822, "Features selected": 737, "Extraction Time": 22.003647, "Selection Time": 4.189112, "Accuracy (%)": 80.851064},
]

df = pd.DataFrame(data)

out_dir = Path(__file__).parent / "analysis"
out_dir.mkdir(exist_ok=True, parents=True)

# Create 1x2 Subplot Figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

# Subplot 1: Prediction Accuracy
for method in ["Minimal", "Efficient", "Comprehensive"]:
    sub = df[df["Method"] == method]
    ax1.plot(
        sub["FFT coefficients"],
        sub["Accuracy (%)"],
        marker="o",
        label=method
    )

ax1.set_title("Prediction Accuracy by Extractor")
ax1.set_xlabel("FFT Coefficients")
ax1.set_ylabel("Prediction Accuracy (%)")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(title="Extractor")

# Subplot 2: Extraction Time
for method in ["Minimal", "Efficient", "Comprehensive"]:
    sub = df[df["Method"] == method]
    ax2.plot(
        sub["FFT coefficients"],
        sub["Extraction Time"],
        marker="o",
        label=method
    )

ax2.set_title("Extraction Time by Extractor")
ax2.set_xlabel("FFT Coefficients")
ax2.set_ylabel("Extraction Time (s)")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(title="Extractor")

plt.tight_layout()

# Save image file
out_file = out_dir / "fourier_combined_subplots.png"
plt.savefig(out_file, dpi=300)
print(f"Saved combined subplot figure to: {out_file}")

# Open window for interactive adjustment
print("Opening interactive Matplotlib window. Adjust aspect ratio as needed!")
plt.show()
