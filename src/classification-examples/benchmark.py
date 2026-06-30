from collections.abc import Callable

import numpy as np
import pandas as pd
import perfplot
import polars as pl
from tsfresh.feature_extraction import feature_calculators as tsfresh

from functime import feature_extractors as fe

pl.Config.set_tbl_rows(100)
pl.Config.set_fmt_str_lengths(60)
pl.Config.set_tbl_hide_column_data_types(True)

# =========================================================
# DATA GENERATION
# =========================================================

# Need >= 1_000_000 rows
n_series = 5_000
series_len = 200

noise_scale = 0.2
trend_scale = 0.01
freq_range = (0.05, 0.2)
seed = 42

rng = np.random.default_rng(seed)

entities = []
times = []
values = []

for i in range(n_series):

    t = np.arange(series_len)

    freq = rng.uniform(*freq_range)
    phase = rng.uniform(0, 2 * np.pi)

    trend = rng.uniform(-trend_scale, trend_scale) * t
    amplitude = rng.uniform(0.5, 2.0)

    signal = amplitude * np.sin(2 * np.pi * freq * t + phase)

    noise = rng.normal(0, noise_scale, size=series_len)

    spikes = (
        rng.choice([0, 1], size=series_len, p=[0.98, 0.02])
        * rng.normal(3, 1, size=series_len)
    )

    x = signal + trend + noise + spikes

    entities.append(np.full(series_len, i))
    times.append(t)
    values.append(x)

df_pl = pl.DataFrame(
    {
        "entity": np.concatenate(entities),
        "time": np.concatenate(times),
        "value": np.concatenate(values),
    }
)

DF_PL_EAGER = df_pl
DF_PANDAS = df_pl.to_pandas()

MAX_ROWS = DF_PL_EAGER.height

print(f"Generated rows: {MAX_ROWS:,}")

# =========================================================
# BENCH CONFIG
# =========================================================

FUNC_PARAMS_BENCH = [
    (fe.absolute_energy, tsfresh.abs_energy, {}, {}),
    (fe.absolute_maximum, tsfresh.absolute_maximum, {}, {}),
    (fe.absolute_sum_of_changes, tsfresh.absolute_sum_of_changes, {}, {}),
    (
        fe.lempel_ziv_complexity,
        tsfresh.lempel_ziv_complexity,
        {"threshold": (pl.col("value").max() - pl.col("value").min()) / 2},
        {"bins": 2},
    ),
    (
        fe.approximate_entropy,
        tsfresh.approximate_entropy,
        {"run_length": 2, "filtering_level": 0.5},
        {"m": 2, "r": 0.5},
    ),
    (fe.autocorrelation, tsfresh.autocorrelation, {"n_lags": 4}, {"lag": 4}),
    (
        fe.binned_entropy,
        tsfresh.binned_entropy,
        {"bin_count": 10},
        {"max_bins": 10},
    ),
    (fe.c3, tsfresh.c3, {"n_lags": 10}, {"lag": 10}),
    (fe.cid_ce, tsfresh.cid_ce, {"normalize": True}, {"normalize": True}),
    (fe.count_above_mean, tsfresh.count_above_mean, {}, {}),
    (fe.count_below_mean, tsfresh.count_below_mean, {}, {}),
    (
        fe.energy_ratios,
        tsfresh.energy_ratio_by_chunks,
        {"n_chunks": 6},
        {
            f"param_{i}": [
                {"num_segments": 6, "segment_focus": i}
            ]
            for i in range(6)
        },
    ),
    (fe.first_location_of_maximum, tsfresh.first_location_of_maximum, {}, {}),
    (fe.first_location_of_minimum, tsfresh.first_location_of_minimum, {}, {}),
    (fe.has_duplicate, tsfresh.has_duplicate, {}, {}),
    (fe.has_duplicate_max, tsfresh.has_duplicate_max, {}, {}),
    (fe.has_duplicate_min, tsfresh.has_duplicate_min, {}, {}),
    (
        fe.index_mass_quantile,
        tsfresh.index_mass_quantile,
        {"q": 0.5},
        {"param": [{"q": 0.5}]},
    ),
    (
        fe.large_standard_deviation,
        tsfresh.large_standard_deviation,
        {"ratio": 0.25},
        {"r": 0.25},
    ),
    (fe.last_location_of_maximum, tsfresh.last_location_of_maximum, {}, {}),
    (fe.last_location_of_minimum, tsfresh.last_location_of_minimum, {}, {}),
    (fe.longest_streak_above_mean, tsfresh.longest_strike_above_mean, {}, {}),
    (fe.longest_streak_below_mean, tsfresh.longest_strike_below_mean, {}, {}),
    (fe.mean_abs_change, tsfresh.mean_abs_change, {}, {}),
    (fe.mean_change, tsfresh.mean_change, {}, {}),
    (
        fe.mean_n_absolute_max,
        tsfresh.mean_n_absolute_max,
        {"n_maxima": 20},
        {"number_of_maxima": 20},
    ),
    (
        fe.number_crossings,
        tsfresh.number_crossing_m,
        {"crossing_value": 0.0},
        {"m": 0.0},
    ),
    (fe.number_peaks, tsfresh.number_peaks, {"support": 5}, {"n": 5}),
    (
        fe.permutation_entropy,
        tsfresh.permutation_entropy,
        {"tau": 1, "n_dims": 3},
        {"tau": 1, "dimension": 3},
    ),
    (fe.root_mean_square, tsfresh.root_mean_square, {}, {}),
    (fe.sample_entropy, tsfresh.sample_entropy, {}, {}),
    (fe.variation_coefficient, tsfresh.variation_coefficient, {}, {}),
]

# =========================================================
# BENCHMARK
# =========================================================

def get_n_range(feature_name: str):

    if feature_name == "approximate_entropy":
        n_range = [10_000]

    elif feature_name in {
        "number_cwt_peaks",
        "sample_entropy",
        "lempel_ziv_complexity",
    }:
        n_range = [10_000, 100_000]

    else:
        n_range = [10_000, 100_000, 1_000_000]

    return [n for n in n_range if n <= MAX_ROWS]


def benchmark(
    f_feat: Callable,
    ts_feat: Callable,
    f_params: dict,
    ts_params: dict,
    is_expr: bool,
):

    n_range = get_n_range(f_feat.__name__)

    if not n_range:
        raise ValueError("No valid benchmark sizes available")

    benchmark_result = perfplot.bench(
        setup=lambda n: (
            DF_PL_EAGER.head(n),
            DF_PANDAS.head(n),
        ),
        kernels=[
            lambda x, _y: (
                x.select(
                    f_feat(pl.col("value"), **f_params)
                )
                if is_expr
                else f_feat(x["value"], **f_params)
            ),
            lambda _x, y: ts_feat(y["value"], **ts_params),
        ],
        n_range=n_range,
        equality_check=False,
        labels=["functime", "tsfresh"],
    )

    return benchmark_result


# =========================================================
# ALL BENCHES
# =========================================================

def all_benchmarks(params: list[tuple], is_expr: bool):

    rows = []

    for x in params:

        f_feat = x[0]

        try:

            print(f"Running: {f_feat.__name__}")

            bench = benchmark(
                f_feat=f_feat,
                ts_feat=x[1],
                f_params=x[2],
                ts_params=x[3],
                is_expr=is_expr,
            )

            for i, n in enumerate(bench.n_range):

                functime_ms = bench.timings_s[0][i] * 1000
                tsfresh_ms = bench.timings_s[1][i] * 1000

                rows.append(
                    {
                        "Feature name": f_feat.__name__,
                        "n": n,
                        "functime (ms)": functime_ms,
                        "tfresh (ms)": tsfresh_ms,
                        "diff (ms)": functime_ms - tsfresh_ms,
                        "diff %": (
                            100 * (functime_ms - tsfresh_ms) / tsfresh_ms
                            if tsfresh_ms != 0
                            else None
                        ),
                        "speedup": (
                            tsfresh_ms / functime_ms
                            if functime_ms != 0
                            else None
                        ),
                    }
                )

        except Exception as exc:
            print(
                f"SKIP {f_feat.__name__}: "
                f"{exc.__class__.__name__}: {exc}"
            )

    return pl.DataFrame(rows)


# =========================================================
# PRETTIFIER
# =========================================================

def table_prettifier(df: pl.DataFrame, n: int):

    filtered = df.filter(pl.col("n") == n)

    if filtered.is_empty():
        return filtered

    return (
        filtered
        .drop("n")
        .sort("speedup", descending=True, nulls_last=True)
        .with_columns(
            pl.col("speedup").round(2).cast(pl.String).alias("speedup")
        )
        .with_columns(
            pl.lit("x ") + pl.col("speedup")
        )
    )


# =========================================================
# RUN
# =========================================================

bench_expr = all_benchmarks(
    params=FUNC_PARAMS_BENCH,
    is_expr=True,
)

bench_series = all_benchmarks(
    params=FUNC_PARAMS_BENCH,
    is_expr=False,
)

# Lazy
df_expr_10k = table_prettifier(bench_expr, 10_000)
df_expr_100k = table_prettifier(bench_expr, 100_000)
df_expr_1m = table_prettifier(bench_expr, 1_000_000)

# Eager
df_series_10k = table_prettifier(bench_series, 10_000)
df_series_100k = table_prettifier(bench_series, 100_000)
df_series_1m = table_prettifier(bench_series, 1_000_000)

# =========================================================
# SAVE
# =========================================================

df_expr_10k.write_csv("expr_10k.csv")
df_expr_100k.write_csv("expr_100k.csv")
df_expr_1m.write_csv("expr_1m.csv")



print("Done.")