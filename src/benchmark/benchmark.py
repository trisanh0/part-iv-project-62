import warnings
import json

import getml
import numpy as np
import pandas as pd
import polars as pl

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from time import perf_counter
from tsfresh import extract_features
from tsfresh.examples.driftbif_simulation import velocity
from tsfresh.utilities.dataframe_functions import impute
from pathlib import Path
from sklearn.model_selection import train_test_split

from my_selectors import *
from my_extractors import *


class Data:

    def __init__(self, X):
        if isinstance(X, Data):
            X = X.data  # unwrap nested Data
        self.data = np.asarray(X)
        self._cache = {}

    def to(self, representation="numpy"):

        if representation in self._cache:
            return self._cache[representation]

        X = self.data

        ####################################################################
        # NumPy
        ####################################################################

        if representation == "numpy":

            converted = X

        ####################################################################
        # Pandas
        ####################################################################

        elif representation == "pandas":

            converted = pd.DataFrame(X)

        ####################################################################
        # Polars
        ####################################################################

        elif representation == "polars":

            converted = pl.DataFrame(X)

        ####################################################################
        # Long format (univariate)
        #
        # id | time | value
        ####################################################################

        elif representation == "long":

            df = pd.DataFrame(X)

            converted = (
                df.reset_index(names="id")
                  .melt(
                      id_vars="id",
                      var_name="time",
                      value_name="value"
                  )
            )

        ####################################################################
        # Long format (multivariate)
        #
        # Assumes shape
        # (samples, timepoints, channels)
        #
        # id | time | kind | value
        ####################################################################

        elif representation == "long_multivariate":

            if X.ndim != 3:
                raise ValueError(
                    "long_multivariate requires "
                    "(samples,timepoints,channels)"
                )

            rows = []

            for sample in range(X.shape[0]):
                for t in range(X.shape[1]):
                    for channel in range(X.shape[2]):

                        rows.append([
                            sample,
                            t,
                            f"channel_{channel}",
                            X[sample, t, channel]
                        ])

            converted = pd.DataFrame(
                rows,
                columns=[
                    "id",
                    "time",
                    "kind",
                    "value"
                ]
            )

        ####################################################################
        # List of numpy arrays
        #
        # [array(...), array(...)]
        ####################################################################

        elif representation == "list":

            converted = [
                np.asarray(row)
                for row in X
            ]

        ####################################################################
        # List of pandas Series
        ####################################################################

        elif representation == "series":

            converted = [
                pd.Series(row)
                for row in X
            ]

        ####################################################################
        # Nested pandas
        #
        # One Series per cell
        #
        # Used by sktime/aeon
        ####################################################################

        elif representation == "nested":

            converted = pd.DataFrame({

                "signal": [
                    pd.Series(row)
                    for row in X
                ]

            })

        ####################################################################
        # Dict
        ####################################################################

        elif representation == "dict":

            converted = {

                i: row

                for i, row in enumerate(X)

            }

        ####################################################################
        # Tensor
        ####################################################################

        elif representation == "tensor":

            try:
                import torch
            except ImportError:
                raise ImportError(
                    "PyTorch not installed."
                )

            converted = torch.tensor(X)

        ####################################################################
        # Unsupported
        ####################################################################

        else:

            raise ValueError(
                f"Unknown representation '{representation}'"
            )

        self._cache[representation] = converted

        return converted
    

class Dataset:
    X_train: Data
    X_test: Data
    y_train: Data
    y_test: Data

    def __init__(self, X_train, X_test, y_train, y_test):
        self.data_train = Data(X_train)
        self.data_test = Data(X_test)
        self.y_train = Data(y_train)
        self.y_test = Data(y_test)

class Extractor:

    def __init__(
        self,
        name,
        function,
        representation="numpy",
        supervised=False,
        **kwargs
    ):

        self.name = name
        self.function = function
        self.representation = representation
        self.supervised = supervised
        self.cache = {}
        self.kwargs = kwargs

    def extract(self, X_train, X_test, y_train, y_test):

        # Convert to the required representation
        X_train = X_train.to(self.representation)
        X_test = X_test.to(self.representation)

        # Extract features
        if self.supervised:
            X_train, X_test = self.function(
                X_train,
                X_test,
                y_train.data,
                **self.kwargs
            )
        else:
            X_train = self.function(
                X_train,
                **self.kwargs
            )

            X_test = self.function(
                X_test,
                **self.kwargs
            )

        return X_train, X_test, y_train, y_test

    


class Selector:

    def __init__(
        self,
        name,
        function,
        representation="numpy",
        **kwargs
    ):

        self.name = name
        self.function = function
        self.representation = representation
        self.kwargs = kwargs

    def select(self, dataset):

        # Convert to the required representation
        X_train = dataset.data_train.to(self.representation)
        X_test = dataset.data_test.to(self.representation)
        # Feature selection is fit on training targets only. For forecasting,
        # selectors designed for scalar supervised targets use the mean of
        # the future horizon as their selection signal. The forecaster still
        # learns the complete multi-step target.
        selection_y = dataset.y_train.data
        selection_kwargs = dict(self.kwargs)

        if np.asarray(selection_y).ndim == 2:
            selection_y = np.mean(selection_y, axis=1)
            if selection_kwargs.get("data_type") == "forecasting":
                selection_kwargs["data_type"] = "regression"

        X_train = self.function(
            X_train,
            selection_y,
            **selection_kwargs
        )

        # align test set to selected features
        X_test = X_test[X_train.columns]

        return Dataset(
            X_train,
            X_test,
            dataset.y_train,
            dataset.y_test
        )
    

import threading
import psutil


class ResourceMonitor:

    def __init__(self, interval=0.1):
        self.interval = interval
        self.process = psutil.Process()

        self.running = False
        self.thread = None

        self.ram_samples = []
        self.cpu_samples = []

        self.gpu_util_samples = []
        self.gpu_memory_samples = []

        self.gpu_available = False
        self.gpu_handles = []

        # --------------------------------------------------
        # Try to initialise NVIDIA GPU monitoring
        # --------------------------------------------------

        try:
            import pynvml

            pynvml.nvmlInit()

            self.pynvml = pynvml
            self.gpu_available = True

            for i in range(pynvml.nvmlDeviceGetCount()):
                self.gpu_handles.append(
                    pynvml.nvmlDeviceGetHandleByIndex(i)
                )

        except Exception:
            self.gpu_available = False

    # ------------------------------------------------------
    # Get current process + child processes
    # ------------------------------------------------------

    def _get_process_tree(self):

        processes = [self.process]

        try:
            processes.extend(
                self.process.children(recursive=True)
            )
        except Exception:
            pass

        return processes

    # ------------------------------------------------------
    # RAM
    # ------------------------------------------------------

    def _get_ram_mb(self):

        total = 0

        for process in self._get_process_tree():

            try:
                total += process.memory_info().rss

            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied
            ):
                pass

        return total / (1024 ** 2)

    # ------------------------------------------------------
    # CPU
    # ------------------------------------------------------

    def _get_cpu_percent(self):

        total = 0

        for process in self._get_process_tree():

            try:
                total += process.cpu_percent()

            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied
            ):
                pass

        return total

    # ------------------------------------------------------
    # GPU
    # ------------------------------------------------------

    def _get_gpu_stats(self):

        if not self.gpu_available:
            return 0.0, 0.0

        total_util = 0.0
        total_memory = 0.0

        for handle in self.gpu_handles:

            try:

                utilisation = (
                    self.pynvml
                    .nvmlDeviceGetUtilizationRates(handle)
                )

                memory = (
                    self.pynvml
                    .nvmlDeviceGetMemoryInfo(handle)
                )

                total_util += utilisation.gpu

                total_memory += (
                    memory.used /
                    (1024 ** 2)
                )

            except Exception:
                pass

        return total_util, total_memory

    # ------------------------------------------------------
    # Monitoring loop
    # ------------------------------------------------------

    def _monitor(self):

        # Prime CPU measurement
        for process in self._get_process_tree():

            try:
                process.cpu_percent()

            except Exception:
                pass

        while self.running:

            self.ram_samples.append(
                self._get_ram_mb()
            )

            self.cpu_samples.append(
                self._get_cpu_percent()
            )

            gpu_util, gpu_memory = (
                self._get_gpu_stats()
            )

            self.gpu_util_samples.append(
                gpu_util
            )

            self.gpu_memory_samples.append(
                gpu_memory
            )

            threading.Event().wait(
                self.interval
            )

    # ------------------------------------------------------
    # Start
    # ------------------------------------------------------

    def start(self):

        self.ram_samples = []
        self.cpu_samples = []
        self.gpu_util_samples = []
        self.gpu_memory_samples = []
        self.start_ram_mb = self._get_ram_mb()

        self.running = True

        self.thread = threading.Thread(
            target=self._monitor,
            daemon=True
        )

        self.thread.start()

    # ------------------------------------------------------
    # Stop
    # ------------------------------------------------------

    def stop(self):

        self.running = False
        

        if self.thread is not None:
            self.thread.join()

        return {

            "peak_ram_mb":
                max(self.ram_samples, default=0.0),

            "avg_cpu_percent":
                np.mean(self.cpu_samples)
                if self.cpu_samples
                else 0.0,

            "peak_cpu_percent":
                max(self.cpu_samples, default=0.0),

            "peak_gpu_percent":
                max(self.gpu_util_samples, default=0.0),

            "peak_gpu_memory_mb":
                max(
                    self.gpu_memory_samples,
                    default=0.0
                ),
            "start_ram_mb":
                self.start_ram_mb,

            "peak_ram_mb":
                max(self.ram_samples, default=0.0),

            "peak_ram_increase_mb":
                max(self.ram_samples, default=0.0)
                - self.start_ram_mb,
        }



def evaluate(
    dataset,
    data_type,
    n_extracted_features=0,
    classifier=None,
    regressor=None,
    forecast_lower=0.05,
    forecast_upper=0.95,
):
    if data_type == "classification":

        classifier.fit(
            dataset.data_train.data,
            dataset.y_train.data
        )

        predictions = classifier.predict(
            dataset.data_test.data
        )

        return {
            "accuracy": accuracy_score(dataset.y_test.data, predictions),
            "n_extracted_features": n_extracted_features,
            "n_selected_features": dataset.data_train.data.shape[1],
            "predictions": predictions
        }

    elif data_type == "regression":

        regressor.fit(
            dataset.data_train.data,
            dataset.y_train.data
        )

        predictions = regressor.predict(
            dataset.data_test.data
        )

        rmse = np.sqrt(
            mean_squared_error(
                dataset.y_test.data,
                predictions
            )
        )

        return {
            "rmse": rmse,
            "mae": mean_absolute_error(
                dataset.y_test.data,
                predictions
            ),
            "r2": r2_score(
                dataset.y_test.data,
                predictions
            ),
            "n_extracted_features": n_extracted_features,
            "n_selected_features": dataset.data_train.data.shape[1],
            "predictions": predictions
        }

    elif data_type == "inverse_regression":
        # Inverse regression:
        # The hidden target is the single tau parameter used to generate the
        # complete benchmark replicate for this seed.
        X_train = np.asarray(dataset.data_train.data)
        X_test = np.asarray(dataset.data_test.data)
        y_train = np.asarray(dataset.y_train.data, dtype=float)
        y_test = np.asarray(dataset.y_test.data, dtype=float)

        if y_train.ndim != 1 or y_test.ndim != 1:
            raise ValueError(
                "Inverse-regression tau targets must be one-dimensional."
            )

        regressor.fit(X_train, y_train)
        predictions = np.asarray(regressor.predict(X_test), dtype=float)

        errors = predictions - y_test
        abs_errors = np.abs(errors)

        return {
            "tau_rmse": float(np.sqrt(np.mean(errors ** 2))),
            "tau_mae": float(np.mean(abs_errors)),
            "tau_bias": float(np.mean(errors)),
            "tau_max_absolute_error": float(np.max(abs_errors)),
            "tau_relative_mae_percent": float(
                np.mean(abs_errors / np.maximum(np.abs(y_test), 1e-12)) * 100.0
            ),
            "tau_r2": float(r2_score(y_test, predictions)),
            "mean_true_tau": float(np.mean(y_test)),
            "mean_predicted_tau": float(np.mean(predictions)),
            "n_extracted_features": n_extracted_features,
            "n_selected_features": dataset.data_train.data.shape[1],
            "predictions": predictions,
            "true_tau": y_test,
            "unique_true_tau": np.unique(y_test),
        }

    elif data_type == "forecasting":
        # Direct multi-step forecasting:
        # one model predicts every horizon simultaneously.
        X_train = np.asarray(dataset.data_train.data)
        X_test = np.asarray(dataset.data_test.data)
        y_train = np.asarray(dataset.y_train.data)
        y_test = np.asarray(dataset.y_test.data)

        if y_train.ndim != 2:
            raise ValueError(
                "Forecasting targets must have shape "
                "(n_samples, forecast_horizon)."
            )

        regressor.fit(X_train, y_train)
        point_forecast = np.asarray(regressor.predict(X_test))

        # Random forests expose each tree's prediction.  The empirical
        # distribution across trees gives a simple, model-native
        # uncertainty interval for every future timestep.
        if not hasattr(regressor, "estimators_"):
            raise TypeError(
                "Forecasting uncertainty currently requires a tree ensemble "
                "with an estimators_ attribute (e.g. RandomForestRegressor)."
            )

        tree_forecasts = np.stack(
            [
                np.asarray(tree.predict(X_test))
                for tree in regressor.estimators_
            ],
            axis=0,
        )

        lower = np.quantile(tree_forecasts, forecast_lower, axis=0)
        upper = np.quantile(tree_forecasts, forecast_upper, axis=0)

        # Overall and horizon-wise metrics.
        errors = point_forecast - y_test
        rmse_by_horizon = np.sqrt(np.mean(errors ** 2, axis=0))
        mae_by_horizon = np.mean(np.abs(errors), axis=0)

        covered = (y_test >= lower) & (y_test <= upper)
        coverage_by_horizon = np.mean(covered, axis=0)
        interval_width_by_horizon = np.mean(upper - lower, axis=0)

        return {
            "forecast": point_forecast,
            "forecast_lower": lower,
            "forecast_upper": upper,
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "mae": float(np.mean(np.abs(errors))),
            "rmse_by_horizon": rmse_by_horizon,
            "mae_by_horizon": mae_by_horizon,
            "interval_coverage": float(np.mean(covered)),
            "interval_coverage_by_horizon": coverage_by_horizon,
            "interval_width": float(np.mean(upper - lower)),
            "interval_width_by_horizon": interval_width_by_horizon,
            "n_extracted_features": n_extracted_features,
            "n_selected_features": dataset.data_train.data.shape[1],
            "predictions": point_forecast,
        }

    else:
        raise ValueError(f"Unknown data_type: {data_type}")


def run(dataset, dataset_name, seed, data_type, extractor=None, selector=None, classifier=None, regressor=None, forecast_lower=0.05, forecast_upper=0.95):
    """
    Generic benchmark pipeline.
    """

    total_start = perf_counter()

    ############################################################
    # Feature Extraction
    ############################################################

    extract_time = 0.0
    n_extracted_features = None

    if extractor is not None:

        cache_key = (dataset_name, seed)

        if cache_key not in extractor.cache:

            monitor = ResourceMonitor()

            monitor.start()

            t0 = perf_counter()

            X_train, X_test, y_train, y_test = extractor.extract(
                dataset.data_train,
                dataset.data_test,
                dataset.y_train,
                dataset.y_test,
            )

            # tsfresh returns DataFrames
            if hasattr(X_train, "to_numpy"):
                impute(X_train)
                X_train = X_train.to_numpy()

            if hasattr(X_test, "to_numpy"):
                impute(X_test)
                X_test = X_test.to_numpy()

            n_extracted_features = X_train.shape[1]

            dataset = Dataset(
                X_train,
                X_test,
                y_train,
                y_test,
            )

            extract_time = perf_counter() - t0

            extract_resources = monitor.stop()

            extractor.cache[cache_key] = (
                dataset,
                extract_time,
                n_extracted_features,
                extract_resources,
            )

        else:

            dataset, extract_time, n_extracted_features, extract_resources = (
                extractor.cache[cache_key]
            )

    ############################################################
    # Feature Selection
    ############################################################

    select_time = 0.0

    select_time = 0.0

    select_resources = {
        "peak_ram_mb": 0.0,
        "avg_cpu_percent": 0.0,
        "peak_cpu_percent": 0.0,
        "peak_gpu_percent": 0.0,
        "peak_gpu_memory_mb": 0.0,
    }

    if selector is not None and extractor is not None:

        monitor = ResourceMonitor()

        monitor.start()

        t0 = perf_counter()

        dataset = selector.select(dataset)

        select_time = perf_counter() - t0

        select_resources = monitor.stop()

    ############################################################
    # Classification / Regression / Inverse Regression / Forecasting
    ############################################################

    t0 = perf_counter()

    results = evaluate(
        dataset,
        data_type=data_type,
        n_extracted_features=n_extracted_features,
        classifier=classifier,
        regressor=regressor,
        forecast_lower=forecast_lower,
        forecast_upper=forecast_upper,
    )

    predict_time = perf_counter() - t0

    ############################################################
    # Timing
    ############################################################

    total_time = perf_counter() - total_start

    results.update({

        "extract_time": extract_time,

        "extract_peak_ram_mb":
            extract_resources["peak_ram_mb"],

        "extract_avg_cpu_percent":
            extract_resources["avg_cpu_percent"],

        "extract_peak_cpu_percent":
            extract_resources["peak_cpu_percent"],

        "extract_peak_gpu_percent":
            extract_resources["peak_gpu_percent"],

        "extract_peak_gpu_memory_mb":
            extract_resources["peak_gpu_memory_mb"],


        "select_time": select_time,

        "select_peak_ram_mb":
            select_resources["peak_ram_mb"],

        "select_avg_cpu_percent":
            select_resources["avg_cpu_percent"],

        "select_peak_cpu_percent":
            select_resources["peak_cpu_percent"],

        "select_peak_gpu_percent":
            select_resources["peak_gpu_percent"],

        "select_peak_gpu_memory_mb":
            select_resources["peak_gpu_memory_mb"],


        "predict_time": predict_time,

        "total_time": total_time,
    })

    return results




def generate_simulated_regression_dataset(
    n_series=1000,
    series_len=200,
    test_size=0.2,
    seed=42
):
    """
    Generate a synthetic time-series regression dataset.

    Each row of X is one independent time series.
    y is a continuous target determined by several
    properties of the underlying series.
    """

    rng = np.random.default_rng(seed)

    X = np.zeros(
        (n_series, series_len),
        dtype=np.float32
    )

    y = np.zeros(
        n_series,
        dtype=np.float32
    )

    t = np.arange(series_len)

    for i in range(n_series):

        # --------------------------------------------------
        # Latent properties
        # --------------------------------------------------

        freq = rng.uniform(0.02, 0.15)

        phase = rng.uniform(
            0.0,
            2.0 * np.pi
        )

        amplitude = rng.uniform(
            0.5,
            2.0
        )

        trend_slope = rng.uniform(
            -0.02,
            0.02
        )

        noise_std = rng.uniform(
            0.05,
            0.5
        )

        # --------------------------------------------------
        # Components
        # --------------------------------------------------

        signal = amplitude * np.sin(
            2.0 * np.pi * freq * t + phase
        )

        trend = trend_slope * t

        noise = rng.normal(
            0.0,
            noise_std,
            size=series_len
        )

        # Random spikes
        spikes = (
            rng.choice(
                [0.0, 1.0],
                size=series_len,
                p=[0.98, 0.02]
            )
            * rng.normal(
                3.0,
                1.0,
                size=series_len
            )
        )

        X[i] = (
            signal
            + trend
            + noise
            + spikes
        )

        # --------------------------------------------------
        # Properties of the observed series
        # --------------------------------------------------

        mean = np.mean(X[i])
        std = np.std(X[i])

        # Change from first half to second half
        first_half = np.mean(
            X[i, :series_len // 2]
        )

        second_half = np.mean(
            X[i, series_len // 2:]
        )

        trend_measure = second_half - first_half

        # Lag-1 autocorrelation
        autocorrelation = np.corrcoef(
            X[i, :-1],
            X[i, 1:]
        )[0, 1]

        # --------------------------------------------------
        # Continuous regression target
        # --------------------------------------------------

        y[i] = (
            2.0 * amplitude
            + 10.0 * trend_measure
            + 1.5 * std
            + 2.0 * autocorrelation
            + 20.0 * freq ** 2
            + rng.normal(0.0, 0.5)
        )

    # ------------------------------------------------------
    # Train/test split
    # ------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed
    )

    return Dataset(
        X_train,
        X_test,
        y_train,
        y_test
    )


def generate_simulated_dataset(
    n_series=1000,
    series_len=200,
    test_size=0.2,
    seed=42
):
    """
    Generate a synthetic binary time-series classification dataset.

    Returns
    -------
    Dataset
        Compatible with your benchmark pipeline.
    """

    rng = np.random.default_rng(seed)

    X = np.zeros((n_series, series_len), dtype=np.float32)
    y = np.zeros(n_series, dtype=np.int32)

    for i in range(n_series):

        t = np.arange(series_len)

        freq = rng.uniform(0.05, 0.20)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        trend = rng.uniform(-0.01, 0.01) * t
        amplitude = rng.uniform(0.5, 2.0)

        signal = amplitude * np.sin(
            2.0 * np.pi * freq * t + phase
        )

        noise = rng.normal(
            0.0,
            0.20,
            size=series_len
        )

        spikes = (
            rng.choice(
                [0.0, 1.0],
                size=series_len,
                p=[0.98, 0.02]
            )
            * rng.normal(
                3.0,
                1.0,
                size=series_len
            )
        )

        X[i] = signal + trend + noise + spikes

        # Binary label
        y[i] = int(amplitude > 1.25)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    return Dataset(
        X_train,
        X_test,
        y_train,
        y_test
    )



def generate_simulated_inverse_regression_dataset(
    n_series_per_tau=40,
    n_tau_values=5,
    series_len=5000,
    tau_min=3.5,
    tau_max=4.5,
    test_size=0.2,
    seed=42,
    kappa_3=0.3,
    Q=1950.0,
    R=0.0003,
    delta_t=0.05,
):
    """Generate one inverse-regression benchmark replicate.

    For each benchmark seed, generate ``n_tau_values`` different tau values.
    Each tau is used to generate ``n_series_per_tau`` independent time-series
    realisations. The complete dataset is then shared by every
    extractor/selector/predictor combination for that seed.

    Thus, within a seed the methods see exactly the same five tau values, the
    same simulated series, and the same train/test split.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if tau_min >= tau_max:
        raise ValueError("tau_min must be smaller than tau_max.")
    if n_tau_values < 2:
        raise ValueError("n_tau_values must be at least 2.")
    if n_series_per_tau < 2:
        raise ValueError("n_series_per_tau must be at least 2.")
    if series_len <= 0:
        raise ValueError("series_len must be positive.")

    # One RNG is used only to choose the five tau values for this benchmark
    # seed.  This means:
    #   * a different benchmark seed gets a different set of five taus;
    #   * the same benchmark seed always gets exactly the same five taus;
    #   * the values are generated ONCE, before any extractor/selector/model
    #     combination is run, so every combination on that seed uses the same
    #     five underlying tau values.
    tau_rng = np.random.default_rng(seed)
    tau_values = np.sort(
        tau_rng.uniform(tau_min, tau_max, size=n_tau_values).astype(np.float64)
    )

    # Use a separate RNG stream for the simulated-series seeds so that changing
    # the tau-selection code cannot accidentally change the generated series.
    simulator_rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), 0x5EED])
    )

    n_series = n_tau_values * n_series_per_tau
    X = np.empty((n_series, series_len), dtype=np.float32)
    y = np.empty(n_series, dtype=np.float64)

    simulator_seeds = simulator_rng.integers(
        0,
        np.iinfo(np.int32).max,
        size=n_series,
        dtype=np.int64,
    )

    row = 0
    for tau_value in tau_values:
        for _ in range(n_series_per_tau):
            simulator_seed = int(simulator_seeds[row])

            process = velocity(
                tau=float(tau_value),
                kappa_3=kappa_3,
                Q=Q,
                R=R,
                delta_t=delta_t,
                seed=simulator_seed,
            )

            simulated = np.asarray(process.simulate(series_len))

            if simulated.ndim == 1:
                series = simulated
            elif simulated.ndim == 2 and simulated.shape[1] >= 1:
                series = simulated[:, 0]
            else:
                raise ValueError(
                    "Unexpected output shape from tsfresh velocity simulator: "
                    f"{simulated.shape}"
                )

            if len(series) != series_len:
                raise ValueError(
                    "The velocity simulator returned an unexpected number of "
                    f"time points: {len(series)} != {series_len}."
                )

            X[row] = series.astype(np.float32)
            y[row] = float(tau_value)
            row += 1

    # Stratification ensures every tau value is represented in both train and
    # test sets, while the exact same split is reused by all methods.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    dataset = Dataset(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # Store the five tau values on the Dataset so the benchmark can report
    # exactly which underlying parameters were used for this seed.
    dataset.inverse_regression_tau_values = tau_values.copy()

    # Sanity check: this benchmark is intended to use five shared tau values.
    if len(dataset.inverse_regression_tau_values) != 5:
        raise ValueError(
            "Inverse-regression benchmark must contain exactly five tau values."
        )

    return dataset

def generate_simulated_forecasting_dataset(
    n_series=100,
    series_len=200,
    history_len=100,
    forecast_horizon=20,
    test_size=0.2,
    seed=42,
):
    """
    Generate a genuinely chronological, windowed forecasting dataset.

    For every independent time series, the first ``1-test_size`` fraction is
    the historical/training period and the final ``test_size`` fraction is the
    future/testing period. Sliding windows are then created as:

        X[t] = values[t : t + history_len]
        y[t] = values[t + history_len : t + history_len + forecast_horizon]

    Training targets never come from the future test region, and every test
    target is strictly in the held-out future region. Test histories may use
    the final observations immediately before the forecast origin, which is
    exactly the information a real forecaster would have available.

    Returns
    -------
    Dataset
        X has shape (n_windows, history_len) and y has shape
        (n_windows, forecast_horizon).
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if series_len <= history_len + forecast_horizon:
        raise ValueError("series_len must exceed history_len + forecast_horizon")

    rng = np.random.default_rng(seed)
    split = int(series_len * (1.0 - test_size))

    if split <= history_len:
        raise ValueError("Training period must be longer than history_len")
    if series_len - split < forecast_horizon:
        raise ValueError("Testing period must contain at least forecast_horizon points")

    X_train, y_train = [], []
    X_test, y_test = [], []

    for _ in range(n_series):
        t = np.arange(series_len, dtype=np.float32)

        # Latent dynamics vary between independent series.
        period = rng.uniform(18.0, 45.0)
        amplitude = rng.uniform(0.5, 2.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        trend_slope = rng.uniform(-0.015, 0.015)
        ar_strength = rng.uniform(0.45, 0.90)
        noise_std = rng.uniform(0.05, 0.25)

        seasonal = amplitude * np.sin(2.0 * np.pi * t / period + phase)
        trend = trend_slope * t

        ar = np.zeros(series_len, dtype=np.float32)
        innovations = rng.normal(0.0, noise_std, size=series_len)
        for j in range(1, series_len):
            ar[j] = ar_strength * ar[j - 1] + innovations[j]

        modulation = 1.0 + 0.15 * np.sin(2.0 * np.pi * t / 80.0)

        shock = np.zeros(series_len, dtype=np.float32)
        possible_starts = np.arange(10, series_len - 3)
        for start_shock in rng.choice(
            possible_starts,
            size=rng.integers(0, min(3, len(possible_starts)) + 1),
            replace=False,
        ):
            magnitude = rng.normal(0.0, 1.0)
            shock[start_shock:start_shock + 3] += magnitude * np.array(
                [1.0, 0.6, 0.3], dtype=np.float32
            )

        series = modulation * seasonal + trend + ar + shock

        # Rolling windows whose targets are wholly inside the training period.
        train_last_start = split - history_len - forecast_horizon
        for start in range(train_last_start + 1):
            X_train.append(series[start:start + history_len])
            y_train.append(
                series[start + history_len:start + history_len + forecast_horizon]
            )

        # Rolling windows whose forecast targets are wholly in the held-out
        # future. The history is allowed to cross the train/test boundary,
        # because those observations are known at the forecast origin.
        test_first_start = max(0, split - history_len)
        test_last_start = series_len - history_len - forecast_horizon
        for start in range(test_first_start, test_last_start + 1):
            forecast_start = start + history_len
            if forecast_start < split:
                continue
            X_test.append(series[start:start + history_len])
            y_test.append(series[forecast_start:forecast_start + forecast_horizon])

    return Dataset(
        np.asarray(X_train, dtype=np.float32),
        np.asarray(X_test, dtype=np.float32),
        np.asarray(y_train, dtype=np.float32),
        np.asarray(y_test, dtype=np.float32),
    )


def load_parquet_dataset(
    ts_path,
    target_path,
    test_size=0.2,
    seed=42,
    data_type="regression",
    history_len=100,
    forecast_horizon=20,
):
    """
    Load a TEMPO-standardised dataset.

    For forecasting, the split is chronological *within each series* and
    sliding windows are created after the split. Training targets are wholly
    inside the historical/training region, while every test target is wholly
    inside the held-out future region.

    Forecasting currently uses the target series itself as the input history.
    This is intentional: future target values are unknown at prediction time,
    so the benchmark must not accidentally use future values as predictors.
    """

    if data_type == "forecasting":
        ts = pl.read_parquet(ts_path)
        targets = pl.read_parquet(target_path)

        if "id" not in ts.columns or "time" not in ts.columns:
            raise ValueError("Time-series parquet must contain 'id' and 'time' columns")
        if "id" not in targets.columns or "target" not in targets.columns:
            raise ValueError("Target parquet must contain 'id' and 'target' columns")
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")

        X_train, y_train, X_test, y_test = [], [], [], []

        ids = ts["id"].unique().sort().to_list()
        for sample_id in ids:
            sample = ts.filter(pl.col("id") == sample_id).sort("time")
            target_table = targets.filter(pl.col("id") == sample_id)
            if "time" in target_table.columns:
                target_table = target_table.sort("time")
            target = (
                target_table.select("target")
                            .to_numpy()
                            .ravel()
            )

            if len(target) != sample.height:
                raise ValueError(
                    f"Series {sample_id} has {sample.height} time points but "
                    f"{len(target)} target values. They must align."
                )

            series_len = len(target)
            split = int(series_len * (1.0 - test_size))
            if split <= history_len:
                raise ValueError(
                    f"Series {sample_id} training period is too short for history_len={history_len}"
                )
            if series_len - split < forecast_horizon:
                raise ValueError(
                    f"Series {sample_id} future period is too short for "
                    f"forecast_horizon={forecast_horizon}"
                )

            train_last_start = split - history_len - forecast_horizon
            for start in range(train_last_start + 1):
                X_train.append(target[start:start + history_len])
                y_train.append(
                    target[start + history_len:start + history_len + forecast_horizon]
                )

            test_first_start = max(0, split - history_len)
            test_last_start = series_len - history_len - forecast_horizon
            for start in range(test_first_start, test_last_start + 1):
                forecast_start = start + history_len
                if forecast_start < split:
                    continue
                X_test.append(target[start:start + history_len])
                y_test.append(target[forecast_start:forecast_start + forecast_horizon])

        return Dataset(
            np.asarray(X_train, dtype=np.float32),
            np.asarray(X_test, dtype=np.float32),
            np.asarray(y_train, dtype=np.float32),
            np.asarray(y_test, dtype=np.float32),
        )

    # Existing classification/regression loader.

    ts = pl.read_parquet(ts_path)
    targets = pl.read_parquet(target_path)

    feature_columns = [
        c for c in ts.columns
        if c not in ("id", "time")
    ]

    X = []
    y = []

    ids = ts["id"].unique().sort().to_list()

    for sample_id in ids:

        sample = (
            ts.filter(pl.col("id") == sample_id)
              .sort("time")
        )

        features = sample.select(feature_columns).to_numpy()

        # flatten multivariate signals
        X.append(features.reshape(-1))

        target = (
            targets
            .filter(pl.col("id") == sample_id)
            .select("target")
            .to_numpy()
            .ravel()
        )

        # if one label per timestep,
        # use the final label
        y.append(target[-1])

    X = np.asarray(X)
    y = np.asarray(y)

    split_kwargs = dict(
        test_size=test_size,
        random_state=seed,
    )
    if data_type == "classification":
        split_kwargs["stratify"] = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, **split_kwargs
    )

    return Dataset(
        X_train,
        X_test,
        y_train,
        y_test,
    )



if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    NO_SEEDS = 1
    # Benchmark category: classification, regression, forecasting, or
    # inverse_regression.
    DATA_TYPE = "classification"

    ############################################################
    # Classifier
    ############################################################

    classifier = RandomForestClassifier(
        random_state=0
    )

    regressor = RandomForestRegressor(
    n_estimators=200,
    random_state=0)

    ############################################################
    # Feature Extractors
    ############################################################

    extractors = [

    # Extractor(
    #     name="Statistics",
    #     function=statistical,
    #     representation="numpy"
    # ),

    # Extractor(
    #     name="TSFresh-Minimal",
    #     function=tsfresh_extractor,
    #     representation="long",
    #     parameter_set="minimal",
    # ),

    # Extractor(
    #     name="TSFresh-Efficient-10FFT",
    #     function=tsfresh_extractor,
    #     representation="long",
    #     parameter_set="efficient",
    #     fft_coefficients=10,
    # ),

    # Extractor(
    #     name="TSFresh-Efficient-25FFT",
    #     function=tsfresh_extractor,
    #     representation="long",
    #     parameter_set="efficient",
    #     fft_coefficients=25,
    # ),

    # Extractor(
    #     name="TSFresh-Efficient-50FFT",
    #     function=tsfresh_extractor,
    #     representation="long",
    #     parameter_set="efficient",
    #     fft_coefficients=50,
    # ),

    # Extractor(
    #     name="TSFresh-Efficient-75FFT",
    #     function=tsfresh_extractor,
    #     representation="long",
    #     parameter_set="efficient",
    #     fft_coefficients=75,
    # ),

    # Extractor(
    #     name="TSFresh-Efficient-100FFT",
    #     function=tsfresh_extractor,
    #     representation="long",
    #     parameter_set="efficient",
    #     fft_coefficients=100,
    # ),

    # Extractor(
    #     name="TSFresh-Comprehensive-10FFT",
    #     function=tsfresh_extractor,
    #     representation="long",
    #     parameter_set="comprehensive",
    #     fft_coefficients=10,
    # ),

    # Extractor(
    #     name="TSFresh-Comprehensive-25FFT",
    #     function=tsfresh_extractor,
    #     representation="long",
    #     parameter_set="comprehensive",
    #     fft_coefficients=25,
    # ),

    # Extractor(
    #     name="TSFresh-Comprehensive-50FFT",
    #     function=tsfresh_extractor,
    #     representation="long",
    #     parameter_set="comprehensive",
    #     fft_coefficients=50,
    # ),

    # Extractor(
    #     name="TSFresh-Comprehensive-75FFT",
    #     function=tsfresh_extractor,
    #     representation="long",
    #     parameter_set="comprehensive",
    #     fft_coefficients=75,
    # ),

    # Extractor(
    #     name="TSFresh-Comprehensive-100FFT",
    #     function=tsfresh_extractor,
    #     representation="long",
    #     parameter_set="comprehensive",
    #     fft_coefficients=100,
    # ),

    # Extractor(
    # name="TSFEL",
    # function=tsfel_extractor,
    # representation="numpy"
    # ),

    Extractor(
    name="TSFeatures",
    function=tsfeatures_extractor,
    representation="numpy"
    ),
    ]

    ############################################################
    # Feature Selectors
    ############################################################

    selectors = [

    # None,

    # Selector(
    #     name="TSFresh",
    #     function=tsfresh_selector,
    #     representation="pandas",
    #     data_type=DATA_TYPE
    # ),

    Selector(
        name="SelectKBest-MI",
        function=select_k_best,
        representation="pandas",
        data_type=DATA_TYPE
    ),

    # Selector(
    #     name="F-Test",
    #     function=f_test_selector,
    #     representation="pandas",
    #     data_type=DATA_TYPE
    # ),

    # Selector(
    #     name="VarianceThreshold",
    #     function=variance_threshold_selector,
    #     representation="pandas",
    #     data_type=DATA_TYPE
    # ),

    # Selector(
    #     name="ExtraTrees",
    #     function=extra_trees_selector,
    #     representation="pandas",
    #     data_type=DATA_TYPE
    # ),

    # Selector(
    #     name="L1",
    #     function=l1_selector,
    #     representation="pandas",
    #     data_type=DATA_TYPE
    # ),

    # Selector(
    #     name="RandomForest",
    #     function=random_forest_selector,
    #     representation="pandas",
    #     data_type=DATA_TYPE
    # ),

    # Selector(
    #     name="Boruta",
    #     function=boruta_selector,
    #     representation="pandas",
    #     data_type=DATA_TYPE
    # ),
    ]

    ############################################################
    # Benchmark
    ############################################################

    benchmark_results = []

    if DATA_TYPE == "classification":

        datasets = {

            "SyntheticClassification":
                lambda seed:
                    generate_simulated_dataset(seed=seed),

            # "Predictive Maintenance":
            #     lambda seed:
            #         load_parquet_dataset(
            #             "datasets/predictive/time_series.parquet",
            #             "datasets/predictive/targets.parquet",
            #             seed=seed,
            #         ),

            # "BEED":
            #     lambda seed:
            #         load_parquet_dataset(
            #             "datasets/beed/time_series.parquet",
            #             "datasets/beed/targets.parquet",
            #             seed=seed,
            #         ),
        }

    elif DATA_TYPE == "regression":
        datasets = {
            "Synthetic Regression":
                lambda seed:
                    generate_simulated_regression_dataset(seed=seed),
        }

    elif DATA_TYPE == "inverse_regression":
        datasets = {
            "Synthetic Inverse Regression (Drift-Bifurcation Tau)":
                lambda seed:
                    generate_simulated_inverse_regression_dataset(
                        n_series_per_tau=40,
                        n_tau_values=5,
                        series_len=5000,
                        tau_min=3.8,
                        tau_max=4.5,
                        test_size=0.2,
                        seed=seed,
                    ),
        }

    elif DATA_TYPE == "forecasting":
        datasets = {
            "Synthetic Forecasting":
                lambda seed:
                    generate_simulated_forecasting_dataset(
                        n_series=100,
                        series_len=200,
                        history_len=100,
                        forecast_horizon=20,
                        test_size=0.2,
                        seed=seed,
                    ),

        }

    else:
        raise ValueError(f"Unknown DATA_TYPE: {DATA_TYPE}")

    for dataset_name, dataset_loader in datasets.items():

        for seed in range(NO_SEEDS):

            print(
                f"\n===== {dataset_name} | Seed {seed} ====="
            )

            dataset = dataset_loader(seed)

            if DATA_TYPE == "inverse_regression":
                true_taus = np.asarray(dataset.inverse_regression_tau_values, dtype=float)
                print(
                    "True tau values for this seed: "
                    + ", ".join(f"{tau:.6f}" for tau in true_taus)
                )

            for extractor in extractors:

                for selector in selectors:

                    

                    extractor_name = (
                        extractor.name
                        if extractor is not None
                        else "None"
                    )

                    selector_name = (
                        selector.name
                        if selector is not None
                        else "None"
                    )

                    print(
                        f"Running: "
                        f"{extractor_name} + {selector_name}"
                    )

                    if DATA_TYPE == "forecasting":
                        original_test_history = np.asarray(dataset.data_test.data).copy()
                        original_test_targets = np.asarray(dataset.y_test.data).copy()

                    results = run(
                        dataset=dataset,
                        dataset_name=dataset_name,
                        seed=seed,
                        data_type=DATA_TYPE,
                        extractor=extractor,
                        selector=selector,
                        classifier=classifier,
                        regressor=regressor,
                        forecast_lower=0.05,
                        forecast_upper=0.95,
                    )

                    if DATA_TYPE == "classification":

                        benchmark_results.append({
                            "Dataset": dataset_name,
                            "Seed": seed,
                            "Extractor": extractor_name,
                            "Selector": selector_name,

                            "Extraction Time": results["extract_time"],
                            "Selection Time": results["select_time"],
                            "Prediction Time": results["predict_time"],

                            "N Extracted Features": results.get("n_extracted_features"),
                            "N Selected Features": results.get("n_selected_features"),

                            "Prediction Accuracy": results["accuracy"],
                            "Total Time": results["total_time"]
                        })

                    elif DATA_TYPE == "regression":

                        benchmark_results.append({
                            "Dataset": dataset_name,
                            "Seed": seed,
                            "Extractor": extractor_name,
                            "Selector": selector_name,

                            "Extraction Time": results["extract_time"],
                            "Extraction Peak RAM (MB)": results.get("extract_peak_ram_mb"),
                            "Extraction Avg CPU (%)": results.get("extract_avg_cpu_percent"),
                            "Extraction Peak CPU (%)": results.get("extract_peak_cpu_percent"),
                            "Extraction Peak GPU (%)": results.get("extract_peak_gpu_percent"),
                            "Extraction Peak GPU RAM (MB)": results.get(
                                "extract_peak_gpu_memory_mb"
                            ),

                            "Selection Time": results["select_time"],
                            "Selection Peak RAM (MB)": results.get("select_peak_ram_mb"),
                            "Selection Avg CPU (%)": results.get("select_avg_cpu_percent"),
                            "Selection Peak CPU (%)": results.get("select_peak_cpu_percent"),
                            "Selection Peak GPU (%)": results.get("select_peak_gpu_percent"),
                            "Selection Peak GPU RAM (MB)": results.get(
                                "select_peak_gpu_memory_mb"
                            ),

                            "Prediction Time": results["predict_time"],

                            "N Extracted Features": results.get("n_extracted_features"),
                            "N Selected Features": results.get("n_selected_features"),

                            "RMSE": results["rmse"],
                            "MAE": results["mae"],
                            "R2": results["r2"],

                            "Total Time": results["total_time"]
                        })

                    elif DATA_TYPE == "inverse_regression":

                        benchmark_results.append({
                            "Dataset": dataset_name,
                            "Seed": seed,
                            "Extractor": extractor_name,
                            "Selector": selector_name,

                            "Extraction Time": results["extract_time"],
                            "Extraction Peak RAM (MB)": results.get(
                                "extract_peak_ram_mb"
                            ),
                            "Extraction Avg CPU (%)": results.get(
                                "extract_avg_cpu_percent"
                            ),
                            "Extraction Peak CPU (%)": results.get(
                                "extract_peak_cpu_percent"
                            ),
                            "Extraction Peak GPU (%)": results.get(
                                "extract_peak_gpu_percent"
                            ),
                            "Extraction Peak GPU RAM (MB)": results.get(
                                "extract_peak_gpu_memory_mb"
                            ),

                            "Selection Time": results["select_time"],
                            "Selection Peak RAM (MB)": results.get(
                                "select_peak_ram_mb"
                            ),
                            "Selection Avg CPU (%)": results.get(
                                "select_avg_cpu_percent"
                            ),
                            "Selection Peak CPU (%)": results.get(
                                "select_peak_cpu_percent"
                            ),
                            "Selection Peak GPU (%)": results.get(
                                "select_peak_gpu_percent"
                            ),
                            "Selection Peak GPU RAM (MB)": results.get(
                                "select_peak_gpu_memory_mb"
                            ),

                            "Prediction Time": results["predict_time"],

                            "N Extracted Features": results.get(
                                "n_extracted_features"
                            ),
                            "N Selected Features": results.get(
                                "n_selected_features"
                            ),

                            "Tau RMSE": results["tau_rmse"],
                            "Tau MAE": results["tau_mae"],
                            "Tau Bias": results["tau_bias"],
                            "Tau Max Absolute Error": results[
                                "tau_max_absolute_error"
                            ],
                            "Tau Relative MAE (%)": results[
                                "tau_relative_mae_percent"
                            ],
                            "Tau R2": results["tau_r2"],
                            "True Tau": results["mean_true_tau"],
                            "Mean Predicted Tau": results["mean_predicted_tau"],

                            # Store both the five shared tau values and the exact
                            # tau values for the held-out test series.
                            "True Tau Values": json.dumps(
                                np.asarray(results["unique_true_tau"]).tolist()
                            ),
                            "Test True Tau Values": json.dumps(
                                np.asarray(results["true_tau"]).tolist()
                            ),
                            "Predicted Tau Values": json.dumps(
                                np.asarray(results["predictions"]).tolist()
                            ),

                            "Total Time": results["total_time"]
                        })

                    elif DATA_TYPE == "forecasting":

                        benchmark_results.append({
                            "Dataset": dataset_name,
                            "Seed": seed,
                            "Extractor": extractor_name,
                            "Selector": selector_name,

                            "Extraction Time": results["extract_time"],
                            "Extraction Peak RAM (MB)": results.get(
                                "extract_peak_ram_mb"
                            ),
                            "Extraction Avg CPU (%)": results.get(
                                "extract_avg_cpu_percent"
                            ),
                            "Extraction Peak CPU (%)": results.get(
                                "extract_peak_cpu_percent"
                            ),
                            "Extraction Peak GPU (%)": results.get(
                                "extract_peak_gpu_percent"
                            ),
                            "Extraction Peak GPU RAM (MB)": results.get(
                                "extract_peak_gpu_memory_mb"
                            ),

                            "Selection Time": results["select_time"],
                            "Selection Peak RAM (MB)": results.get(
                                "select_peak_ram_mb"
                            ),
                            "Selection Avg CPU (%)": results.get(
                                "select_avg_cpu_percent"
                            ),
                            "Selection Peak CPU (%)": results.get(
                                "select_peak_cpu_percent"
                            ),
                            "Selection Peak GPU (%)": results.get(
                                "select_peak_gpu_percent"
                            ),
                            "Selection Peak GPU RAM (MB)": results.get(
                                "select_peak_gpu_memory_mb"
                            ),

                            "Prediction Time": results["predict_time"],

                            "N Extracted Features": results.get(
                                "n_extracted_features"
                            ),
                            "N Selected Features": results.get(
                                "n_selected_features"
                            ),

                            "Forecast RMSE": results["rmse"],
                            "Forecast MAE": results["mae"],
                            "Interval Coverage": results["interval_coverage"],
                            "Mean Interval Width": results["interval_width"],

                            # Store complete forecast outputs for downstream
                            # visualisation. JSON preserves the (samples,
                            # forecast_horizon) structure inside the CSV.
                            "History Values": json.dumps(
                                original_test_history.tolist()
                            ),
                            "True Values": json.dumps(
                                original_test_targets.tolist()
                            ),
                            "Predictions": json.dumps(
                                np.asarray(results["forecast"]).tolist()
                            ),
                            "Prediction Lower": json.dumps(
                                np.asarray(results["forecast_lower"]).tolist()
                            ),
                            "Prediction Upper": json.dumps(
                                np.asarray(results["forecast_upper"]).tolist()
                            ),

                            "Total Time": results["total_time"]
                        })

                # --------------------------------------------------------
                # Free this extractor's cached feature matrix before
                # moving to the next extractor.
                # --------------------------------------------------------

                if extractor is not None:
                    extractor.cache.clear()

    ############################################################
    # Results
    ############################################################

    results_df = pd.DataFrame(benchmark_results)

    print("\n")
    print(results_df)

    index = 0

    while True:
        filename = Path(f"benchmark_results{index}.csv")

        if not filename.exists():
            results_df.to_csv(filename, index=False)
            break

        index += 1


    print(f"\nSaved benchmark_results{index}.csv")




