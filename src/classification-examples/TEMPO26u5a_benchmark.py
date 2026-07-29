import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from time import perf_counter
import warnings
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
        **kwargs
    ):

        self.name = name
        self.function = function
        self.representation = representation
        self.kwargs = kwargs

    def extract(self, X_train, X_test, y_train, y_test):

        # Convert to the required representation
        X_train = X_train.to(self.representation)
        X_test = X_test.to(self.representation)

        # Extract features
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
        # feature selection (TRAIN ONLY)
        X_train = self.function(X_train, dataset.y_train.data)

        # align test set to selected features
        X_test = X_test[X_train.columns]

        return Dataset(
            X_train,
            X_test,
            dataset.y_train,
            dataset.y_test
        )
    


from sklearn.metrics import accuracy_score


from sklearn.metrics import accuracy_score


def evaluate(dataset, classifier):

    classifier.fit(
        dataset.data_train.data,
        dataset.y_train.data
    )

    predictions = classifier.predict(
        dataset.data_test.data
    )

    return {
        "accuracy": accuracy_score(dataset.y_test.data, predictions),
        "n_features": dataset.data_train.data.shape[1],
        "predictions": predictions
    }



from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute


def run(dataset, extractor=None, selector=None, classifier=None):
    """
    Generic benchmark pipeline.

    Parameters
    ----------
    dataset : Dataset
    extractor : Extractor or None
    selector : Selector or None
    classifier : sklearn estimator

    Returns
    -------
    dict
        accuracy
        n_features
        predictions
        extract_time
        select_time
        predict_time
        total_time
    """

    total_start = perf_counter()

    ############################################################
    # Feature Extraction
    ############################################################

    extract_time = 0.0

    if extractor is not None:

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

        dataset = Dataset(
            X_train,
            X_test,
            y_train,
            y_test,
        )

        extract_time = perf_counter() - t0

    ############################################################
    # Feature Selection
    ############################################################

    select_time = 0.0

    if selector is not None and extractor is not None:

        t0 = perf_counter()

        dataset = selector.select(dataset)

        select_time = perf_counter() - t0

    ############################################################
    # Classification
    ############################################################

    t0 = perf_counter()

    results = evaluate(dataset, classifier)

    predict_time = perf_counter() - t0

    ############################################################
    # Timing
    ############################################################

    total_time = perf_counter() - total_start

    results.update({
        "extract_time": extract_time,
        "select_time": select_time,
        "predict_time": predict_time,
        "total_time": total_time,
    })

    return results

import numpy as np
from sklearn.model_selection import train_test_split


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


if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    ############################################################
    # Classifier
    ############################################################

    classifier = RandomForestClassifier(
        random_state=0
    )

    ############################################################
    # Feature Extractors
    ############################################################

    extractors = [

    Extractor(
        name="Statistics",
        function=statistical,
        representation="numpy"
    ),

    Extractor(
        name="TSFresh-Minimal",
        function=tsfresh_extractor,
        representation="long",
        parameter_set="minimal",
    ),

    # Extractor(
    #     name="TSFresh-Efficient-10FFT",
    #     function=tsfresh_extractor,
    #     representation="long",
    #     parameter_set="efficient",
    #     fft_coefficients=10,
    # ),

    Extractor(
        name="TSFresh-Efficient-25FFT",
        function=tsfresh_extractor,
        representation="long",
        parameter_set="efficient",
        fft_coefficients=25,
    ),

    # Extractor(
    #     name="TSFresh-Efficient-50FFT",
    #     function=tsfresh_extractor,
    #     representation="long",
    #     parameter_set="efficient",
    #     fft_coefficients=50,
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

    Extractor(
    name="TSFEL",
    function=tsfel_extractor,
    representation="numpy"
    )
    ]

    ############################################################
    # Feature Selectors
    ############################################################

    selectors = [

    # None,

    Selector(
        name="TSFresh",
        function=tsfresh_selector,
        representation="pandas"
    ),

    Selector(
        name="SelectKBest",
        function=select_k_best,
        representation="pandas"
    ),

    Selector(
    name="Boruta",
    function=boruta_selector,
    representation="pandas"
    )
    ]

    ############################################################
    # Benchmark
    ############################################################

    benchmark_results = []
    NO_SEEDS = 5
    for seed in range(NO_SEEDS):

        print(f"\n===== Dataset {seed} =====")

        dataset = generate_simulated_dataset(
            seed=seed
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

                results = run(
                    dataset=dataset,
                    extractor=extractor,
                    selector=selector,
                    classifier=classifier
                )

                benchmark_results.append({

                    "Dataset": seed,

                    "Extractor": extractor_name,

                    "Selector": selector_name,

                    "Extraction Time": results["extract_time"],

                    "Selection Time": results["select_time"],

                    "Prediction Time": results["predict_time"],

                    "Prediction Accuracy": results["accuracy"],

                    "Total Time": results["total_time"]

                })

    ############################################################
    # Results
    ############################################################

    results_df = pd.DataFrame(benchmark_results)

    print("\n")
    print(results_df)
    while True:
        index = 0
        try:
            results_df.to_csv(
                f"benchmark_results{index}.csv",
                index=False
            )
            break
        except:
            index += 1


    print(f"\nSaved benchmark_results{index}.csv")




