import numpy as np
import pandas as pd
import polars as pl
from tsfresh import extract_features, select_features
from sklearn.ensemble import RandomForestClassifier
from time import perf_counter
import warnings


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
    """
    t2 = perf_counter()
    # Feature extraction
    if extractor is not None:
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
        print(f"EXTRACT = {perf_counter() - t2:.2f}s")

    t3 = perf_counter()
    # Feature selection
    if selector is not None:
        dataset = selector.select(dataset)
        print(f"SELECT = {perf_counter() - t3:.2f}s")
    print(f"EXTR+SELEC = {perf_counter() - t2:.2f}s")

    # Classification
    results = evaluate(dataset, classifier)

    return results


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        module="tsfresh"
    )
    t1 = perf_counter()
    X_train = pd.read_csv("X_train.csv").to_numpy()
    X_test = pd.read_csv("X_test.csv").to_numpy()
    y_train = pd.read_csv("y_train.csv").squeeze().to_numpy()
    y_test = pd.read_csv("y_test.csv").squeeze().to_numpy()
    print(f"IMPORT = {perf_counter() - t1:.2f}s")

    dataset = Dataset(
        X_train,
        X_test,
        y_train,
        y_test,
    )
    extractor = Extractor(
        name="tsfresh",
        function=extract_features,
        representation="long",
        column_id="id",
        column_sort="time"
    )

    t3 = perf_counter()
    selector = Selector(
        name="tsfresh",
        representation="pandas",
        function=select_features,
    )

    classifier = RandomForestClassifier(
        random_state=0
    )

    results = run(
        dataset=dataset,
        extractor=extractor,
        selector=selector,
        classifier=classifier,
    )

    print(f"ACCURACY = {results['accuracy'] * 100:.2f}%")
    print(f"TOTAL TIME = {perf_counter() - t1:.2f}s")




