28-05-2026

## Feature Extraction and Polars Optimisation
- Used Polars for complex entropy calculations as it is faster.
- Manually implemented feature extraction in Polars since its implementation lacks a built-in tsfresh feature extraction function.
- Swap out pipeline components on a case-by-case basis if Polars calculations (such as entropy) prove faster.
- Noted that tsfresh contains a built-in time-series classification benchmark function derived from random walk or Brownian motion.

## Parallelisation and Alternative Approaches
- Replaced pandas with numpy. Gained promising results for MinimalFcParameters, with large speed increase and memory efficiency. 
- For EfficientFCParameters, used pandas - numpy translator to generate code on-the-fly. Proved memory efficient but significant overhead with translation and Python calls (too many!)
- Try Ivan's approach of directly calling Python functions using the same SciPy and NumPy without the Pandas wrapper.
- Ensure data indices are properly managed when parallelising the workflow.
- Leverage tsfresh's core design concept where functions run independently across a large number of time-series (e.g., collecting all regression or FFT functions and pushing them to specific single cores).
- Explore Dask for parallelized dataframes, as tsfresh currently only supports a partially parallelized dataframe version.

## Dataset Selection and Benchmarking
- Establish a definitive, standardized benchmark dataset and review it with Andreas.
- Prioritize freezing the dataset and taking the time to describe it thoroughly.
- Focus initially on a single domain, such as binary classification, utilizing a balanced mixture of simulated and real data.
- Consider expanding the dataset later to cover time-series classification and extrinsic regression (where the signal's target value is a noise value).
- Break down massive datasets into time-series models with a maximum of 1,000 rows (e.g., chunking 1 million measurements into 1,000x1,000 samples to capture temporal changes).
- Conduct systematic benchmarking measurements across multiple dimensions on time-series of varying lengths and quantities.
- Look into Jupyter Notebooks for compilation purposes.

## Contexere Framework Integration
- Integrate the updated and more complete Contexere framework into the project workflow.
- Utilize Contexere to automatically assist with naming conventions.
- Set up a framework to test various machine learning features using the Contexere structure, which copies best practices and builds on a cookiecutter project template.

## Fourier Coefficients and Performance Evaluation
- Analyze tsfresh's default setting of 50 Fourier coefficients (which generates 200 features) and benchmark how scaling is affected when increasing to 100 or 200 coefficients.
- Note how scaling behaves with long time-series, where the first 50 coefficients capture the high-frequency components.
- Evaluate using fewer than 50 features, noting that the impact on accuracy varies; low-frequency components contain periodic/seasonal trends (e.g., weather data) while high-frequency components capture small day-to-day fluctuations.
- Acknowledge Scott's test: looping through the class to pull meaningful features achieved 85% accuracy (Polars + scikit-learn feature selection), whereas full tsfresh extraction and selection achieved 96% accuracy.
- Provide a tool within the benchmarking framework to directly compare time complexity against performance.
- Test the framework's ability to drill down from 60,000 features to 20 features and discard the rest, ensuring it remains fast in both the exploratory phase and the restricted 20-feature phase.
- Use the built-in tsfresh function that deletes computationally expensive functions, change coefficient numbers easily, and document how easy it is to subselect features.

## Action Items
- Test Ivan's direct SciPy/NumPy approach and manage indices during parallelisation
- Conduct systematic benchmarking measurements on varying time-series lengths and FFT coefficient counts
- Explore Dask for parallelised dataframes
- Standardise, describe, and freeze the benchmark dataset, then review it with Andreas
- Integrate the Contexere framework structure into the workflow
- Document the ease of feature subselection within the framework

## Unprocessed
updates
- used polars, complex, faster for entropy calculations. 
- what you can do: consider doing systematic measurements on timeseries of different lengths and different numbers.
- when doing these comparisons, need to do it across several things
- difficult was that tsfresh implementation in polars does not have feature extraction function, have to do it manually
- theres a function in tsfresh that makes a time series classification benchmark, taken from random walk or brownian motion.
- might be case by case - e.g if entropy is faster, swap that piece out.

re: pandas to numpy
- try ivans approach, directly calling python functions.
- tsfresh uses scipy and numpy, wrapped with pandas
- ivans approach directly uses the same scipy and numpy
- when parallelising, make sure indices are taken care of.
- when we designed tsfresh, had the idea that we have a huge number of time series, and each of the funcitons can run independently. 
- e.g. can collect all the regression functions, and push them to a single core.
- e.g. can do the same with FFT and push them into another core

re: scott polars
- tsfresh generally faster for smaller things
- not comprehensively faster
- break problem into TS models that have no more than 1000 rows.
- e.g. when my time series sample had 1 mill measurements, chunk it down into 1000x1000 samples
- each time series becomes a time series, but captures how it is changing over time.

- try to come up with a definite benchmark dataset
- look into jupyter notesbooks for compilation.
- tsfresh supports dataframe version which is somewhat parallel.
- explore dask for parallelised dataframes
- one of the most important features are the FFTs. default setting is 50 coefficients - from these, generate 200 features.
- interesting benchmark - what happens if you go from 50 FFT coeffs to 100 to 200 - how does it scale.
- if you have a long time series, how does it scale - first 50 are the hgih frequency ones.
- standardise dataset, run it by andreas

- contexere is now more complete and workable. 
- how to use it and integrate it in the workflow.
- can help with the naming automatically.
- want to set up a framework that tests various machine learning features - can use contexere structure.
- copies the best practices, builds on cookie cutter project

re: fourier coefficients
- can do less than 50 features
- accuracy impact depends
- low freq. components are periodic trends. for example, weather data, low frequency components are the seasonal trends.
- small fluctuations are the high frequency components, day to dat variation.
- low freq. components still do contain meaningful information
- scott tried to loop through the class to pull the meaningful ones
- compared against tsfresh, tsfresh was just better.
- full tsfresh extraction+selection 96%, polars + scikit learn feature selection 85%.
- scott: when we make the whole benchmarking thing, what level of time complexity should we have?
- want to give tool to compare time cmplexity vs performance
- drill down 60000 features to 20 features, just use those and throw out the others. similar to data mining.
- can test if something fast in exploratory phase, but also fast when you just have 20 features.
- freeze the dataset is a key priority.
- take the time to describe the dataset, fine to focus on one thing now, e.g. binary classification
- have a nice mixture of simulated and real data.
- one could focus on TS classification, extrinsic regression (signal, target value is noise value).
- focus on TS, can widen the dataset.

- default in tsfresh is 50 fourier coeffs.
- there is a function that deletes all functions that are computationally expensive
- can look into how these functions are created, can easily change the number.
- make a note on how easy it is to subselect features. 
