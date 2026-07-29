## Feature Extraction and Engineering
- scipy's FFT algorithm computes a fixed set of Fourier coefficients, so individual coefficients should not be dropped in isolation; features should be removed in functional clusters or blocks.
- `tsfresh` / `fftfreq` generates over 400 features categorized by function, including standalone metrics (e.g., mean), grouped statistics (e.g., quantiles/median), and sliding-window linear aggregated trends.
- Linear aggregated trends should be removed as a group, though differing window sizes (e.g., window 10 vs window 50) allow establishing a hierarchy to selectively drop specific sub-groups.
- Additional features can be engineered by taking ratios between time series from different physical sources, calculating first derivatives of time series, or taking differences between measurements sharing the same units.
- Scott's current plan is to allow specifying a parameter for the desired number of FFT coefficients.
- Splitting Fourier angles into sine and cosine components can yield valuable additional features for certain applications.

## Benchmark Datasets and Cross-Validation
- Created an expandable shared dataset split into X_train, X_test, y_train, and y_test for each random seed.
- Review the Time-Series Classification Bake-Off methodology (10 repeated CV runs per dataset, with $\ge 100$ hyperparameter optimisation iterations per fold across 100 folds) to design a convenient pipeline configuration interface.

## Data Storage Architecture for NeSI Execution
- Executing ~10,000 experiment combinations on NeSI shares identical feature extraction/selection logic; feature extraction must only be run once overall rather than repeatedly per run.
- Data structure formulation:
  - Let $W = \{w_1, w_2, \dots, w_t\}$ be the set of data windows (adjacent or overlapping).
  - Each window $w_i$ contains a set of time series $w_i = \{z_{i1}, z_{i2}, \dots, z_{i\gamma}\}$.
  - Each time series generates a feature vector $x_{i,j}$ containing ~700 `tsfresh` features split into functional blocks (e.g., Fourier coefficients).
  - Horizontal slices are taken for training/testing samples, while vertical slices select specific feature blocks.
- Pre-compute and store 100% of features in a database/file structure, running feature selection on a subset first before evaluating the full feature matrix.

## Strategy and Next Steps
- Andreas previously used a directory hierarchy with the input time-series type as top-level folders containing database files.
- Avoid premature optimisation; follow tracer bullet development and step back up to articulate high-level goals.
- Prepare a pitch on why time-series machine learning (TSML) is important and review `tsfresh`.

## Action Items
- Review Time-Series Classification Bake-Off methodology and setup configuration schema (Owners: Trisan & Scott | Timeframe: Next meeting / TBD)
- Draft pitch on the importance of TSML and conduct a detailed review of `tsfresh` features (Owners: Trisan & Scott | Timeframe: Next meeting / TBD)
- Prototype tracer-bullet database structure for single-pass feature extraction on NeSI (Owners: Trisan & Scott | Timeframe: Next meeting / TBD)

## Clarifications
- Are the target timeframes for these action items set for the next meeting, or are there specific calendar deadlines?
- Should any of the action items be assigned individually to Trisan or Scott rather than jointly?
# Unprocessed
- choose feature geenration, selecior, classifier, 
- q: 4 categories of each f coeff. i interpreted this as it would be bad to remove an entire cat., e.g. wanted to test one coeffs., take each, instead of e.g. just one real.
- a: how it is implemented is taht it uses FFT algo from scipy. this alg. always computes a certrain amount of fourier coeffs. removing one doesn’t really make sense. you would remove clusters of feature extractors.
- clustrered wrt. funciton. fftfreq gens. >400 features. some are standalion, e.g. mean, then quantiles (which sometimes = median), then linear agg. trend (sliding window, fits lin. regr then returns linregr.). would remove linear aggregated trends together (window of 10 and window of 50 is distinct). can establish a hierarchy to remove one but not theother.
- if you have several ts, mesuresd from diff. physical sources, can makenew features by dividing/making a ratio.
- can compute first deriv. of the time series, could be some information in there.
- if you have lots of measurements iwth the same unit, can compute differences too.
- sm: current idea is to take a number of fft parameters, specify how many that you want.
- phd student found application where splitting the angle into sin and cos gave additional features
- created shared dataset with variety of datasets, can be improved upon and developed upon as we go.
- turned the dataset into x_train and x_test, y_train and y_test for each seed
- take a look at ts classfication bake off. 10 repeated cv, on each dataset. for each of these 100 folds, hyperparameter optimisation for the classification, of which has at least 100 iterations. need a convenient way of configuring this.
- with nesi, if you have 10000 experiements, with every combination of experiement, there are a lot of combos. the feature selection is the same between all of them. come up with a dtabase that can access the faeture for a give sample/window. in the end, you have 100% of features, only need certain rows for training, and certain blocks of features.
- let W be set of windows of data in the dataset. {w1, w2….wt}
- dp. on problem, windows may be adjacent or overlapping, etc.
- for each window, have a set of time series. might be composed of a few time series. w1={z11 to z1_gammea} to wt = {zt1 to z1_gamma}
- fpr each TS, can make a feature vector, x1_1, which would have the 700 tsfresh features.
- within the feature vectors of all the TSs, there are block of specific data, e.g. fourier coeffs.
- we take differnet horiz. slices for training and testing.
- during feature selection, might decide we only want certain samples and features.
- dont want to select several times. 
- run a full run on a subset, then select, then run on the rest.
- for this benchmarking framework, dont want to do feature extraction again.
- akl: i previously decomposing this into directory hierarchy, then used the type of input TS as one directory folder, then within that have a sets of “databases” as files.
- root of all evil is optimsiing too early. follow tracer bullet development….
- from what we have, need to move back up.
- pitch idea about why tsml is important. look at tsfresh