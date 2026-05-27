# Meeting Minutes: Project #62 — Time-Series Machine Learning Model Selection Framework

## Documentation and Citation
- Refer to the GitHub project when citing `contexere`, as the paper is currently incomplete.
- Andreas will share the latest version of the paper; this must be kept confidential.

## TSML Methodology and tsfresh
- Andreas defaults to `tsfresh` for new Time-Series Machine Learning (TSML) problems due to its superior feature selection capabilities.
- Most alternative packages only perform feature engineering, whereas `tsfresh` handles both engineering and selection.
- There is a key distinction between engineering features and engineering time series themselves.
- Trisan identified a primary opportunity in developing a pipeline that automates these integrated workflows.

## Performance and Optimization of `tsfresh`
- Current feature extraction relies on the CPU.
- Transitioning to C/C++ or using GPUs for operations like linear regression (which is essentially matrix multiplication) on sliding windows could significantly improve performance.
- NumPy's foundation is already in C++.
- Ivan saw significant performance gains by moving away from Pandas, though this created issues with multi-processing and maintaining data order.

## Project Administration and Licensing
- Avoid using packages with GNU licenses.
- Andreas will follow up with Ivan to schedule a meeting for next week.

## TSML Applications
- In order of increasing difficulty:
    1. Classification (Starting point).
    2. Extrinsic Regression (Estimating continuous variables not part of the TS).
    3. Time-Series Regression (Predicting one TS with another; significantly harder).
    4. Forecasting (Predicting future values; the most difficult due to causality constraints and cross-validation complexities).

## Implementation Details for Next Steps
- Every dataset and model combination requires hyperparameter optimization.
- The high number of hyperparameters combined with cross-validation folds leads to high computational requirements.
- Special care is needed during data loading to ensure pre-processing results in a useful format.

## Action Items
- Share latest version of *contexere* | Owner: Andreas | Timeframe: ASAP
- Bump Ivan to arrange meeting | Owner: Andreas | Timeframe: For next week
- Pick 2 UCI ML repository datasets and develop 2 TS classification notebooks (total 4 notebooks for the team) | Owner: Trisan & Scott | Timeframe: Before next meeting
- Identify commonalities between the 4 notebooks to define pipeline requirements | Owner: Trisan & Scott | Timeframe: Following notebook completion


# Unprocessed
- Re: how to city contexere, given that it’s an incomplete paper?
	- To cite contexere, just refer to the GitHub project.
	- Andreas will share the latest version - keep confidential.
- Re: how would Andreas approach a new TSML problem?
	- Would always default to tsfresh due to strength of feature seelction.
	- Key differences between tsfresh and other packages. Most other pacakges only do feature engineering, no selection.
	- Engineering of features vs engineering of time series (tsfresh).
- Trisan: Key opportunity lies in developing a pipeline that can do these sort of things.
- Re: thoughts on changing tsfresh to C/C++?
	- Currenlty, feature extraction only uses CPU, no GPU
	- E.g. linear regression (essentially matmul) on sliding window can be done with GPUs, instead of CPU.
	- NumPy has foundation in C++.
	- Ivan ditched Pandas, gave big performance boost. Problem is w/ multi-processing.
		- Running multiple processes, it’s important to put them back into the order that they came into.
- Arrange meeting with Ivan for next week.
	- Andreas will do this and bump Ivan.
- Avoid GNU licence when looking at other packages.
	- Unsure of reason
- Focus on application of tsfresh on all different TS applications. Start with classification, then extrinsic regression (est. cont. variable not part of the TS), then TS regression (predict aother TS with TS, much harder); hardest of all is forecasting (variative regression, predict future value, -can only use data from the past, makes CV hard.)
- To work on next:
	- Trisan + scott to Pick 2 different TS classification project each form the UCI ML repo, dev two steps:
		- Pick dataset indiividually
		- Dev ML workflow with tsfresh, incl. out of sample testing and hyperparam tuning.
		- Will end up with 4 diff. Jupyter notebooks.
		- Look at what is common between them, to ID what the pipeline needs to have
		- Need to be careful with the loading of data, to pre-process into a useful form.
- Every dataset + model combination needs hyperparam optimisation. With many hyperparams + CV folds, results in 
- 
