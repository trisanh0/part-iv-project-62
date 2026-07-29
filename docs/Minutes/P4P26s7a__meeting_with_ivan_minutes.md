# Meeting Minutes: Project #62 — Discussion with Ivan regarding tsfresh and Optimisation

## Project Progress and Planning
- Scott and Trisan have completed the four initial pipelines; the next step is to compare them.
- A conceptual pipeline will be sketched out over the coming week.
- Project outcomes may eventually be fed back into the main `tsfresh` repository.

## tsfresh Internal Architecture and NumPy Integration
- Within `tsfresh`, "runners" handle the execution of feature calculators.
- Ivan’s specific example is applied on a number of features, with 12 currently selected.
- Ivan’s work focuses on removing the pandas DataFrame wrapping to use NumPy directly for feature extraction.
- The `tsfresh` "combiner" applies a single NumPy function during extraction from a pandas DataFrame.
- Examples of feature extraction behavior:
    - **Mean:** Returns a single value.
    - **Fourier Transform:** Computes coefficients (e.g., first 50 coefficients return 100 parts consisting of real and imaginary components).
    - **Transformations:** 140 input variables result in 140 coefficients which then require a reverse transformation.

## Multiprocessing and Parallelisation
- The use of `imap` is critical in multiprocessing to ensure the order of data is preserved, as indexing becomes difficult otherwise.
- Ivan’s current work is optimised for CPUs; GPU acceleration was not investigated.
- GPUs could significantly accelerate features that rely on matrix multiplications (matmuls), such as FFTs or linear fitting on chunked time-series data.
- Not all features can be parallelised as some require bespoke code.
- It is theoretically possible to run multiple pipelines simultaneously to approximate parallelisation.

## Performance Benchmarking and Alternatives
- **Polars vs. Pandas:** Polars is a potential alternative to Pandas that should be tested.
- **Fastpandas:** Another alternative to consider for performance gains.
- System performance may vary; some implementations are iterative and require convergence. Where other implementations might converge, `tsfresh` typically returns `NaN`.
- The goal is to develop a framework that solves problems within a reasonable timeframe.

## Dataset Scaling and Methodology
- To ensure a robust pipeline, datasets must be tested at different scales:
    - **Data Points:** 1,000 vs. 1,000,000 points.
    - **Time Series:** 1,000 vs. 1,000,000 series.
- `tsfresh` is effective at creating a well-defined feature space from time series of varying lengths.
- **Proposed Workflow:**
    1. **Feature Extraction:** Should always generate a binary file. Time and memory usage must be measured.
    2. **Selection:** Use a smaller dataset and classifier, chunking the data down as needed.
- Note: Integrating directly into Scikit-learn is considered inefficient; feature extraction should be isolated from the [unspecified component/downstream process].

## Action Items
- Share Master's thesis | Owner: Ivan | Timeframe: ASAP
- Sketch out conceptual pipeline | Owner: Trisan & Scott | Timeframe: Over the next week
- Compare the 4 completed pipelines | Owner: Trisan & Scott | Timeframe: Before next meeting
- Trial different implementations (one person testing NumPy, one testing Polars) | Owner: Trisan & Scott | Timeframe: Next steps

---
# Unprocessed
- Scott and Trisan completed the 4 pipelines, need to compare.
- Will sketch out conceptual pipeline over the next week.
- Within tsfresh, there are runners, which have both: 
	- Was exxperimenting with the unmber of features.
	- Develop selected features - currently 12 features being selected.
	- Apply feature fn uses import lib and tsfres hfeature calculatros.
	- In tsfresh, when using combiner ,applying one nimpy function once, when extracting from pandas df; if I hae some features, e.g. Fourier Coefficient w/ k = 3, done by the name of the feature. 
	- Using tsfresh reatures;
	- Does not have the wrapping in pd df.
	- feature extraction for tsfresh
		- Mean of TS, mean fn retursn one number.
		- e.g. the fourier tform, which conputes first 50 coeffs, computes real/im parts of the coeffs.
			- returns 100 parts.
		- e.g. 140 input vars, then get 140 coeffs, then have to transform back.
		- combiner does something then extracts the data.
	- using multiprocessing
	- imap is important important; ensures that order doesnt change bc indexing can be difficult w/ multiprocessing.
	- outcome of the project can be the development of something that feeds back into tsfresh.
- ivan to share thesis with scott and trisan. his work:
	- this works for cpu; didnt ivnestigate working with gpus.
	- gpu acceleration when doing matmuls. 
	- for some features, e.g. fft, are just matmuls.
	- e.g. long vector for of data, then fft is matrix. 
	- w/ sequence of TS data, it becomes matmul.
	- TS can be chunked down then fit linearly, which again is just matmul.
	- can defintely be applied to a lot of features.
	- some bespoke code for each feature, at the end of the day, not all of them can be parallelised.
	- in theory, can run multiple pipelines at the same time in a way that approximates parallelisation.
- polars vs pandas?
	- has been sold to me as an alternative.
	- can try it out!
- nextsteps:
	- if one of you can try numpy, one tried polars, then have three
	- also fastpandas.
	- a single system may not be universally faster, as some are iterative and need to converge. tsfresh would return NaN in that case; other implementations would return something else.
	- idea is to have a framework that solves the problem in a reasonable amount of time 
	- worth checking out tsfresh +polars combos.
	- need to make dataset(s) that have differnet cclaes, to ensure robust pipeline
		- different length e..g 1000 vs a mill datapoints
		- different scale e.g. 1000 vs a mill ts
- misc:
	- just use a regular predictor to predict with tsfresh.
	- naive macine learning, have to improvise + impute.
	- tsfresh can work with ts of diff lengths, creates well-defined feature space.
	- feature extraction should always generate binary file with tsml, should measure the ima dn memory usage. second step is selection, smaler dataset + classifier. have to chunk it down.
	- have to incorporate into scikit; but inefficient. should always isolate feature extaction from the ???