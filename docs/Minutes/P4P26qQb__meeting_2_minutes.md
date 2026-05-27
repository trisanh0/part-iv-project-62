# Meeting Minutes: Project #62 - Time-Series ML Model Selection Framework

## Project Progress & Technical Exercises
- Completed readings and initial assignments provided by the supervisor.
- Finished the coffee bean analysis and walking/running exercise.
- Noted the difference in feature space: coffee bean data is fixed in space and time, while walking/running data varies significantly in cadence.
## Python Development & Koans
- Completed approximately 60% of the Python Koans, currently at the "Classes" section.
- Metaclasses are identified as an important upcoming layer of abstraction.
- Importance of classes noted for handling class differences within a feature matrix.
- Adopted "Tracer Bullet Development" strategy: prioritize creating a working end-to-end pipeline as quickly as possible.
- Avoid manual copying between notebooks; reassess workflow if redundancy occurs.
## Computing Resources (Nectar vs. NeSI)
- **Nectar:** A self-service Virtual Machine (VM) where specific hardware specs can be requested. Access is via UPI and University sign-in.
- **NeSI:** A shared cluster where jobs are queued and results are returned.
## Literature Review & Feature Extraction
- `tsfresh` is established; current task is to identify its predecessors and newer improvements.
- Observation: Newer libraries are fast but often omit the feature extraction process.
- Project goal: Analyze the performance and efficacy of these extraction processes.
- Identified potential research direction: `tsfresh` relies on Fourier coefficients; GPU support should be investigated.
- Proposed technical improvement: Replace (Mann Whitney) U test (which compares medians) with the Cramer-Von Mises test to better handle non-normal distributions or those with long tails.
## Tooling & `contexere`
- Using the new `nxt` command with the `--project` option to initialize the working environment.
- The environment includes a Makefile to manage dependencies and custom modules via `make create_environment`.
- Modules installed via this method are importable within Jupyter.
## Domain Applications & Model Robustness
- Time-series ML assumes samples are drawn from the same distribution; robust to small variations but requires retraining for large changes (e.g., changing from running to hopping).
- Previous testing confirmed that a model trained on one person’s data (using 2 selected sensors and 20 features) worked on a different person.
- Discussion of MLOps: Necessity of monitoring for model drift, regular QA, and A/B testing.
- Industry examples:
    - **Exoskeletons:** Anticipating knee angles using IMU data.
    - **Medical:** Differentiating spine injury recovery stages in rats.
    - **Predictive Maintenance:** Monitoring elevators, noting that repairs can "kill" a dataset by changing the baseline.
## Action Items

| Action Item                                                     | Owner          | Timeframe             |
| :-------------------------------------------------------------- | :------------- | :-------------------- |
| Complete Python Koans through to at least the "Classes" section | Scott & Trisan | Before next meeting   |
| Research recent improvements to `tsfresh`                       | Scott & Trisan | Ongoing               |
| Investigate GPU support within existing time-series libraries   | Scott & Trisan | Next literature phase |
| Initialize project folder and environment using `nxt --project` | Scott & Trisan | Immediate             |
| Connect Scott & Trisan with Ivan                                | Andreas        | Post meeting          |

---
# Exhaustive Minutes
- We've been working on reading through the readings and the assignments that he's given us.
- Worked through the coffee bean analysis and walking/running exercise.
- Difference in the two is the feature space - coffee bean analysis is fixed in space and time, whereas the walking and running can vary a lot more in cadence, etc.
- Re: koans. Around ~60% of the way through, to classes. Metreclasses coming up, which are important and are another layer of abstraction. Scott and Trisan to continue our way through the koans, at least to the end of classes.
- Classes are very important, e.g. for a feature matrix, allows us to handle the class difference.
- Tracer bullet development --> trying to make something that works as fast as possible. Make sure not to be copying stuff between notebooks for example, then need to take a step back and reconsider how we are doing things.
- Re: supercomputer. Nectar is a self-service VM, and NeSI is a shared cluster. 
- NeSI you load up your jobs and get the results back.
- For Nectar you can request certain specs for your needs. 
- Access to Nectar is through UPI and uni sign-in.
- Re: literature organisation. `tsfresh` is no longer new, there are new improvements to it. Need to hunt down predecessors of `tsfresh`.
- Impression of new libraries is using fast library, but do not include feature extraction process, which is likely important for the process. This is our job to analyse the performance and efficacy of it.
- Re: contexere: new `nxt` command, which has option `--project`. This creates a project folder for us to work in and play around in.
- Includes a Makefile, which permits `make create_environment` to install Py env specific modules, incl. own modules. 
- E.g. if you have Jupyter running, can import using `import ...`.
- Q: Looking through `tsfresh`, what other applications are there for it? E.g. we do run/walk, if we wanted to apply it to someone else's data, would it work? E.g. people who do race walking, how would it work for them? E.g. for a medical case, will it have to keep being updated?
- A: For all TS ML, always assumes all samples are drawn from the same dist. Robust to small variations, but not large ones. Have to retrain with big changes, e.g. hopping. If training in Scott's data, it should work on Trisan's. We've tested that, and it works. Had 10 sensors, downselected to 2 sensors. Took 20 features, and applied to brand new person, and it worked.
- E.g. for exoskeleton. Need to anticipate next move. Must anticipate angle of knee - need to be generalised/substitute model to take IMU data, then can make an estimation. If you have a set of models, may need to be adjusted on a case-by-case basis.
- E.g. looking at spine injury recovery. Recording spine signals during recovery, can see how recovery works over time, can we differentiate injured/non-injured rats?
- Q: `tsfresh` is pre-trained. What's stopping you from continuing to train the model on new data?
- A: That's always possible, but that comes down to computational effort. Have to set up a monitoring process to see whether the model is still performaing, and have regular update processing.
- MLOps --> regular updating and QA for the model. A/B testing.
- A model can degrade over time, need to capture any drift over time and cache that model.
- E.g. elevators, predictive maintanence. Repairs would introduce new problems, effectively killing your dataset.
- For literature review, pay attention to GPU support. `tsfresh` relies on Fourier coeffs., which may be a good opportunity for a direction to move in.
- Q: `tsfresh` does hypothesis tests, covariance matrix, etc. Any ideas for things to add?
- A: Ivan omitted the Pandas dataframe to improve computational time of `tsfresh`.
- `utest`: if you have a feature, start splitting the feature wrt. diff. classes. `utest` tests that the medians of the cond. dists are diff. 
- However, for TSML, the median may not be relevant. E.g. if you have a ~normal dist, then a dist. with a long tail, hard to compare using median.
- Can try using the Cramer-Van Mises test to improve feature extraction.
- Lecture week six, Monday 12pm-1pm.