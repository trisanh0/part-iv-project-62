
---
## Forecasting and Drift-Bifurcation Simulation
- Scott added forecasting functionality, which currently works on synthetic data only. Trisan will need to integrate this into the codebase with empirical data.
- There is a challenge with evaluating random walks: `tsfresh` tries to pick up on behaviours like how often direction changes, but in many time-series forecasting tasks the only thing that can be forecast is the expected change.
- In a pure random walk, `tsfresh` picks up on perceived behaviour that does not actually exist, so forecasting performance will inherently be poor.
- Instead of direct forecasting, the problem to solve is estimating the parameter that generated the random walk. The `tsfresh` simulated data can be framed as either a regression or classification problem.
- In `driftbif_simulation.py`, the system is governed by parameters $\tau$ and critical threshold $\tau_c$, where velocity is proportional to $\sim\sqrt{\tau - \tau_c}$.
- Velocity is an important dynamic within `tsfresh` from `driftbif_simulation.py`.
- Andreas suggests generating time series with this function, choosing $\tau \ge 1/0.3$ (where $0.3$ is $\kappa_3$).
- Make $\tau$ a random variable sampled from $[1/0.3, 4]$ across e.g. 1,000 samples, generating a time series for each sample.
- For a given value of $\tau$, the resulting random walk exhibits distinct characteristics, such as how often it changes direction.
- This is suggested as a benchmark for time-series extrinsic regression, where the model tries to estimate the underlying $\tau$ parameter.
- For a classification problem, the task can be framed as predicting whether $\tau$ is smaller or larger than the critical parameter $\tau_c$.
- `tsfresh` includes a dedicated feature set designed specifically for this problem.
- Using this data allows for a dedicated report section on physical interpretation (as described in the docstring), representing a random walk transitioning from Brownian motion on the LHS to active Brownian motion on the RHS.

---

## Report Writing and Narrative Structure
- Andreas will share the report template in Overleaf.
- Regarding narrative vs. objective style: focus primarily on the final product and framework functionality if it works as expected.
- If certain components do not work, switch to a narrative-based approach to explain what was attempted, providing clear context for future students to build upon.
- Focus on what needs to be done and clearly explain the design decisions made.
- Include interesting and exploratory ideas in the future outlook section.
- Be quantitative when discussing implementations and runtime measurements.
- Use statistical analysis where appropriate, though note its practical limitations for ML evaluation. Consider using packages like `scikit-posthocs` or `bambi` for plots.

---

## Literature Review and Discussion
- The literature review should cover existing methods used to compare performance as-is, summarising them and describing their shortfalls.
- In the discussion, remain critical of the work done, highlighting areas for improvement and points where walls were hit.
- Always identify shortfalls in the analysis. For example, when benchmarking with the drift-bifurcation dataset, note that `tsfresh` only uses one type of feature for this problem, whereas a realistic problem requires much more data.
- Ensure the discussion directly addresses the opportunities outlined in the literature review.

---

## Methodology and Reproducible Research
- Describe how the project was set up for reproducible research in the methodology section, including reproducibility reviews.
- Explain the role of `contexere` in supporting reproducible research workflows.
- For `contexere` figures, include the tracking link alongside the image so that each plot can be traced back to its source.

---

## Action Items
- Integrate forecasting module into the codebase with empirical data | Owner: Trisan | Timeframe: Next Sprint
- Generate synthetic benchmark dataset using `driftbif_simulation.py` ($\tau \in [1/0.3, 4]$, 1,000 samples) | Owner: Scott & Trisan | Timeframe: Next Sprint
- Share Overleaf report template | Owner: Andreas | Timeframe: ASAP
- Draft literature review covering existing benchmarking methods and their shortfalls | Owner: Trisan & Scott | Timeframe: Next Sprint
- Add Contexere traceability links next to figures and draft reproducibility section | Owner: Trisan & Scott | Timeframe: Next Sprint

---

## Unprocessed

- added forecasting, currently synthetic only. will need trisan to integrate into the code with his data.
- there’s a challenge. tsfresh in this example gets an idea of how often the direction is changes. in many ts forecasting, only thing that can be forecast is the expected change.
- tsfresh is picking up on a behaviour, when that behaviour doesn’t actually exist since it’s a random walk.
- due to the fact that it is a random walk, it will always be pretty bad.
- the problem that you are trying to solve is finding the parameter that generated the random walk. the tsfresh simulated data can either be a classification problem or regression problem.
- there is a parameter $\tau$ and $\tau_C$, which is then proportional to $\sim\sqrt{\tau - \tau_c}$ .
- within tsfresh, velocity is important from `driftbif_simulation.py`.
- should create a few ts with this function, then choose the parameter $\tau$ should be chosen equal to or larger than 1/0.3 (where 0.3 is $\kappa_3$).
- sample from this ts dist. make tau a rand. var., choosing from [1/0.3 to 4]. sample e.g. 1000 values. for each sample, create a time series. 
- for one specific value of tau, the random walk should have some characteristics, e.g. how often is changes direction.
- this is what i would suggest to benchmark a time series regression problem
- if we use this data, you can basically have a nice section about the physical interpretation of that data - see the docstring. it is the random walk of something something, where the LHS is brownian motion, and then RHS is active brownian motion.
- ----- / / / / / 
- the time series should try to estimate the tau parameter. it is an extrinsic time series. 
- for a classification problem, can estimate smaller or larger than the critical parameter.
- tsfresh has a dedicated feature set for this problem.

- for lit review, cover what is used to compare performance as-is. 
- find, summarise, describe shortfalls.

- andreas to share template in overleaf

- re: telling story vs objective. at the end of the report, you have a framework that has some of the expected functionality. focus on the final product. if you dont have something that works, then that’s when you switch to a narrative based approach, with the intention to have a next geenration of students try further.
- focus on what needs to be done
- focus on design decisions
- have some fun and interesting things in the future outlook.
- when we go forward to the implementations and measuring runtimes, always try to be quantitative.
- try to use statistical analysis as much as possible, which is nice but for ML not really useful. 
- can try posthocs or bambi for plots.

- in discussion, be critical of what you did, things to improve, when you hit a wall.
- always important to identify shortfalls of the analysis. 
- e.g. do benchmarking with this dataset, one criticism is that tsfresh only one type of features for this problem. a more realistic problem would include a lot more data.
- address opportunity outlined in literature review.

- in methodology, describe how the project was set-up as reproducible research, that’s where something like `contexere`would come in.
- reproducible research, reproducibility review,

- re: contexere, have the link on the side of the image so that it can be traced back.