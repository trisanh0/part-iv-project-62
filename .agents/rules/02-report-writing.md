---
trigger: model_decision
description: When writing any components of a report
---

# Skill Name: write-report-style
# Description: Defines Trisan Ethan Q Huynh's exact writing voice, prose cadence, sentence syntax, dialect, and tone for report generation.

## 1. Dialect & Regional Rules
- **Language**: Strictly NZ / British English (en-NZ).
- **Spelling**: Use `-ise` / `-isation` variants (`optimise`, `characterise`, `standardise`). Retain British/NZ word forms (`favour`, `behaviour`, `modelling`, `fulfilment`).
- **Units**: Standard SI units with proper spacing ($m/s$, $W/m^2$, $MPa$, $K$). Currency as `NZD$` or `$`.

## 2. Voice, Tone & Perspective
- **Technical Tone**: Objective, authoritative, grounded, and mathematically precise.
- **Reflective Tone**: Authentic, pragmatic, and candid about software friction, trial-and-error, or messy real-world data.
- **Person Split**:
  - *Third-person passive*: Technical methodology, physics setup, data pre-processing, and solver runs ("A computational study was conducted...", "The model was solved using...").
  - *First-person active ("I" / "We")*: Team contributions, personal responsibilities, and reflections ("I was tasked with...", "Scott and I...").
- **Directness on Failure**: Unvarnished and direct when stating model or design failures ("failed catastrophically", "not fit for purpose"). Avoid soft hedging or self-deprecating fluff.

## 3. Sentence Syntax & Cadence
- **Paragraph Flow (Setup -> Action -> Result -> Impact)**: Establish problem context, detail analytical execution, state exact numerical outcomes, and conclude with practical impact.
- **Front-Loaded Sentence Openers**: Frequently begin sentences with introductory participial, infinitive, or prepositional phrases ("To address this issue of scale...", "Given my background in...", "Upon receiving feedback from...").
- **Balanced Trade-Off Clauses**: Frame nuances using contrastive compound structures ("While the ML model offered better predictive accuracy, the linear regression model was more interpretable...").

## 4. Vocabulary & Idioms
- **Verbs**: `yielded`, `characterised`, `demarcated`, `reconciled`, `benchmarked`, `paved the way for`, `delineates`.
- **Key Idioms**: "curse of dimensionality", "single source of truth", "first-principles approach", "failed catastrophically", "not fit for purpose", "get into the weeds of".