---
trigger: model_decision
description: To be used when writing any components of a report
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

## 4. Final Research Report Specifications (Official & Binding)
- **Word Count**:
  - **Typical Length**: 12,000 words.
  - **Minimum Threshold**: No shorter than 8,000 words (**all reports must include at least 8,000 words**).
  - **Maximum Threshold**: No longer than 13,000 words.
- **Page Limit**:
  - **Maximum Length**: Must not exceed a total of **30 pages**.
- **Counting Scope (Words and Pages)**:
  - Both words and pages are counted strictly from the **Introduction to the Future Work section inclusive** (Sections 1 through 7).
  - Excludes front matter (Title Page, Abstract, Declaration, Contents, Lists of Figures/Tables, Glossary) and back matter (References, Appendices).
- **Typography & Formatting**:
  - **Font**: Times New Roman.
  - **Font Size**: 12 pt.
- **Verification Tool**: Keep [`report/check_stats.py`](file:///Users/trisanh/Documents/2026/ENGSCI%20700/part-iv-project-62/report/check_stats.py) aligned with these exact bounds.