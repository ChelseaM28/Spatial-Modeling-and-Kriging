# Spatial-Geostatistics-Analysis

## WIP - Expected completion No Later Than September 20, 2026.

Reproducible spatial correlation modeling and spatial tail dependence analysis pipeline. 
---

Plan: 
1. Model spatial correlation of within-event ground-motion residuals from a real earthquake's station network by fitting an empirical semivariogram and covariance model via weighted least squares, then perform ordinary kriging and validate its calibration through leave-one-out cross-validation (LOOCV) of formal vs. empirical prediction error.
2. Quantify extremal dependence between station pairs using empirical χ(u) statistics across increasing exceedance thresholds, testing whether simultaneous extreme ground-motion events remain correlated in the tail or decouple toward the asymptotic independence a Gaussian covariance model predicts.
3. Construct an entirely reproducible pipeline with industry-standard file structure, dependency control, pytests, configuration settings, and object-oriented structures.

---

## Motivating Publications

Webster & Oliver, "A tutorial guide to geostatistics: Computing and modelling variograms and 
       kriging" (Geoderma, 2013).

Coles, Heffernan & Tawn, "Dependence Measures for Extreme Value Analyses" (Extremes, 1999).

Baker, Jack, and Nirmal Jayaram. Rep. EFFECTS OF SPATIAL CORRELATION OF GROUND MOTION PARAMETERS FOR
       MULTI-SITE SEISMIC RISK ASSESSMENT. USGS, 2008. 

---
PyGMM Documentaion
https://pygmm.readthedocs.io/models.html
---
Chelsea Momoh
Statistics, UC Davis 
