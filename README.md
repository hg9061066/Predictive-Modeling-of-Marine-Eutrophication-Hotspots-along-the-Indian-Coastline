# Predictive Modeling of Marine Eutrophication Hotspots along the Indian Coastline

## 1. Project Overview

This repository contains the complete codebase and documentation for a machine learning project focused on identifying, analyzing, and forecasting marine eutrophication hotspots in India. Using a comprehensive water quality dataset from 2020-2023, this project builds a predictive framework to enable proactive environmental management.
The project's key deliverables are a **Random Forest classification model with 99.14% accuracy** and a **time-series forecasting module** that serves as an early-warning system for identifying future at-risk locations.

---

## 2. Key Features

-   **Robust Data Preprocessing:** A custom script to clean and structure complex, real-world environmental data with multi-level headers.
-   **High-Accuracy Classification:** A machine learning model that reliably identifies current pollution hotspots.
-   **Predictive Forecasting:** A time-series module to forecast future water quality at high-risk locations for the upcoming year.
-   **Model Explainability:** Analysis of feature importance to identify the primary drivers of marine pollution, pinpointing Dissolved Oxygen (DO) and Biochemical Oxygen Demand (BOD) as the most critical factors.
-   **Reproducible Workflow:** Sequenced scripts to ensure the entire analysis, from data cleaning to forecasting, is transparent and reproducible.

---

## 3. Project Workflow

The system architecture is a sequential pipeline that processes data from its raw state to final, actionable insights.

1.  **Preprocessing:** `preprocess.py` ingests the raw `Merged file.csv` and produces the clean `cleaned_water_quality_data.csv`.
2.  **Model Training:** `train_model.py` trains the Random Forest classifier on the clean data, evaluates its performance, and saves the feature importance plot.
3.  **Forecasting:** `forecast.py` uses the clean data to train linear regression models for high-risk stations and predicts future parameter values, saving trend plots.

---
