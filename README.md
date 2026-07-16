# ThermalTwin

Object-oriented framework for indoor temperature prediction using Netatmo weather station data — a grey-box digital twin for apartment thermal modelling and temperature forecasting, laying the groundwork for a smarter home climate *strategy*.

## Table of Contents

- [Motivation](#motivation)
- [Overview](#overview)
- [Concept](#concept)
- [Models](#models)
- [Current Inputs](#current-inputs)
- [Planned Features](#planned-features)
- [Technology](#technology)
- [Environment Setup](#environment-setup)
- [Results](#results)
- [Status](#status)

## Motivation

Most home climate systems use fixed rules and don't account for the thermal behaviour of the building they operate in. ThermalTwin investigates whether an apartment can learn its own thermal behaviour from sensor data, and use that understanding to inform better climate decisions.

## Overview

ThermalTwin explores how an apartment can learn its own thermal behaviour using sensor data.

The project combines:
- physical thermal modelling
- machine learning
- system identification

to predict indoor temperature and build the foundation for a smarter climate strategy — informing decisions such as when to heat, ventilate, or let a space respond passively to outdoor conditions.

The long-term goal is to use this predictive insight to shape a climate *strategy*, rather than to build a fully automated control system.

## Concept

A simplified view:

```text
Outside conditions
        |
        v
+------------------+
|  Thermal model   |
| + ML correction  |
+------------------+
        |
        v
Indoor temperature prediction
        |
        v
Climate strategy recommendations
```

## Models

All models share a common `BasePredictor` interface (`fit` / `predict` / `save` / `load`).

**Temperature forecasting**
- `LinearOneStepPredictor` — linear regression, single-step-ahead
- `RidgeOneStepPredictor` — L2-regularized regression, single-step-ahead
- `NeuralOneStepPredictor` — small feed-forward network, single-step-ahead
- `LinearWeatherOnlyPredictor` / `RidgeWeatherOnlyPredictor` — regression over a windowed weather-only history
- `NeuralWeatherOnlyPredictor` — LSTM over a windowed weather-only history, multi-step horizon

**Physical modelling**
- `Thermal2R2CPredictor` — grey-box 2R2C (two-resistance, two-capacitance) thermal model, fit via numerical optimization

The project currently implements only this predictive layer — there is no control or strategy-optimization layer yet. That's the planned next direction, described below.

## Current Inputs

Initially:

- Indoor temperature
- Indoor humidity
- Outdoor temperature
- Outdoor humidity
- Time information

Future inputs:
- CO₂ / occupancy estimation
- Window state
- Solar radiation
- Ventilation state

## Planned Features

- [x] Data collection and exploration
- [x] Temperature forecasting
- [x] Thermal parameter estimation (2R2C model)
- [x] Grey-box apartment model
- [x] Hidden state estimation
- [ ] Climate strategy engine (optimization-based recommendations)
- [ ] Ventilation strategy recommendations

## Technology

Built with:

- Python 3.11
- NumPy / pandas / SciPy
- scikit-learn
- TensorFlow / Keras
- Matplotlib / seaborn
- Jupyter

## Environment Setup

This project uses a dedicated Conda environment to keep dependencies isolated from the system Python installation.

### Create the environment

Run:

```bash
./setup_env.sh
```

This will:

- Create the `weather_nn` Conda environment
- Install all required packages
- Register the environment as a Jupyter kernel

### Start Jupyter Notebook

Run:

```bash
./start_jupyter.sh
```

This automatically activates the correct environment and starts Jupyter Notebook.

### Manual activation

If needed, the environment can also be activated manually:

```bash
conda activate weather_nn
```

### Updating dependencies

After adding or changing packages, update the environment:

```bash
conda env update -f environment.yml
```

To save the current working environment:

```bash
conda env export > environment_locked.yml
```

## Results

- For 5-minute-ahead prediction, the previous indoor temperature was already an extremely strong predictor due to the building's thermal inertia.
- A simple persistence model (predicting the next temperature as the current temperature) achieved the best performance, outperforming both regression models and the neural network. This indicates that short-term indoor temperature dynamics are dominated by the current indoor state rather than by external weather conditions.

<p align="center">
  <img src="./reports/2026_06_23_first_results.png" alt="First results" width="600" />
</p>

## Status

The project currently has a working predictive layer only: temperature forecasting models and the 2R2C physical model described in [Models](#models). No control or strategy logic has been implemented yet — the near-term focus is finishing the object-oriented refactor (extracting models out of the exploratory notebooks into the `models/` package) before starting on the climate-strategy layer.
