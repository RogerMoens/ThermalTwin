# ThermalTwin

A grey-box digital twin for apartment thermal modelling, temperature prediction, and future climate control.

## Overview

ThermalTwin explores how an apartment can learn its own thermal behaviour using sensor data.

The project combines:
- physical thermal modelling
- machine learning
- system identification

to predict indoor temperature and eventually control the indoor climate.

The long-term goal is to use **Model Predictive Control (MPC)** to optimize actions such as ventilation, heating, or cooling.

## Concept

A simplified view:


Outside conditions
|
v
+----------------+
| Thermal model |
| + ML correction|
+----------------+
|
v
Indoor temperature prediction
|
v
MPC controller
|
v
Climate control


## Current inputs

Initially:

- Indoor temperature
- Outdoor temperature
- Time information

Future inputs:

- Humidity
- CO₂ / occupancy estimation
- Window state
- Solar radiation
- Ventilation state

## Planned features

- [x] Data collection and exploration
- [ ] Temperature forecasting
- [ ] Thermal parameter estimation
- [ ] Grey-box apartment model
- [ ] Hidden state estimation
- [ ] Model Predictive Control (MPC)
- [ ] Smart ventilation control

## Technology

Built with:

- Python
- NumPy
- Pandas
- Scikit-learn

Future:

- PyTorch
- Optimization libraries for MPC

## Motivation

Most home climate systems use fixed rules.

ThermalTwin investigates whether an apartment can learn its own behaviour and make better climate decisions using predictions.

## Status

Early development 🚧
