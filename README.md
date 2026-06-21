# ThermalTwin

A grey-box digital twin for apartment thermal modelling, temperature prediction, and future climate control.

## Motivation

Most home climate systems use fixed rules, don't consider thermal behaviour of the system.

ThermalTwin investigates whether an apartment can learn its own behaviour and make better climate decisions using predictions.

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
MPC controller
        |
        v
Climate control
```


## Current inputs

Initially:

- Indoor temperature
- Indoor humidity
- Outdoor temperature
- Outdoor humidty
- Time information

Future inputs:
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

Future:

- PyTorch
- Optimization libraries for MPC
