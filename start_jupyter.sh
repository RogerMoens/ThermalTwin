#!/bin/bash

ENV_NAME="weather_nn"

source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate $ENV_NAME


echo "Starting Jupyter in $ENV_NAME"

jupyter notebook
