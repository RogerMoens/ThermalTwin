#!/bin/bash

ENV_NAME="weather_nn"

echo "Creating conda environment: $ENV_NAME"

# check if environment exists
if conda env list | grep -q "$ENV_NAME"; then
    echo "Environment already exists"
else
    conda env create -f environment.yml
fi


echo ""
echo "Activating environment..."

source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate $ENV_NAME


echo ""
echo "Installing Jupyter kernel..."

python -m ipykernel install \
    --user \
    --name $ENV_NAME \
    --display-name "Python (weather_nn)"


echo ""
echo "Done!"
echo ""
echo "Activate with:"
echo "conda activate $ENV_NAME"
