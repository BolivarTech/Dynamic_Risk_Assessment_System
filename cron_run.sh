#!/bin/bash

# Activate the conda environment
source /usr/local/bin/anaconda3/bin/activate risk_pipeline
# change current directory to pipeline root 
cd /home/jbolivarg/udacity/Dynamic_Risk_Assessment_System
# Run full pipeline process
python3 fullprocess.py
# Deactivate conda environment
conda deactivate
