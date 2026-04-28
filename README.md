# Network Intrusion Detection ML - Group 14

## Overview
Machine Learning project for Network Intrusion Detection using EDA and preprocessing.

## Project Structure


EDA & PD/
├── src/
│   └── eda_preprocessing.py
├── dataset/
│   └── cleaned_ids_dataset.csv
└── images/
    ├── attack_distribution.png
    └── heatmap.png


## Requirements

pip install pandas numpy matplotlib seaborn 

## Libary
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import VarianceThreshold


## Run

python src/eda_preprocessing.py


## Dataset Source
https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset/code

## Outputs
- Data preprocessing
- Feature analysis
- Attack distribution visualization
- Correlation heatmap
