
# Overview

This project implements a stock trading bot with DQN and non-linear reward functions.

## Structure of This Folder
run-experiment.py: main file to run for training agents with customized stock and hyperparameter config
data-collection.py: file to collect and export training, validation, and testing csvs to data folder
evaluation.py: file to run for validation and testing with a saved model
result-visualize.ipynb: jupyter notebook to visualize results
data: folder including all training, valid, and testing csvs from data-collection.py
trading-bot: folder including all set ups for RL agents
model: folder that contains sample trained models, ready to be evaluated right away
csv_output: folder that contains sample csv output during testing 

## To Get Started
run data-collection.py to collect desired stock 
run run-experiment.py to train agents
run result-visualize.ipynb to visualize results

OR
use presaved model and run evaluation.py directly to see how the model fits your goal

## References
-https://github.com/pskrunner14/trading-bot