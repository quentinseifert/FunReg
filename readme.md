# First-Order Gradient Based Functional Regression

This repository contains described in the working paper entitled "First-Order Gradient Based Functional Regression
". It contains code for generating simulated datasets from different functional settings,
as well as the code for fitting the respective models.


**sim_code**: contains code necessary for the simulation
+ simulation_fcts.py: functions for generating data and fitting models
+ sim_datasets.py: generates datasets, saves them in *sim_data/*
+ sim_study.py: runs the estimations, saves the results in *sim_results/*

**deepdl/**: contains codes for smoothers used in this projects (and more). Will be refined  
and written as its own repo soon

**bricks.py**: code used for the keras models used in this project
