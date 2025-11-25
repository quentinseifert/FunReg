# First-Order Gradient Based Functional Regression

This repository contains code described in the working paper entitled "First-Order Gradient Based Functional Regression".
Getting cleaned up, reworked and restructured at the moment.


**sim_code**: contains code necessary for the simulation
+ simulation_fcts.py: functions for generating data and fitting models
+ sim_datasets.py: generates datasets, saves them in *sim_data/*
+ sim_study.py: runs the estimations, saves the results in *sim_results/*

**deepdl/**: contains codes for smoothers used in this project (and more). Will be refined  
and written as its own repo soon

The smoothers are implemented as simple class-prototypes which could already find application outside
of this specific project. They closely follow the definition in Simon Wood's "Generalized Additive Models" (2017) and
the according \texttt{mgcv} implementation.

*Univariate smoothers*: 
+ CubicSplines
+ BSplines
+ TPSplines

*multivariate smoothers*:
+ TPSplines: technically usable for any `d` , really only case `d=2`  considered
+ TeProductSpline: takes two uncentered univariate splines, creates basis from row-wise Kronecker.
 Used for estimating margins + interaction (`s(x1) + s(x2) + s(x1, x2)` )
+ TiProductSpline: Takes centered Splines, only for interaction

*discrete smoothers*:
+ RandomEffect
+ MRFSmooth: basis is an indicator matrix for discrete regional units, penalty created from list of spatial polygons as
 neighborhood structure

**bricks.py**: code used for the keras models used in this project
