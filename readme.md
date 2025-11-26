# First-Order Gradient Based Functional Regression

This repository contains code described in the working paper entitled "First-Order Gradient Based Functional Regression".
Getting cleaned up, reworked and restructured at the moment.


To be able to run the code, I recommend to create and activate a virtual environment and then run
```` 
pip install -r requirements.txt
````
(requirements.txt might be a little bloated).

## Code:
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
+ `CubicSplines`
+ `BSplines`
+ `TPSplines`

*multivariate smoothers*:
+ `TPSplines`: technically usable for any `d` , really only case `d=2`  considered
+ `TeProductSpline`: takes two uncentered univariate splines, creates basis from row-wise Kronecker.
 Used for estimating margins + interaction (`s(x1) + s(x2) + s(x1, x2)` )
+ `TiProductSpline`: Takes centered Splines, only for interaction

*discrete smoothers*:
+ `RandomEffect`
+ `MRFSmooth`: basis is an indicator matrix for discrete regional units, penalty created from list of spatial polygons as
 neighborhood structure

**bricks.py**: code used for the keras models used in this project, contains the "heart" of this project. Classes used for 
estimation of regression models (very much prototypes rather than finished generally usable classes). 
+ `PenaltyMatrix`: Inherits from tf.keras.Model, needs to be initialized with penalty matrices from smooth terms. Makes use of
 `tf.linalg` for more efficient construction and use of block diagonals and such. Within fitting process, calculates penalty
 that is added to model loss. Is able to handle multiple penalty from interaction terms.
+ `RegressionModel`: Inherits from tf.keras.Model, simply translates Regression as a keras-model. Penalty always included,
 if not desired, needs to be a dummy with zeros. If only a contant parameter is to be estimated, rather than an actual 
 regression model, use ´constant=True´. Initialized with PenaltyMatrix-objects
+ `MainModel`: Takes RegressionModels as input and is the object that actually gets trained. Has own `train_step()`-method for
calculating penalized loss. Distributions are hard-coded at the moment.
+ `LossLog_simplerer`: Inherits from tf.keras.Callback. In spirit an extension of EarlyStopping. Checks whether loss converges,
 once a pre-specified number of epochs does not improve, updates the smoothing parameters as described in paper. Achieves this 
 pretty efficiently and elegantly using np.einsum.