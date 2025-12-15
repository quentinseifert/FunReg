"""

Contains functions necesarry for simulation studies. The function named "sim_something" create datasets (called in
sim_datasets.py, functions name fit_something fit models based on the simulated data (called in sim_study.py)

"""




import os.path
import pickle
import sys
from pathlib import Path
#sys.path.append(str(Path(__file__).resolve().parent.parent))


import numpy as np
import pandas as pd
from bricks import *
from deepdl.splines import *
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
import time
from scipy.spatial import distance_matrix
from sympy.physics.quantum import TensorProduct



mpl.use("TkAgg")


from tensorflow.keras import backend as K
import gc
#from mat_utils import *
#import torch
import tensorflow.keras.backend as K
#import tensorflow_addons as tfa
tfd = tfp.distributions
#import rpy2.robjects as robjects


import numpy as np


def interaction_effect(x, y, marginal=False):


    x_scaled = x * 10 - 5  # linear mapping


    spacing = 4.0
    sigma = 1.2
    A = 1.0
    drift = 0.3
    alpha = 0.2
    beta = 0.1

    z = 0.0
    # Loop over centers
    for i in np.arange(-5, 6, spacing):
        for j in np.arange(-5, 6, spacing):
            y_shifted = j + drift * i
            sign = (-1) ** ((i // spacing) + (j // spacing))
            z += sign * A * np.exp(-((x_scaled - i) ** 2 + (y - y_shifted) ** 2) / (2 * sigma ** 2))


    z += alpha * x_scaled + beta * y

    return z



def sim_function_lin(N, seed=123):
    np.random.seed(seed)
    x = np.random.normal(0, 1, (N, 1))
    t = np.tile(np.linspace(0, 1, 100), (N, 1))
    func_int = stats.beta(2, 7).pdf(t) + 1
    func_t = stats.norm.pdf(4 * (t - 0.2))
    func_t = (func_t - np.mean(func_t)) / np.std(func_t)
    func_x = x * func_t
    y = func_int + func_x + np.random.normal(0, .5, (N, 100))

    df = pd.DataFrame(np.array([(np.repeat(np.arange(N), 100)), y.flatten(), np.repeat(x, 100), t.flatten()]).T,
                 columns=["id", "y", "x", "t"])
    if not os.path.exists(f"./sim_data/lin{N}/"):
        os.makedirs(f"./sim_data/lin{N}/")
    df.to_csv(f"sim_data/lin{N}/df{seed}.csv")

    return df

def sim_function_smoo(N, seed=123):
    np.random.seed(seed)
    t = np.tile(np.linspace(0, 1, 100), (N, 1))
    func_int = stats.beta(2, 7).pdf(t) + 1
    x = np.random.normal(0, 1, (N, 1))
    func_t = stats.norm.pdf(4 * (t - 0.2))
    func_t = (func_t - np.mean(func_t)) / np.std(func_t)
    func_x = x * func_t

    x2 = np.random.uniform(-5, 5, (N, 1))
    eta = func_int + func_x + interaction_effect(t, x)
    y = eta + np.random.normal(0,.5, (N, 100))
    df = pd.DataFrame(np.array([(np.repeat(np.arange(N), 100)), y.flatten(), np.repeat(x, 100), np.repeat(x2, 100)]).T,
                      columns=["id", "y", "x" ,"x2"])
    if not os.path.exists(f"./sim_data/smoo{N}/"):
        os.makedirs(f"./sim_data/smoo{N}/")
    df.to_csv(f"sim_data/smoo{N}/df{seed}.csv")
    # cleanup

    return pd.DataFrame(np.array([y.flatten(), t.flatten(), np.repeat(x, 100),np.repeat(x2, 100), eta.flatten()]).T, columns=["y", "t", "x", "x2", "eta"])



def sim_function_beta(N, seed=123):
    tmp_data = sim_function_smoo(N)
    z = tf.math.sigmoid(tmp_data.eta - 2.2)
    tmp_data["mu_beta"] = z
    tmp_data["phi_beta"] = 10.

    phi = 10.
    alpha = z * phi
    beta = (1 - z) * phi
    dist = tfd.Beta(concentration1=alpha, concentration0=beta,
                    force_probs_to_zero_outside_support=False,
                    allow_nan_stats=True)
    tmp_data["beta"] = dist.sample()
    if not os.path.exists(f"./sim_data/beta{N}/"):
        os.makedirs(f"./sim_data/beta{N}/")
    tmp_data["id"] = np.repeat(np.arange(N), 100)
    tmp_data.to_csv(f"sim_data/beta{N}/df{seed}.csv")

    return tmp_data



def fit_lin(i, N=100, dummy=False):
    np.random.seed(i)
    data = pd.read_csv(f"sim_data/lin{N}/df{i}.csv", index_col=0)
    start = time.time()
    data["t"] = np.tile(np.linspace(0, 1, 100), N)
    spline = BSplines(data.t, k=20, pen_order=2)
    #spline_x = (spline.transform_new(data.t).T * data.x.to_numpy()).T
    spline_x = BSplines(data.t, k=6, by=data.x.to_numpy()[:, None], pen_order=2)
    design1 = tf.concat([spline.basis, spline_x.basis], 1)
    design2 = tf.ones([data.shape[0], 1])
    pens = [spline.penalty, spline_x.penalty]
    penalty1 = PenaltyMatrix(pens, reg_param=[1000., 1000.])
    penalty2 = PenaltyMatrix([np.array([[0.]])], reg_param=[0.])
    y = tf.cast(np.expand_dims(data.y, 1), "float32")
    design = (tf.cast(design1, "float32"), design2)
    dataset = tf.data.Dataset.from_tensor_slices((design, y)).batch(100)
    logger = LossLog_simplerer(dataset=dataset, huge=False, tolerance=0, outer_max=10, n_samples=5)
    model1 = RegressionModel(penalty1, seed=i)
    model2 = RegressionModel(penalty2, constant=True, seed=i+1000)
    main_model = MainModel([model1, model2], data.shape[0], categorical=False, dist="Normal")
    main_model.compile(optimizer=Adam(0.01), loss=negloglik)
    main_model(design)
    if dummy:
        return main_model, spline, spline_x
    main_model.fit(design, y, epochs=1000, callbacks=[logger], batch_size=int((N * 100) / (100)))
    end = time.time()



    # with open(f"sim_results/lin_{N}_{i}.pkl", "wb") as file:
    #    pickle.dump(main_model.weights, file)
    metrics = [logger.RMSE, logger.ll, logger.AIC, logger.BIC, end - start]
    K.clear_session()
    del main_model
    gc.collect()
    print(logger.outer)
    return metrics


def fit_smooth(i, N=100, forplot=False, dummy=False):
    np.random.seed(i)
    data = pd.read_csv(f"sim_data/smoo{N}/df{i}.csv", index_col=0)
    data["t"] = np.tile(np.linspace(0, 1, 100), N)
    start = time.time()
    spline_t = BSplines(data.t, k=20)
    spline_t_lin = BSplines(data.t, k=5, by=data["x"].to_numpy()[:, None])
    spline_t2 = BSplines(data.t, k=6, includescale=True, pen_order=1)
    spline_x2 = BSplines(data.x2, k=8, includescale=True, pen_order=2)
    spline_ti = TiProductSpline(spline_x2, spline_t2)
    design1 = tf.concat([spline_t.basis, spline_t_lin.basis, spline_ti.basis], 1, "float32")
    design2 = tf.ones((data.shape[0], 1), dtype="float32")
    penalty1 = PenaltyMatrix([spline_t.penalty, spline_t_lin.penalty], ti_penalties=[spline_ti.penalty],
                             reg_param=[100., 100., 100., 100.])
    penalty2 = PenaltyMatrix([np.array([[0.]])], reg_param=[0.])
    model1 = RegressionModel(penalty1, seed=i)
    model2 = RegressionModel(penalty2, constant=True, seed=i+1000)
    y = tf.cast(np.expand_dims(data.y, 1), "float32")
    design = (tf.cast(design1, "float32"), design2)
    dataset = tf.data.Dataset.from_tensor_slices((design, y)).batch(100)
    logger = LossLog_simplerer(dataset=dataset, huge=False, tolerance=0, outer_max=10, n_samples=5)

    # main_model = MainModel([embedding, model1, model2], N, categorical=True)

    main_model = MainModel([model1, model2], data.shape[0], categorical=False, dist="Normal")
    main_model.compile(optimizer=Adam(0.01), loss=negloglik)
    main_model(design)

    main_model.fit(design, y, epochs=1000, callbacks=[logger], batch_size=int((N * 100) / (100)))
    end = time.time()


    #with open(f"sim_results/smooth_{N}_{i}.pkl", "wb") as file:
    #    pickle.dump(main_model.weights, file)
    metrics = [logger.RMSE, logger.ll, logger.AIC, logger.BIC, end - start]
    K.clear_session()
    del main_model
    gc.collect()
    return metrics



def fit_beta(i, N=100):
    np.random.seed(i)
    data = pd.read_csv(f"sim_data/beta{N}/df{i}.csv", index_col=0)

    start = time.time()
    spline_t = BSplines(data.t, k=20)
    spline_t_lin = BSplines(data.t, k=5, by=data["x"].to_numpy()[:, None])
    spline_t2 = BSplines(data.t, k=6, includescale=True, pen_order=1)
    spline_x2 = BSplines(data.x2, k=8, includescale=True, pen_order=2)
    spline_ti = TiProductSpline(spline_x2, spline_t2)
    design1 = tf.concat([spline_t.basis, spline_t_lin.basis, spline_ti.basis], 1, "float32")
    design2 = tf.ones((data.shape[0], 1))
    penalty1 = PenaltyMatrix([spline_t.penalty, spline_t_lin.penalty], ti_penalties=[spline_ti.penalty],
                             reg_param=[100., 100., 100., 100.])
    penalty2 = PenaltyMatrix([np.array([[0.]])], reg_param=[0.])
    model1 = RegressionModel(penalty1, seed=i)
    model2 = RegressionModel(penalty2, constant=True, seed=i+1000)
    y = (np.expand_dims(data.beta, 1) * (N*100 - 1) + 0.5) / (N*100)
    #y = np.clip(np.expand_dims(data.beta, 1), np.finfo(np.float64).eps, np.finfo(np.float64).eps)
    y = tf.cast(y, "float32")
    design = (tf.cast(design1, "float32"), design2)
    dataset = tf.data.Dataset.from_tensor_slices((design, y)).batch(1000)
    logger = LossLog_simplerer(dataset=dataset, huge=False, tolerance=0, outer_max=10)


    main_model = MainModel([model1, model2], data.shape[0], categorical=False, dist="Beta")
    main_model.compile(optimizer=Adam(0.01), loss=negloglik)
    main_model(design)
    main_model.fit(design, y, epochs=1000, callbacks=[logger], batch_size=int((N*100)/(100)))
    end = time.time()


    #with open(f"sim_results/smoo_{N}_{i}.pkl", "wb") as file:
    #    pickle.dump(main_model.weights, file)
    metrics = [logger.RMSE, logger.ll, logger.AIC, logger.BIC, end - start]
    K.clear_session()
    del main_model
    gc.collect()
    return metrics


