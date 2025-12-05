import pickle
import sys
from bricks import *
from deepdl.splines import *
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("TkAgg")
import time
np.set_printoptions(suppress=True)
tf.config.run_functions_eagerly(False)


for N in [100, 1000, 10000]:
    fits = []
    for i in range(0, 100):
        fits.append(fit_lin(i, N=N))
    metrics = np.array([fit for fit in fits])
    with open(f"sim_results/fits_{N}linlong.pkl", "wb") as f:
        pickle.dump(metrics, f)
##

for N in [100, 1000, 10000]:
    fits = []
    for i in range(0, 100):
        fits.append(fit_smooth(i, N=N))
    metrics = np.array([fit for fit in fits])
    with open(f"sim_results/fits_{N}smoothlong.pkl", "wb") as f:
        pickle.dump(metrics, f)

##
##


for N in [100, 1000, 10000]:
    fits = []
    for i in range(0, 100):
        fits.append(fit_beta(i, N=N))
    metrics = np.array([fit for fit in fits])
    with open(f"sim_results/fits_{N}betalong.pkl", "wb") as f:
        pickle.dump(metrics, f)



for i in [100, 1000, 10000]:
    with open(f"sim_results/fits_{i}linlong.pkl", "rb") as f:
        metrix = pickle.load(f)

    metrics = metrix.mean(0)
    print(f"{metrics.round(4)[0]} & {int(-metrics.round(0)[1])} & {int(metrics.round(0)[2])} & {int(metrics.round(0)[3])} & {int(metrics.round(0)[4])}")
    print(metrics[4])



for i in [100, 1000, 10000]:
    with open(f"sim_results/fits_{i}smoothlong.pkl", "rb") as f:
        metrix = pickle.load(f)

    metrics = metrix.mean(0)
    print(f"{metrics.round(4)[0]} & {int(-metrics.round(0)[1])} & {int(metrics.round(0)[2])} & {int(metrics.round(0)[3])} & {int(metrics.round(0)[4])}")
    print(metrics[4])



for i in [100, 1000, 10000]:
    with open(f"sim_results/fits_{i}betalong.pkl", "rb") as f:
        metrix = pickle.load(f)

    metrics = metrix.mean(0)
    print(f"{metrics.round(4)[0]} & {int(-metrics.round(0)[1])} & {int(metrics.round(0)[2])} & {int(metrics.round(0)[3])} & {int(metrics.round(0)[4])}")
    print(metrics[4].round(4))
    

