"""

This script simulates all the datasets needed for the simulation study. Simply loops
over simulation functions. Files are written to "./sim_data"

"""


import sys
from pathlib import Path


from bricks import *


for i in range(100):
    sim_function_lin(100, seed=i)
    sim_function_lin(1000, seed=i)
    sim_function_lin(10000, seed=i)
    sim_function_smoo(100, seed=i)
    sim_function_smoo(1000, seed=i)
    sim_function_smoo(10000, seed=i)
    sim_function_beta(100, seed=i)
    sim_function_beta(1000, seed=i)
    sim_function_beta(10000, seed=i)
