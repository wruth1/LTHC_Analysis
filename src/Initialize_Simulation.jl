
# using Pkg
# Pkg.activate("LTHC Analysis - Julia/LTHC Julia Project")

using LTHC_Analysis

using Random
using Pipe#: @pipe
# using Plotly
using DataFrames
using Optim
# using Plots
using Dates # For current date/time using now()
# using StatsPlots    # For plotting data frames
using ProgressMeter
using BenchmarkTools
using JLD2
using LinearAlgebra # For eigvals()
using LogExpFunctions # For logsumexp() in weight calculations
using StatsBase     # For weighted covariance estimation


using Plots


# using CodecBzip2  # For importing data
# using RData       # For importing data



# -------------------------------- Import data ------------------------------- #
# data_raw = load("LTHC Analysis - Julia/LTHC_Raw_Data.rda")
# all_Ys = data_raw["BC_LTHC_outbreaks_100Imputs"][1]["time_series"]
# Y = all_Ys[1]



# --------------------------- True model parameters -------------------------- #
phi_0_0 = 1.5
gamma_0 = 0.3
lambda_0 = 0.5
theta_0 = [phi_0_0, gamma_0, lambda_0]

phi_0, gamma, lambda = theta_0

# Previous values:
# gamma_0 = 0.15, matched with gamma_init = 0.1 (changed 2022-12-09)

# ------------------------- Initial parameter guesses ------------------------ #
# ------------------------ Order: phi_0, gamma, lambda ----------------------- #
theta_init1 = [1.25, 0.25, 0.45]
theta_init2 = [1.6, 0.325, 0.525]
theta_init3 = theta_0
all_theta_inits = [theta_init1, theta_init2, theta_init3]

# ------------------------- Number of cases at time 0 ------------------------ #
Y_1 = 1



# ---------------------------------------------------------------------------- #
#                                 Run Algorithm                                #
# ---------------------------------------------------------------------------- #

# Random.seed!(1)

# --------------------------- Algorithm parameters --------------------------- #
W = 3  # Number of datasets to generate and analyze
N = 5   # Number of trajectories to include in each dataset
M = 100   # Size of Monte Carlo sample
B = 100   # Number of MCEM iterations
R = 10    # Number of times to repeat MCEM on the same dataset
L = 3     # Number of different points to use for the initial guesses
t_max_raw = 25  # Number of time steps to simulate. Trailing zeros will be dropped.


# ----------------------------- Parameter Bounds ----------------------------- #
par_lower = [0.0, 0.0, 0.0]
par_upper = [Inf, Inf, 1]



# ------------------------------- Generate data ------------------------------ #
Random.seed!(1)
all_Ys, all_Xs = make_complete_data(t_max_raw, theta_0, Y_1, N, W)
Y_vec = all_Ys[1]
X_traj_vec = all_Xs[1]
omega_traj_vec = omega_traj_one_sample(Y_vec, X_traj_vec)

Y = Y_vec[4]
X = X_traj_vec[4]