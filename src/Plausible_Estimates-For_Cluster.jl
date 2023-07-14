


# using Pkg
# Pkg.activate("LTHC Analysis - Julia/LTHC Julia Project")


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
using StatsBase  
using Distributions
using Statistics



include("Define_All_Distributions.jl")
include("Helper_Functions.jl")
include("Generate_Data.jl")
# include("Obs_Data_Likelihood_Functions.jl")
include("Complete_Data_Likelihood_Functions.jl")
# include("Conditional_Distribution_Functions.jl")
# include("EM_Functions.jl")
include("MCEM_Functions.jl")
include("Ascent_MCEM_Functions.jl")
include("SAEM_Functions.jl")

# ----------------------------- Parameter Bounds ----------------------------- #
par_lower = [0.0, 0.0, 0.0]
par_upper = [Inf, Inf, 1]

eta = 1.5    




d = parse(Int, ARGS[1])


B = 100



"""
Estimate the likelihood ratio for theta_num divided by the theta value used to generate all_Xs.
all_fixed_vals is a vector of all log-weights minus all log-liks for the reference parameter value. I.e. All values which don't change with the point at which we're evaluating the log-LR.
"""
function get_log_LR_hat(theta_num, all_Xs, all_fixed_vals)
    all_log_liks = [complete_data_log_lik(theta_num, Y_vec, this_X) for this_X in all_Xs]
    all_diffs = all_log_liks .+ all_fixed_vals
    log_LR_hat = logsumexp(all_diffs)
    return log_LR_hat
end

function get_log_LR_hat(theta_num, theta_den, all_Xs, all_wts)
    all_log_wts = log.(all_wts)
    all_ref_log_liks = [complete_data_log_lik(theta_den, Y_vec, this_X) for this_X in all_Xs]
    all_fixed_vals = all_log_wts .- all_ref_log_liks

    log_LR_hat = get_log_LR_hat(theta_num, all_Xs, all_fixed_vals)
    return log_LR_hat
end


# ---------------------------------------------------------------------------- #
#                                 Run analysis                                 #
# ---------------------------------------------------------------------------- #

@load "output/sim_results/sim_results-$d.jld2" all_theta_hat_trajectories_MCEM all_theta_hats_MCEM all_cov_hats_MCEM all_Ms_MCEM all_num_iterations_MCEM


S = length(all_theta_hats_MCEM)     # Number of MCEM initializations
M_lik_rat = 100    # Number of samples to use for estimating the likelihood ratio from theta_LR to theta_hat
B = 100 # Number of times to repeat estimation to get empirical SE

theta_LR = [1.0, 0.1, 0.1]  # Reference parameter from which likelihood ratios are estimated

# ---------------- Construct observed data for this imputation --------------- #

data_from_R = load("src/data/all_Y_trajectories.RData")
all_Y_imputations = data_from_R["all_Ys"]
Y_vec_raw = all_Y_imputations[d]

# Add single-case outbreaks
for i in 1:35
    push!(Y_vec_raw, [1])
end

Y_vec = Vector{Int64}[]
for i in eachindex(Y_vec_raw)
    push!(Y_vec, convert.(Int, Y_vec_raw[i]))
end



# --------------------- Get initial estimates of log-LRs --------------------- #

# Generate a single MC sample from reference parameter value
Random.seed!(1)
Xs_LR, unnorm_untrunc_log_wts_LR = get_importance_sample(theta_LR, Y_vec, M_lik_rat, norm_weights=false);
unnorm_log_wts_LR = truncate_weights(unnorm_untrunc_log_wts_LR);
wts_LR = normalize_weights(unnorm_log_wts_LR, truncate=false);


# Evaluate quantities which don't change with target parameter value
ref_log_liks = [complete_data_log_lik(theta_LR, Y_vec, this_X_vec) for this_X_vec in Xs_LR];
log_wts = log.(wts_LR);
fixed_terms = log_wts .- ref_log_liks;

# Compute log-LR estimates
log_LR_estimates = zeros(length(all_theta_hats_MCEM));
log_LR_SEs = zeros(length(all_theta_hats_MCEM));
@showprogress for i in eachindex(log_LR_estimates)
    this_theta_hat = all_theta_hats_MCEM[i]
    this_log_liks = [complete_data_log_lik(this_theta_hat, Y_vec, this_X_vec) for this_X_vec in Xs_LR]

    this_log_terms = this_log_liks .+ fixed_terms

    this_log_LR = logsumexp(this_log_terms)
    log_LR_estimates[i] = this_log_LR
end

ind_max = argmax(log_LR_estimates)
theta_max = all_theta_hats_MCEM[ind_max]


# --------------------- Get empirical SEs of log-LRs --------------------- #

all_deltas_prog = Progress(B*S + 2*B, "Computing differences in log-LRs")

all_deltas = zeros(B, S)
Threads.@threads for b in 1:B
    Xs_LR, wts_LR = get_importance_sample(theta_LR, Y_vec, M_lik_rat, norm_weights=true);
    next!(all_deltas_prog)

    LR_max = get_log_LR_hat(theta_max, theta_LR, Xs_LR, wts_LR)
    next!(all_deltas_prog)
    
    all_LRs = zeros(S)
    for s in 1:S
        this_theta_hat = all_theta_hats_MCEM[s]
        this_LR = get_log_LR_hat(this_theta_hat, theta_LR, Xs_LR, wts_LR)
        all_LRs[s] = this_LR
        next!(all_deltas_prog)
    end

    this_deltas = LR_max .- all_LRs
    all_deltas[b, :] = this_deltas
end

# Get empirical SEs for each LR difference
delta_SDs = [std(all_deltas[:, s]) for s in 1:S]

# Get LR differences on original sample
original_deltas = maximum(log_LR_estimates) .- log_LR_estimates

difference_bounds = original_deltas - delta_SDs
inds_small_diffs = findall(difference_bounds .<= 0)


# ----------- Get estimates and covariances for promising initializations ---------- #
all_theta_hats_close = all_theta_hats_MCEM[inds_small_diffs]
all_cov_hats_close = all_cov_hats_MCEM[inds_small_diffs]






# ---------------------------------------------------------------------------- #
#                                  Estimate R0                                 #
# ---------------------------------------------------------------------------- #

# ----------------------- Estimates for each initialization ---------------------- #
all_phi_0_hats = getindex.(all_theta_hats_MCEM, 1)
all_lambda_hats = getindex.(all_theta_hats_MCEM, 3)
all_R0_hats = all_phi_0_hats ./ all_lambda_hats

# ------------------------ SEs for each initialization ----------------------- #

# Construct Jacobian matrix for transformation
all_As = 1 ./ all_lambda_hats
all_Bs = -all_phi_0_hats ./ all_lambda_hats.^2
all_jacobs = [[all_As[i], all_Bs[i]] for i in eachindex(all_As)]

# Compute standard errors
all_SEs = [sqrt.(all_jacobs[i]' * all_cov_hats_MCEM[i][[1,3], [1,3]] * all_jacobs[i]) for i in eachindex(all_jacobs)]


# --------- Extract estimates and SEs from promising initializations --------- #
all_R0s_close = all_R0_hats[inds_small_diffs]
all_R0_SEs_close = all_SEs[inds_small_diffs]




# ---------------------------------------------------------------------------- #
#                  Save results from promising initializations                 #
# ---------------------------------------------------------------------------- #

@save "output/estimates_long_careful/estimates_long_careful-$d.jld2" all_theta_hats_close all_cov_hats_close all_R0s_close all_R0_SEs_close inds_small_diffs


test = 1
check = 3

@save "a_file.jld2" test
@save "a_file.jld2" check

@load "a_file.jld2"
