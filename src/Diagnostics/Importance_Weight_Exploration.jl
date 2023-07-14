
# Generate observations, X, from the proposal distribution. Compute their raw importance weights 


# ---------------------------------------------------------------------------- #
#                Set parameter values and generate observed data               #
# ---------------------------------------------------------------------------- #


using LTHC_Analysis
using Test

using ReverseDiff
using Random
using Distributions
using Statistics
using LinearAlgebra
using Optim
using JLD2
using LogExpFunctions
import SpecialFunctions.loggamma

using ProgressMeter

using Plots


# ---------------------------------------------------------------------------- #
#                        Construct some data for testing                       #
# ---------------------------------------------------------------------------- #


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


# ---------------- Some values of theta at which to run tests ---------------- #
theta1 = theta_0
theta2 = theta_init1



# ---------------------------------------------------------------------------- #
#                                 Generate Data                                #
# ---------------------------------------------------------------------------- #

# Random.seed!(1)

# --------------------------- Algorithm parameters --------------------------- #
W = 3  # Number of datasets to generate and analyze
N = 20   # Number of trajectories to include in each dataset
M = 100   # Size of Monte Carlo sample
B = 100   # Number of MCEM iterations
R = 10    # Number of times to repeat MCEM on the same dataset
L = 3     # Number of different points to use for the initial guesses
t_max_raw = 25  # Number of time steps to simulate. Trailing zeros will be dropped.



# ------------------------------- Generate data ------------------------------ #
Random.seed!(1)
all_Ys, all_Xs = make_complete_data(t_max_raw, theta_0, Y_1, N, W)
Y_vec = all_Ys[1]
X_traj_vec = all_Xs[1]
omega_traj_vec = omega_traj_one_sample(Y_vec, X_traj_vec)

Y = Y_vec[4]
X = X_traj_vec[4]



# ---------------------------------------------------------------------------- #
#                           Generate proposal sample                           #
# ---------------------------------------------------------------------------- #

"""
Return the distribution used to generate proposals for X.
"""
function get_proposal_dist(lambda, eta)
    r = (1 - lambda) / (eta^2 - lambda)
    q = lambda / eta^2

    return ProposalDistribution(r, q)
end

eta = 1.5





#! This function prints an indicator every time it needs to reset due to a prematurely extinct trajectory. When prioritizing performance over checking stability, comment out the indicated lines.
#* Lines to comment are indicated in this color
"""
Generate a sequence of durations conditional on a sequence of case counts. Returns the sequence and its probability weight (obtained from the conditioning event). Proposals are from a negative binomial distribution with mean equal to the mean of X and variance scaled up by eta^2.
"""
function one_X_given_Y(Y, phi_0, gamma, lambda, eta)
    t_max = length(Y)
    duration_dist = DurationDistribution(lambda)
    prop_dist = get_proposal_dist(lambda, eta)
    
    # @test mean(duration_dist) ≈ mean(prop_dist)
    # @test var(prop_dist) ≈ eta^2 * var(duration_dist)

    jump_count = 0 #**********************************************************************************
    @label time_1   # Beginning of outbreak simulation

    X = Vector(undef, length(Y)) # Initialize X so the for loop can edit it
    # weight = BigFloat(1.0)    # Initialize weight so the for loop can edit it
    #                             # Uses high-precision arithmetic to handle low-probability trajectories
    log_weight = 0.0

    for t in eachindex(Y)
        this_Y = Y[t]

        # ---------------------------- Generate durations ---------------------------- #
        X[t] = rand(prop_dist, this_Y)

        # ----- Check if the outbreak goes extinct before more cases are observed ---- #
        # ---- Only check when durations have been generated and more cases remain --- #
        if this_Y > 0 && t < t_max
            required_duration = time_to_next_case(Y, t)
            next_omega = get_omega_after_traj(Y[1:t], X, t + required_duration)
            if next_omega == 0  # Check whether there will be any active cases for next observed infection
            # if all(X[t] .< required_duration) 
                jump_count += 1 #******************************************************************************
                if jump_count > 50
                    println("Jump to time_1 number $jump_count at time $t of $t_max")   #**************************
                end
                @goto time_1    # If so, start a new outbreak
            end
        end
        
        # ----------- Contribution to the weight from importance sampling X ---------- #
        if length(X[t]) >= 1
            these_log_imp_ratios = logpdf.(duration_dist, X[t]) .- logpdf.(prop_dist, X[t])
            log_imp_ratio = sum(these_log_imp_ratios)
            log_weight += log_imp_ratio
        end

        # ------------- Contribution to the weight from conditioning on Y ------------ #
        # ----------------- Initial number of cases is assumed fixed ----------------- #
        if t > 1
            phi = get_phi(phi_0, gamma, t)
            omega = get_omega(Y, X, t)
            offspring_dist = OffspringDistribution(omega * phi)

            # Contribution of Y to the weight
            log_weight += logpdf(offspring_dist, this_Y)
        end
    end

    # ----------------- Compute weight for end of observed cases ----------------- #
    # ------- Note: probability is computed on the original (not log) scale ------ #
    # weight *= post_trajectory_log_lik(Y, X, phi_0, gamma, lambda; log_scale = false)
    log_weight += post_trajectory_log_lik(Y, X, phi_0, gamma, lambda; log_scale = true)
    
    # output = Dict("X" => X, "weight" => weight)
    output = Dict("X" => X, "log_weight" => log_weight)
    return output
end



"""
Generate one X for each Y in Y_vec. Returns a vector of Xs and the (log) weight corresponding to this MC draw.
"""
function one_MC_draw(Y_vec, phi_0, gamma, lambda, eta)
    n = length(Y_vec)
    X_vec = Vector(undef, n)
    log_weight = 0.0

    for i in 1:n
        Y = Y_vec[i]
        output = one_X_given_Y(Y, phi_0, gamma, lambda, eta)
        X_vec[i] = output["X"]
        log_weight += output["log_weight"]
    end

    return X_vec, log_weight
end

"""
Generate a sample of M iid duration trajectories for each Y conditional on that Y. Returns an Mxn array of X trajectories, and a vector of the MC samples' (standardized and non-log) weights.
Optionally, can return the un-normalized log-weights. Caution: This is rarely a good idea.
"""
function multiple_MC_draws(M, Y_vec, theta, eta; norm_weights=true)
    n = length(Y_vec)

    #? I might regret this, but I'm changing all_X_vecs from a matrix to a vector of vectors.
    # all_X_vecs = Array{Any}(undef, M, n)
    all_X_vecs = Vector{Any}(undef, M)
    all_log_weights_raw = Vector(undef, M)

    for i in 1:M
        all_X_vecs[i], all_log_weights_raw[i] = one_MC_draw(Y_vec, theta..., eta)
    end

    if norm_weights
        # -------- Normalize weights using highly stable log-sum-exp function -------- #
        # --------- Also exponentiates away the log in the way you need it to -------- #
        all_weights = normalize_weights(all_log_weights_raw)
    else
        all_weights = all_log_weights_raw
    end

    return all_X_vecs, all_weights
end





"""
Generate an importance sample for X, as well as the corresponding importance weights.
Note: If norm_weights is false, then the weights are returned unnormalized and on log-scale.
"""
function get_importance_sample(theta, Y_vec, M, eta; norm_weights=true)
    return multiple_MC_draws(M, Y_vec, theta, eta, norm_weights=norm_weights)
end


# --- Generate observations and complete importance weights (on log scale) --- #
Random.seed!(1234)
MC_phi_0, MC_gamma, MC_lambda = theta1
MC_eta = eta
all_prop_Xs, all_log_weights = get_importance_sample(theta1, Y_vec, M, eta; norm_weights=false)

get_ESS(normalize_weights(all_log_weights))


# ---------------------------------------------------------------------------- #
#            Compute intermediate importance weights (on log scale)            #
# ---------------------------------------------------------------------------- #


# ------------------- Helper functions for computing dF/dG ------------------- #

function log_w1_one_t(y, omega, phi_t)
    A = y * log(omega)
    B = omega * phi_t

    return A - B
end

function log_w1_one_t(y, omega, phi_0, gamma, t)
    phi_t = get_phi(phi_0, gamma, t)
    return log_w1_one_t(y, omega, phi_t)
end

function log_w1_one_facility(Y, omega_traj, phi_0, gamma)
    log_w1 = 0.0

    T_star = length(omega_traj)-1   # Duration of outbreak
    for t in 2:T_star
        if t <= length(Y)
            y = Y[t]
        else
            y=0
        end
        omega = omega_traj[t]
        log_w1 += log_w1_one_t(y, omega, phi_0, gamma, t)
    end

    return log_w1
end




function log_w1(Y_vec, omega_traj_vec, phi_0, gamma)
    log_w1 = 0.0

    for i in 1:length(Y_vec)
        Y = Y_vec[i]
        omega_traj = omega_traj_vec[i]
        log_w1 += log_w1_one_facility(Y, omega_traj, phi_0, gamma)
    end

    return log_w1
end


# ----------------- Helper functions for computing dG/dG_eta ----------------- #
function log_w2_one_case(x, lambda, eta)
    r = get_proposal_r(lambda, eta)
    
    A = (x-1) * log(r)
    B = 2 * (x-1) * log(eta)
    C = loggamma(x)
    D = loggamma(x + r - 1)

    return A + B + C - D
end

function log_w2_one_t(X, lambda, eta)
    log_w2 = 0.0

    for x in X
        log_w2 += log_w2_one_case(x, lambda, eta)
    end

    return log_w2
end

function log_w2_one_facility(X_traj, lambda, eta)
    log_w2 = 0.0

    for X in X_traj
        log_w2 += log_w2_one_t(X, lambda, eta)
    end

    return log_w2
end

function log_w2(X_traj_vec, lambda, eta)
    log_w2 = 0.0

    for X_traj in X_traj_vec
        log_w2 += log_w2_one_facility(X_traj, lambda, eta)
    end

    return log_w2
end



# ------------------------ Compute importance weights ------------------------ #

all_prop_omegas = [omega_traj_one_sample(Y_vec, X_traj_vec) for X_traj_vec in all_prop_Xs]

all_log_w1s = [log_w1(Y_vec, omega_traj_vec, MC_phi_0, MC_gamma) for omega_traj_vec in all_prop_omegas]
all_log_w2s = [log_w2(X_traj_vec, MC_lambda, MC_eta) for X_traj_vec in all_prop_Xs]


#! This difference should be constant, at least approximately. Something is going wrong.
all_log_ws = all_log_w1s + all_log_w2s
all_log_weights - all_log_ws




lambda = 0.5
eta = 3

X_traj_vec1 = all_prop_Xs[1]
X_traj_vec2 = all_prop_Xs[2]
X_traj_vec3 = all_prop_Xs[3]

a1 = log_w2(X_traj_vec1, lambda, eta)
a2 = log_w2(X_traj_vec2, lambda, eta)
a3 = log_w2(X_traj_vec3, lambda, eta)

b1 = log_w2b(X_traj_vec1, lambda, eta)
b2 = log_w2b(X_traj_vec2, lambda, eta)
b3 = log_w2b(X_traj_vec3, lambda, eta)


a1 - b1
a2 - b2
a3 - b3