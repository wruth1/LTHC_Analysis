
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
#            Define new importance sampler using NegBin distribution           #
# ---------------------------------------------------------------------------- #

# As proposal, we use a negative binomial random variable (plus 1 to match the support of X). Its mean is the same as X (i.e. 1/lambda), and its variance is eta^2 times that of X (i.e. eta^2 * (1-lambda) / lambda^2)

"""
Return the distribution used to generate proposals for X.
"""
function get_proposal_dist(lambda, eta)
    r = (1 - lambda) / (eta^2 - lambda)
    q = lambda / eta^2

    return ProposalDistribution(r, q)
end


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
    @label time_1   # Beginning of outbreak generation

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
            if next_omega == 0  # Better to check whether there will be any active cases for next observed infection
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




some_etas = round.(sqrt.(2 .^ (0:0.5:2)), sigdigits=3)
# some_etas = round.(sqrt.(2 .^ (0:0.2:1)), sigdigits=3)
M_max = 20000

# Container for raw log-weights
all_raw_weights = Array{Any}(undef, M_max, length(some_etas))

# Container for effective sample sizes
all_ESSs = Vector{Any}(undef, length(some_etas))

Random.seed!(11)

for i in eachindex(some_etas)
    these_ESSs = Vector{Float64}(undef, M_max)

    this_eta = some_etas[i]

    _, all_log_weights = get_importance_sample(theta1, Y_vec, M_max, this_eta; norm_weights=false)
    all_raw_weights[:, i] = all_log_weights

    this_prog = Progress(M_max, "Eta number $i of $(length(some_etas))")


    for j in reverse(eachindex(these_ESSs))
        these_log_weights = all_log_weights[1:j]
        these_weights = normalize_weights(these_log_weights)
        these_ESSs[j] = get_ESS(these_weights)

        next!(this_prog)
    end

    all_ESSs[i] = these_ESSs
end


some_colors = [:red, :blue, :green, :orange, :purple, :black, :brown, :pink, :gray, :olive]

ESS_plot = plot(1:M_max, all_ESSs[1], label="eta = $(some_etas[1])", color=some_colors[1], xlabel = "MC Sample Size", ylabel = "Effective Sample Size");
for i in 2:length(some_etas)
    plot!(ESS_plot, 1:M_max, all_ESSs[i], label="eta = $(some_etas[i])", color=some_colors[i]);
end

short_ESS_plot = deepcopy(ESS_plot);
# short_ESS_plot = ESS_plot;

xlims!(short_ESS_plot, (0, 1000));
ylims!(short_ESS_plot, (0, 30));

plot(short_ESS_plot, ESS_plot, size = (1100, 1000), plot_title = "Eta: Overdisp. Mult. for SD of Proposal Dist")



# ---------------------------------------------------------------------------- #
#                             Agapiou et al. (2017)                            #
# ---------------------------------------------------------------------------- #

# ------------------------------- Estimate tau ------------------------------- #

# The un-normalized weights are sufficiently small that we can only work with them on log-scale
# Thus, we compute tau from the normalized weights. This is an easy adjustment from the formula in Agapiou et al.

all_normed_weights = [normalize_weights(all_raw_weights[:, i], truncate=true) for i in eachindex(some_etas)]
all_taus = M_max .* [sum(all_normed_weights[i].^2) for i in eachindex(all_normed_weights)]


# Asymptotically, ESS ≈ M / tau
# Add the lines M/tau to our ESS plot
for i in eachindex(some_etas)
    plot!(ESS_plot, 1:M_max, (1/all_taus[i]).* (1:M_max), label=nothing, color=some_colors[i], linewidth=1, linestyle=:dash);
end
plot(ESS_plot, size = (1100, 1000))