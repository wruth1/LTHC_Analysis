


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


# ---------------------------------------------------------------------------- #
#                            Import and process data                           #
# ---------------------------------------------------------------------------- #


data_from_R = load("src/data/all_Y_trajectories.RData")


all_Y_imputations = data_from_R["all_Ys"]

Y_vec_raw = all_Y_imputations[2]

# ------------------------- Add single case outbreaks ------------------------ #
for i in 1:35
    push!(Y_vec_raw, [1])
end

Y_vec = Vector{Int64}[]
for i in eachindex(Y_vec_raw)
    push!(Y_vec, convert.(Int, Y_vec_raw[i]))
end

# # Check initial outbreak sizes
# using FreqTables
# day_one_sizes = getindex.(Y_vec, 1)
# println(day_one_sizes)
# freqtable(day_one_sizes)

# ---------------------------------------------------------------------------- #
#                              Run AMCEM Algorithm                             #
# ---------------------------------------------------------------------------- #

# ------------ Set up initial values for phi_0, gamma, and lambda ------------ #

# Values for each variable individually
all_phi_0_inits = [0.1, 1.0, 3.0]
all_gamma_inits = [0.01, 0.1, 0.5]
all_lambda_inits = [0.05, 0.1, 0.2]
num_combs = length(all_phi_0_inits) * length(all_gamma_inits) * length(all_lambda_inits)

# # Under this configuration, combination 7 or 8 has a numerical -infinity in a log-likelihood, even on log-scale
# # It's not so bad though. This tells us that the likelihood is very small, so we don't need to bother with these values.
# all_phi_0_inits = [1.0, 5.0]
# all_gamma_inits = [0.1, 1, 5]
# all_lambda_inits = [0.01, 0.1]


# Create vector of all combinations of initial values
# Two copies of each combination
num_replicates = 2
all_theta_inits = []
for phi_0_init in all_phi_0_inits
    for gamma_init in all_gamma_inits
        for lambda_init in all_lambda_inits
            for _ in 1:num_replicates
                push!(all_theta_inits, [phi_0_init, gamma_init, lambda_init])
            end
        end
    end
end

# theta_init = [3.0, 0.1, 0.1]
# theta_old = theta_init

# --------------- Set control parameters for ascent-based MCEM --------------- #
alpha1 = 0.1    # confidence level for checking whether to augment MC sample size
alpha2 = 0.3    # confidence level for computing next step's initial MC sample size
alpha3 = 0.2    # confidence level for checking whether to terminate MCEM
k = 2           # when augmenting MC sample, add M/k new points
atol = 1e-3     # Absolute tolerance for convergence. 


control = Ascent_MCEM_Control(alpha1, alpha2, alpha3, k, atol)
ascent_MCEM_control = control


M_init = 5 # Initial number of MC samples to use for MCEM

M_SE = 200   # Minimum number of MC samples to use for covariance estimation after terminating
            # Actual number is the larger of this and the final M used in MCEM


# ---------- Build containers for parameter and covariance estimates --------- #
all_theta_hat_trajectories_MCEM = []
all_theta_hats_MCEM = []
all_cov_hats_MCEM = []
all_Ms_MCEM = []
all_num_iterations_MCEM = []


# ------------------------------- Run algorithm ------------------------------ #
Random.seed!(1)
@showprogress for i in eachindex(all_theta_inits)
    println("\n Iteration $i")
    Random.seed!(i^2)
    # Random.seed!(i^3)
    # println("Parameter combination $i")
    # i+=1
    theta_init = all_theta_inits[i]

    # Run MCEM
    this_traj_MCEM, M_final, num_iterations = run_ascent_MCEM(theta_init, Y_vec, M_init, control, diagnostics=true)

    # Extract and store theta hat
    theta_hat_MCEM = this_traj_MCEM[end]
    push!(all_theta_hat_trajectories_MCEM, this_traj_MCEM)
    push!(all_theta_hats_MCEM, theta_hat_MCEM)

    # Estimate covariance
    # Note: Due to MC variability, we sometimes get negative estimated variances. In this case, generate a new estimate
    this_M_SE = max(M_SE, M_final)
    pos_def_check = false
    this_cov_hat = []   # Initialize this_cov_hat so it's available outside the while loop
    num_cov_est_attempts = 0

    while !pos_def_check
        num_cov_est_attempts += 1
        if num_cov_est_attempts > 1
            println("Covariance estimation attempt $num_cov_est_attempts")
        end
        # The Hermitian() function gets rid of roundoff error making cov_hat not quite symmetric. Without this extra step, isposdef() would fail
        this_cov_hat = Hermitian(MCEM_COV_formula(theta_hat_MCEM, Y_vec, theta_hat_MCEM, this_M_SE, false))
        pos_def_check = isposdef(this_cov_hat)
    end

    push!(all_cov_hats_MCEM, this_cov_hat)
    push!(all_Ms_MCEM, M_final)
    push!(all_num_iterations_MCEM, num_iterations)
end

# Save results to file
using JLD2
# @save "src/Data/Truncated_MCEM_Results-tol=1e-3.jld2" all_theta_hat_trajectories_MCEM all_theta_hats_MCEM all_cov_hats_MCEM all_Ms_MCEM all_num_iterations_MCEM
# @save "src/Data/Truncated_MCEM_Results-No_Singles.jld2" all_theta_hat_trajectories_MCEM all_theta_hats_MCEM all_cov_hats_MCEM all_Ms_MCEM all_num_iterations_MCEM

# # Load results from file
# using JLD2
# @load "src/Data/Truncated_MCEM_Results.jld2" all_theta_hat_trajectories_MCEM all_theta_hats_MCEM all_cov_hats_MCEM all_Ms_MCEM all_num_iterations_MCEM
@load "From_Cluster/output/sim_results/sim_results-2.jld2"




# ---------------------------------------------------------------------------- #
#           Estimate approximate EM convergence rate at each estimate          #
# ---------------------------------------------------------------------------- #

M_rate_estimation = 200

# Estimate each EM rate matrix
all_EM_rates = []

@showprogress for i in eachindex(all_theta_hats_MCEM)
    # println("\n Iteration $i")

    theta_hat = all_theta_hats_MCEM[i]

    # Get MC sample
    Random.seed!(i^3)
    all_Xs, all_wts = get_importance_sample(theta_hat, Y_vec, M_rate_estimation)

    # Compute relevant matrices
    this_complete_cond_info = Hermitian(MC_complete_cond_info(theta_hat, Y_vec, all_Xs, all_wts))
    this_inv_complete_cond_info = inv(this_complete_cond_info)
    this_missing_cond_info = Hermitian(MC_score_cov(theta_hat, Y_vec, all_Xs, all_wts))

    # Compute EM rate
    this_EM_rate = this_inv_complete_cond_info * this_missing_cond_info
    push!(all_EM_rates, this_EM_rate)
end


# Get spectral radius of each EM rate matrix. Also, give eigenvector corresponding to largest eigenvalue
# Note: I have computed the rate as fraction of information which is missing. Thus, large eigenvalues correspond to slow convergence
EM_rate_spec_rads = []
EM_rate_eigvecs = []
for this_EM_rate in all_EM_rates
    this_EM_rate_eigvals, this_EM_rate_eigvecs = eigen(this_EM_rate)
    this_EM_rate_spec_rad = maximum(this_EM_rate_eigvals) / minimum(this_EM_rate_eigvals)
    push!(EM_rate_spec_rads, this_EM_rate_spec_rad)
    push!(EM_rate_eigvecs, this_EM_rate_eigvecs[:, argmax(abs.(this_EM_rate_eigvals))])
end

# Size of third component in worst-case direction eigenvector
EM_lambda_in_worst_directions = []
for this_eigvec in EM_rate_eigvecs
    push!(EM_lambda_in_worst_directions, this_eigvec[3])
end

data_EM_conv = DataFrame(:spec_rad => EM_rate_spec_rads, :lambda_size => EM_lambda_in_worst_directions)

# Compute correlation between spectral radius and lambda component of worst direction
a = EM_rate_spec_rads
b = abs.(EM_lambda_in_worst_directions)
cor(a, b)


eigen(all_EM_rates[1])


# ---------------------------------------------------------------------------- #
#          Estimate obs data log-lik ratio for each parameter estimate         #
# ---------------------------------------------------------------------------- #

#* Note: We use the same MC sample to estimate each LR. This reduces MC noise. If I felt ambitious, I could repeat this process for a few such MC samples.

theta_LR = [1.0, 0.1, 0.1]  # Reference parameter from which likelihood ratios are estimated
M_lik_rat = 100    # Number of samples to use for estimating the likelihood ratio from theta_LR to theta_hat

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



using JLD2
final_theta_hat_MCEM = all_theta_hats_MCEM[argmax(log_LR_estimates)]
save("src/Data/Truncated_MCEM_Estimate-tol=1e-3.jld2", "final_theta_hat_MCEM", final_theta_hat_MCEM)
# save("src/Data/Truncated_MCEM_Estimate-No_Singles.jld2", "final_theta_hat_MCEM", final_theta_hat_MCEM)







# ---------------------------------------------------------------------------- #
#                  Compile Output Summaries Into a Data Frame                  #
# ---------------------------------------------------------------------------- #

using DataFrames
MCEM_data = DataFrame(par_comb = repeat(1:num_combs, inner=num_replicates), phi_0_init = getindex.(all_theta_inits, 1), gamma_init = getindex.(all_theta_inits, 2), lambda_init = getindex.(all_theta_inits, 3), phi_0_hat = getindex.(all_theta_hats_MCEM, 1), gamma_hat = getindex.(all_theta_hats_MCEM, 2), lambda_hat = getindex.(all_theta_hats_MCEM, 3))

# Standard errors
all_phi_0_SEs = sqrt.(abs.(getindex.(diag.(all_cov_hats_MCEM), 1)))
all_gamma_SEs = sqrt.(abs.(getindex.(diag.(all_cov_hats_MCEM), 2)))
all_lambda_SEs = sqrt.(abs.(getindex.(diag.(all_cov_hats_MCEM), 3)))
MCEM_data[!, :phi_0_SE] = all_phi_0_SEs
MCEM_data[!, :gamma_SE] = all_gamma_SEs
MCEM_data[!, :lambda_SE] = all_lambda_SEs

# Correlations
all_cor_mats = cov2cor.(all_cov_hats_MCEM)
all_phi_0_gamma_cor = getindex.(all_cor_mats, 1, 2)
all_phi_0_lambda_cor = getindex.(all_cor_mats, 1, 3)
all_gamma_lambda_cor = getindex.(all_cor_mats, 2, 3)
MCEM_data[!, :phi_0_gamma_cor] = all_phi_0_gamma_cor
MCEM_data[!, :phi_0_lambda_cor] = all_phi_0_lambda_cor
MCEM_data[!, :gamma_lambda_cor] = all_gamma_lambda_cor

# Final MC sizes and number of iterations
MCEM_data[!, :M_final] = all_Ms_MCEM
MCEM_data[!, :num_iters] = all_num_iterations_MCEM

# Likelihood Ratios
MCEM_data[!, :log_LR_estimates] = log_LR_estimates

# Check which lambda SEs are negative
all_phi_0_SEs_sign = sign.(getindex.(diag.(all_cov_hats_MCEM), 1))
all_gamma_SEs_sign = sign.(getindex.(diag.(all_cov_hats_MCEM), 2))
all_lambda_SEs_sign = sign.(getindex.(diag.(all_cov_hats_MCEM), 3))
MCEM_data[!, :phi_0_SE_sign] = all_phi_0_SEs_sign
MCEM_data[!, :gamma_SE_sign] = all_gamma_SEs_sign
MCEM_data[!, :lambda_SE_sign] = all_lambda_SEs_sign

# Save data
using CSV
CSV.write("src/Data/Truncated_Results-tol=1e-3.csv", MCEM_data)
# CSV.write("src/Data/Truncated_Results-No_Singles.csv", MCEM_data)

# # Load data
# MCEM_data = CSV.read("src/Data/Truncated_Results.csv", DataFrame)
# all_phi_0_hats = MCEM_data[!, :phi_0_hat]
# all_gamma_hats = MCEM_data[!, :gamma_hat]
# all_lambda_hats = MCEM_data[!, :lambda_hat]
# all_theta_hats_MCEM = [[all_phi_0_hats[i], all_gamma_hats[i], all_lambda_hats[i]] for i in eachindex(all_phi_0_hats)]





# ---------------------------------------------------------------------------- #
#                 Investigate optimal estimate and those nearby                #
# ---------------------------------------------------------------------------- #

ind_max = argmax(log_LR_estimates)
theta_hat = all_theta_hats_MCEM[ind_max]

# Build and save analysis based on final parameter estimate
data_theta_hat = DataFrame(parameter = ["phi_0", "gamma", "lambda"], estimate = theta_hat)
final_SEs = [all_phi_0_SEs[ind_max], all_gamma_SEs[ind_max], all_lambda_SEs[ind_max]]
data_theta_hat[!, :SE] = final_SEs
final_lcls = theta_hat - 1.96*final_SEs
final_ucls = theta_hat + 1.96*final_SEs
data_theta_hat[!, :lcl] = final_lcls
data_theta_hat[!, :ucl] = final_ucls

CSV.write("src/Data/Truncated_Estimate_Analysis-tol=1e-3.csv", data_theta_hat)
# CSV.write("src/Data/Truncated_Estimate_Analysis-No_Singles.csv", data_theta_hat)


# ---------------- Compute SE of log-LR estimate at maximizer ---------------- #


B = 100 # Number of times to repeat estimation to get empirical SE

all_LR_hats = zeros(B)

prog_bar = Progress(B)

Threads.@threads for b in 1:B
    # Generate a single MC sample from reference parameter value
    Random.seed!(b^2)
    Xs_LR, wts_LR = get_importance_sample(theta_LR, Y_vec, M_lik_rat, norm_weights=true);
    log_wts_LR = log.(wts_LR)

    # Evaluate log-likelihood ratio
    ref_log_liks = [complete_data_log_lik(theta_LR, Y_vec, this_X_vec) for this_X_vec in Xs_LR]
    opt_log_liks = [complete_data_log_lik(theta_hat, Y_vec, this_X_vec) for this_X_vec in Xs_LR]
    log_LR = opt_log_liks .- ref_log_liks

    # Compute log-LR estimate
    all_LR_hats[b] = logsumexp(log_wts_LR .+ log_LR)

    next!(prog_bar)
end


# Estimate SE of log-LR estimate
SE_log_LR = std(all_LR_hats)

# --------- Extract all estimates with log-LRs within 1 SE of the max -------- #
max_log_LR = maximum(log_LR_estimates)
subset(MCEM_data, :log_LR_estimates => x -> x .>= max_log_LR - SE_log_LR)




# ---------------------------------------------------------------------------- #
#                Inference based on the final parameter estimate               #
# ---------------------------------------------------------------------------- #

# Test whether gamma <= 0
using Distributions
test_norm = Normal(0, final_SEs[2])
p_val = ccdf(test_norm, final_theta_hat_MCEM[2])




# ---------------------------------------------------------------------------- #
#                          Construct and display plots                         #
# ---------------------------------------------------------------------------- #

using Plots
all_plots = []

# for i in eachindex(all_theta_hat_trajectories_MCEM)
for i in 1:Int(length(all_theta_hat_trajectories_MCEM)/2)

    # Indices of trajectories for this parameter combination
    j1 = 2*i - 1
    j2 = 2*i

    # Create plots from first trajectory
    all_phi_0_hats1 = getindex.(all_theta_hat_trajectories_MCEM[j1], 1);
    all_gamma_hats1 = getindex.(all_theta_hat_trajectories_MCEM[j1], 2);
    all_lambda_hats1 = getindex.(all_theta_hat_trajectories_MCEM[j1], 3);

    phi_0_plot = plot(all_phi_0_hats1, label=nothing, title="Comb'n $i");
    gamma_plot = plot(all_gamma_hats1, label=nothing);
    lambda_plot = plot(all_lambda_hats1, label=nothing);


    # Add second trajectory to plots
    all_phi_0_hats2 = getindex.(all_theta_hat_trajectories_MCEM[j2], 1);
    all_gamma_hats2 = getindex.(all_theta_hat_trajectories_MCEM[j2], 2);
    all_lambda_hats2 = getindex.(all_theta_hat_trajectories_MCEM[j2], 3);

    plot!(phi_0_plot, all_phi_0_hats2, label=nothing);
    plot!(gamma_plot, all_gamma_hats2, label=nothing);
    plot!(lambda_plot, all_lambda_hats2, label=nothing);


    # Combine plots into a single figure 

    # this_plot = plot(phi_0_plot, gamma_plot, lambda_plot, layout=(3,1), size=(1200, 1000));
    this_plot = plot(phi_0_plot, gamma_plot, lambda_plot, layout=(3,1), size=(2000, 1200));
    push!(all_plots, this_plot)
end

full_plot = plot(all_plots..., size=(2000, 1200))
savefig(full_plot, "src/Plots/Truncated_Trajectories-tol=1e-3.pdf")
# savefig(full_plot, "src/Plots/Truncated_Trajectories-No_Singles.pdf")




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

# Construct confidence intervals
all_lcls = all_R0_hats - 1.96*all_SEs
all_ucls = all_R0_hats + 1.96*all_SEs


# Extract estimate and CI for optimal theta hat
ind_max = argmax(log_LR_estimates)
R0_hat = all_R0_hats[ind_max]
R0_lcl = all_lcls[ind_max]
R0_ucl = all_ucls[ind_max]

# Compute p-value for test that R0 <= 1
using Distributions
test_norm = Normal(0, all_SEs[ind_max])
p_val = 2*ccdf(test_norm, R0_hat - 1)


# ---------------------------------------------------------------------------- #
#                  Investigate Importance Weight Distribution                  #
# ---------------------------------------------------------------------------- #

theta_old = all_theta_inits[10]
theta_new = MCEM_update(theta_old, Y_vec, 10)

M_wt_var = 100


# Don't actually use this function. Better to run it line-by-line in the REPL
function get_weight_vars(theta_old, Y_vec, M_wt_var)
    all_Xs, all_wts = get_importance_sample(theta_old, Y_vec, M_wt_var, norm_weights=true)

    theta_new = MCEM_update(theta_old, Y_vec, all_Xs, all_wts)


    rho_hat = M_wt_var * sum(all_wts.^2)


    # Compute (2 + epsilon)th moment of test functions under proposal distribution
    epsilon = 0.1
    all_objectives_old = [complete_data_log_lik(theta_old, Y_vec, X) for X in all_Xs]
    all_objectives_new = [complete_data_log_lik(theta_new, Y_vec, X) for X in all_Xs]

    sq_plus_old = mean(abs.(all_objectives_old) .^ (2+epsilon))
    sq_plus_new = mean(abs.(all_objectives_new) .^ (2+epsilon))

    sq_wt_phi_old = mean((all_objectives_old .* all_wts) .^ 2)
    sq_wt_phi_new = mean((all_objectives_new .* all_wts) .^ 2)
end



# ---------------------------------------------------------------------------- #
#              Correlation Between Outbreak Size and Facility Size             #
# ---------------------------------------------------------------------------- #

# using CodecBzip2
# using RData


# imputed_data = load("src/data/Raw_Data/BC_LTHC_outbreaks_100Imputs.rda")["BC_LTHC_outbreaks_100Imputs"]

# all_cors = zeros(length(imputed_data));

# for i in eachindex(imputed_data)
#     a = imputed_data[i]["num_cases"]
#     b = imputed_data[i]["capacity"]
#     all_cors[i] = cor(a,b)
# end

# # Correlation between outbreak size and facility size is ~4.8%. This is computed on only the multi-day outbreaks.