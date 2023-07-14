


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
using StatsBase  
using Distributions
using Statistics


B = 100


# ---------------------------------------------------------------------------- #
#                             One Estimate Per Run                             #
# ---------------------------------------------------------------------------- #


all_theta_hats = Vector(undef, B);
all_theta_covs = Vector(undef, B);
all_R0_hats = Vector(undef, B);
all_R0_SEs = Vector(undef, B);


for i in 1:B
    @load "From_Cluster/output/estimates_short/estimates_short-$i.jld2" theta_hat theta_cov R0_hat R0_SE

    all_theta_hats[i] = theta_hat
    all_theta_covs[i] = theta_cov
    all_R0_hats[i] = R0_hat
    all_R0_SEs[i] = R0_SE
end


# ----------------------- Synthesize results for theta ----------------------- #

# Average of all theta_hats
theta_star_short = mean(all_theta_hats)

# Covariance between theta_hats
theta_between_runs_cov_short = cov(all_theta_hats)

# Average estimated asymptotic covariance
theta_mean_stat_cov_short = mean(all_theta_covs)
mean_stat_SE_short1 = sqrt.(diag(theta_mean_stat_cov_short))



# ------------------------- Synthesize results for R0 ------------------------ #

# Average of all R0_hats_
R0_star_short = mean(all_R0_hats)

# SD between R0s
R0_between_runs_SD_short = std(all_R0_hats)

# Pooled statistical SD
R0_mean_stat_SD_short = sqrt(mean(all_R0_SEs.^2))





# ---------------------------------------------------------------------------- #
#                          Multiple Estimates Per Run                          #
# ---------------------------------------------------------------------------- #

theta_hats_by_run = Vector()
theta_covs_by_run = Vector()
R0_hats_by_run = Vector()
R0_SEs_by_run = Vector()

for i in 1:B

    @load "From_Cluster/output/estimates_long/estimates_long-$i.jld2" all_theta_hats_close all_cov_hats_close all_R0s_close all_R0_SEs_close

    push!(theta_hats_by_run, all_theta_hats_close)
    push!(theta_covs_by_run, all_cov_hats_close)
    push!(R0_hats_by_run, all_R0s_close)
    push!(R0_SEs_by_run, all_R0_SEs_close)
end


all_num_estimates_long = length.(theta_hats_by_run)



# ----------------------- Synthesize results for theta ----------------------- #

all_theta_means = mean.(theta_hats_by_run)

# Pool all theta hats into a single estimator
theta_star_long = mean(all_theta_means) # Mean of all iterations' averages
other_theta_star_long = mean(vcat(theta_hats_by_run...))    # Mean of all estimates

# Relative difference between pooled estimates
norm(theta_star_long - other_theta_star_long) / norm(theta_star_long)
abs.(theta_star_long - other_theta_star_long) ./ theta_star_long

# Covariance of average estimate from run
between_runs_cov_long = cov(all_theta_means)


# Average estimated asymptotic covariance
mean_stat_cov_long = mean(vcat(theta_covs_by_run...))


# Average covariance between estimates from same run
all_between_starts_covs = cov.(theta_hats_by_run)
ind_nan = findall(isnan, getindex.(all_between_starts_covs, 1, 1))
deleteat!(all_between_starts_covs, ind_nan)
between_starts_cov_long = mean(all_between_starts_covs)

# Simple covariance of all estimates
global_cov_long = cov(vcat(theta_hats_by_run...))

theta_star_long
mean_stat_cov_long
between_runs_cov_long
between_starts_cov_long
global_cov_long

# Standard errors of estimates for each flavour of variability
SE_stat_long = sqrt.(diag(mean_stat_cov_long))
SE_imputs_long = sqrt.(diag(between_runs_cov_long))
SE_inits_long = sqrt.(diag(between_starts_cov_long))

SE_inits_long ./ SE_stat_long


# Test hypothesis that gamma <= 0
using Distributions, Random
gamma_hat = theta_star_long[2]
gamma_hat_SE = sqrt(mean_stat_cov_long[2, 2])
Z_score = gamma_hat / gamma_hat_SE
ccdf(Normal(), Z_score)


# ------------------------- Synthesize results for R0 ------------------------ #


R0_means = mean.(R0_hats_by_run)

# Pool all R0s into a single estimator
R0_star_long = mean(R0_means) # Mean of all iterations' averages
other_R0_star_long = mean(vcat(R0_hats_by_run...))    # Mean of all estimates

# Relative difference between pooled estimates
abs(R0_star_long - other_R0_star_long) / R0_star_long

# Compare against simple estimate induced by theta_star_long
theta_star_long[1] / theta_star_long[3]


# Covariance of average estimate from run
R0_between_runs_cov_long = std(R0_means)


# Average estimated asymptotic covariance
R0_mean_stat_cov_long = sqrt(mean(vcat(R0_SEs_by_run...).^2))


# Average covariance between estimates from same run
all_R0_between_starts_covs = cov.(R0_hats_by_run)
R0_ind_nan = findall(isnan, getindex.(all_R0_between_starts_covs, 1, 1))
deleteat!(all_R0_between_starts_covs, R0_ind_nan)
R0_between_starts_cov_long = mean(all_R0_between_starts_covs)


std(vcat(R0_hats_by_run...))
R0_mean_stat_cov_long
R0_between_runs_cov_long
R0_between_starts_cov_long


# Construct 95% CI for R0
R0_star_long - 1.96 * R0_mean_stat_cov_long
R0_star_long + 1.96 * R0_mean_stat_cov_long
p_val = 2 * ccdf(Normal(), abs(R0_star_long / R0_mean_stat_cov_long))



# ---------------------------------------------------------------------------- #
#                                Make some plots                               #
# ---------------------------------------------------------------------------- #

using Plots
using LaTeXStrings

# ----------------------------- Histogram for R0 ----------------------------- #
# --------------------------- One estimate per run --------------------------- #

histogram(all_R0_hats, bins=20, label="R0 estimates", xlabel=L"\hat{R}_0", ylabel="Frequency", title="Histogram of R0 estimates")




# ---------------------------------------------------------------------------- #
#                          Explore all initializations                         #
# ---------------------------------------------------------------------------- #

# Compare statistical variability at the maximizer to initialization variability, averaged across all imputations

all_estimates_across_imputs = Vector()

for i in 1:B

    @load "From_Cluster/output/sim_results/sim_results-$i.jld2" all_theta_hat_trajectories_MCEM all_theta_hats_MCEM all_cov_hats_MCEM all_Ms_MCEM all_num_iterations_MCEM

    push!(all_estimates_across_imputs, all_theta_hats_MCEM)
end


all_init_covs = cov.(all_estimates_across_imputs)
all_init_vars = diag.(all_init_covs)
deleteat!(all_init_vars, ind_nan)

all_init_SEs = Vector()
for i in eachindex(all_init_covs)
    push!(all_init_SEs, sqrt.(diag(all_init_covs[i])))
end

deleteat!(all_init_SEs, ind_nan)


all_init_covs_restricted = all_between_starts_covs
all_init_SEs_restricted = [sqrt.(diag(this_cov)) for this_cov in all_init_covs_restricted]
all_init_vars_restricted = diag.(all_init_covs_restricted)


all_stat_covs = all_theta_covs
all_stat_vars = diag.(all_stat_covs)
deleteat!(all_stat_vars, ind_nan)



# mean_init_SEs = mean(all_init_SEs)

# mean_init_over_stat_SE = mean_init_SEs.^2 ./ mean_stat_SE_short1.^2
# mean_init_over_stat_SE_restricted = SE_inits_long.^2 ./ mean_stat_SE_short1.^2


all_init_over_stat_vars = [all_init_vars[i] ./ all_stat_vars[i] for i in eachindex(all_init_vars)];
mean_init_over_stat_var = mean(all_init_over_stat_vars)
sqrt.(mean_init_over_stat_var)

all_init_over_stat_vars_rest = [all_init_vars_restricted[i] ./ all_stat_vars[i] for i in eachindex(all_init_vars_restricted)];
mean_init_over_stat_var_rest = mean(all_init_over_stat_vars_rest)
sqrt.(mean_init_over_stat_var_rest)







# ---------------------------------------------------------------------------- #
#               Multiple estimates per run, chosen by SE of diff               #
# ---------------------------------------------------------------------------- #


theta_hats_careful = Vector()
theta_covs_careful = Vector()
R0_hats_careful = Vector()
R0_SEs_careful = Vector()

for i in 1:B

    @load "From_Cluster/output/estimates_long_careful/estimates_long_careful-$i.jld2" all_theta_hats_close all_cov_hats_close all_R0s_close all_R0_SEs_close

    push!(theta_hats_careful, all_theta_hats_close)
    push!(theta_covs_careful, all_cov_hats_close)
    push!(R0_hats_careful, all_R0s_close)
    push!(R0_SEs_careful, all_R0_SEs_close)
end


all_num_estimates_careful = length.(theta_hats_careful)



# ----------------------- Synthesize results for theta ----------------------- #

all_theta_means = mean.(theta_hats_careful)

# Pool all theta hats into a single estimator
theta_star_careful = mean(all_theta_means) # Mean of all iterations' averages
other_theta_star_careful = mean(vcat(theta_hats_careful...))    # Mean of all estimates

# Relative difference between pooled estimates
norm(theta_star_careful - other_theta_star_careful) / norm(theta_star_careful)
abs.(theta_star_careful - other_theta_star_careful) ./ theta_star_careful

# Covariance of average estimate from run
between_runs_cov_careful = cov(all_theta_means)


# Average estimated asymptotic covariance
mean_stat_cov_careful = mean(vcat(theta_covs_careful...))


# Average covariance between estimates from same run
all_between_starts_covs = cov.(theta_hats_careful)
ind_nan = findall(isnan, getindex.(all_between_starts_covs, 1, 1))
deleteat!(all_between_starts_covs, ind_nan)
between_starts_cov_careful = mean(all_between_starts_covs)

# Simple covariance of all estimates
global_cov_careful = cov(vcat(theta_hats_careful...))

theta_star_careful
mean_stat_cov_careful
between_runs_cov_careful
between_starts_cov_careful
global_cov_careful



# Standard errors of estimates for each flavour of variability
SE_stat_careful = sqrt.(diag(mean_stat_cov_careful))
SE_imputs_careful = sqrt.(diag(between_runs_cov_careful))
SE_inits_careful = sqrt.(diag(between_starts_cov_careful))

SE_inits_careful ./ SE_stat_careful

# Pooled SDs
sqrt.(SE_stat_careful.^2 + SE_imputs_careful.^2 + SE_inits_careful.^2)



# Test hypothesis that gamma <= 0
using Distributions, Random
gamma_hat = theta_star_careful[2]
gamma_hat_SE_stat = sqrt(mean_stat_cov_careful[2, 2])
Z_score_stat = gamma_hat / gamma_hat_SE_stat
ccdf(Normal(), Z_score_stat)

gamma_hat_SE_pooled = sqrt(SE_stat_careful[2]^2 + + SE_imputs_careful[2]^2 + SE_inits_careful[2]^2)
Z_score_pooled = gamma_hat / gamma_hat_SE_pooled
ccdf(Normal(), Z_score_pooled)


# ------------------------- Synthesize results for R0 ------------------------ #


R0_means = mean.(R0_hats_careful)

# Pool all R0s into a single estimator
R0_star_careful = mean(R0_means) # Mean of all iterations' averages
other_R0_star_careful = mean(vcat(R0_hats_careful...))    # Mean of all estimates

# Relative difference between pooled estimates
abs(R0_star_careful - other_R0_star_careful) / R0_star_careful

# Compare against simple estimate induced by theta_star_careful
theta_star_careful[1] / theta_star_careful[3]


# Covariance of average estimate from run
R0_between_runs_cov_careful = std(R0_means)


# Average estimated asymptotic covariance
R0_mean_stat_cov_careful = sqrt(mean(vcat(R0_SEs_careful...).^2))


# Average covariance between estimates from same run
all_R0_between_starts_covs = cov.(R0_hats_careful)
R0_ind_nan = findall(isnan, getindex.(all_R0_between_starts_covs, 1, 1))
deleteat!(all_R0_between_starts_covs, R0_ind_nan)
R0_between_starts_cov_careful = sqrt(mean(all_R0_between_starts_covs))


std(vcat(R0_hats_careful...))
R0_mean_stat_cov_careful
R0_between_runs_cov_careful
R0_between_starts_cov_careful


# Construct 95% CI for R0
R0_star_careful - 1.96 * R0_mean_stat_cov_careful
R0_star_careful + 1.96 * R0_mean_stat_cov_careful
2 * ccdf(Normal(), abs(R0_star_careful / R0_mean_stat_cov_careful))

R0_SE_pooled = sqrt(R0_mean_stat_cov_careful^2 + R0_between_runs_cov_careful^2 + R0_between_starts_cov_careful^2)
R0_star_careful - 1.96 * R0_SE_pooled
R0_star_careful + 1.96 * R0_SE_pooled
2 * ccdf(Normal(), abs(R0_star_careful / R0_SE_pooled))