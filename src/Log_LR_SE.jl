

theta_LR = [1.0, 0.1, 0.1]  # Reference parameter from which likelihood ratios are estimated
M_lik_rat = 500    # Number of samples to use for estimating the likelihood ratio from theta_LR to theta_hat

# Generate a single MC sample from reference parameter value
Random.seed!(1)
Xs_LR, unnorm_untrunc_log_wts_LR = get_importance_sample(theta_LR, Y_vec, M_lik_rat, norm_weights=false);
unnorm_log_wts_LR = truncate_weights(unnorm_untrunc_log_wts_LR)
wts_LR = normalize_weights(unnorm_log_wts_LR, truncate=false)

opt_log_liks = [complete_data_log_lik(final_theta_hat_MCEM, Y_vec, this_X_vec) for this_X_vec in Xs_LR]
ref_log_liks = [complete_data_log_lik(theta_LR, Y_vec, this_X_vec) for this_X_vec in Xs_LR]

all_log_weights = unnorm_log_wts_LR
all_log_LRs = opt_log_liks - ref_log_liks


# ---------------------------------------------------------------------------- #
#                 Plot trajectories of second empirical moments                #
# ---------------------------------------------------------------------------- #

# Compute cumulative log-sum of squared weights

cum_log_sum_sq_wts = [logsumexp(2 * all_log_weights[1:i]) for i in eachindex(all_log_weights)]

exp(maximum(unnorm_log_wts_LR) - minimum(unnorm_log_wts_LR))




"""
Computes the log of X_bar, where log_X is the logs of the terms in X.
"""
function robust_log_mean(log_X)
    M = length(log_X)
    return logsumexp(log_X) - log(M)
end


# ---------------------------------------------------------------------------- #
#                                 Compute means                                #
# ---------------------------------------------------------------------------- #

"""
log of mean (un-normalized) weight
"""
function get_l_w_bar(all_log_weights)
    return robust_log_mean(all_log_weights)
end

"""
log of mean (un-normalized) weight times LR
"""
function get_l_wR_bar(all_log_LRs, all_log_weights)
    return robust_log_mean(all_log_LRs + all_log_weights)
end


# ---------------------------------------------------------------------------- #
#                               Compute variances                              #
# ---------------------------------------------------------------------------- #

function get_l_wR_var(all_log_LRs, all_log_weights)
    A = robust_log_mean(2 .* all_log_LRs + 2 .* all_log_weights)
    B = get_l_wR_bar(all_log_LRs, all_log_weights)

    return logsubexp(A, 2*B)
end

function get_l_w_var(all_log_weights)
    A = robust_log_mean(2 .* all_log_weights)
    B = get_l_w_bar(all_log_weights)

    return logsubexp(A, 2*B)
end

function get_l_cov(all_log_LRs, all_log_weights)
    A = robust_log_mean(all_log_LRs + 2 .* all_log_weights)
    B = get_l_w_bar(all_log_weights)
    C = get_l_wR_bar(all_log_LRs, all_log_weights)

    D = B + C
    return logsubexp(A, D)
end



# ---------------------------------------------------------------------------- #
#      Assemble pieces into numerator and denominator of variance formula      #
# ---------------------------------------------------------------------------- #

"""
Numerator of our estimator for the variance of the log-likelihood ratio.
"""
function get_log_numerator(l_w_bar, l_wR_bar, l_sigma2_n, l_sigma2_d, l_sigma_nd)
    A = l_sigma2_n + 2 * l_w_bar
    B = log(2) + l_sigma_nd + l_w_bar + l_wR_bar
    C = l_sigma2_d + 2 * l_wR_bar

    D = logsumexp(A, C) # Add exp(A) and exp(C), then log

    log_numerator = logsubexp(D, B) # Subtract exp(B) from exp(D), then log

    return log_numerator
end



"""
Numerator of our estimator for the variance of the log-likelihood ratio.
"""
function bad_get_log_numerator(l_w_bar, l_wR_bar, l_sigma2_n, l_sigma2_d, l_sigma_nd)
    A = l_sigma2_n + 2 * l_w_bar
    B = log(2) + l_sigma_nd + l_w_bar + l_wR_bar
    C = l_sigma2_d + 2 * l_wR_bar

    D = logsumexp(A, B) # Add exp(A) and exp(C), then log

    log_numerator = logsubexp(D, C) # Subtract exp(B) from exp(D), then log

    return log_numerator
end



"""
Denominator of our estimator for the variance of the log-likelihood ratio.
Note: Does account for the MC sample size, M.
"""
function get_log_denominator(M, l_w_bar, l_wR_bar)
    return log(M) + 2*l_w_bar + 2*l_wR_bar
end




# ---------------------------------------------------------------------------- #
#                           Compute variance estimate                          #
# ---------------------------------------------------------------------------- #

function log_var_hat_log_LR(all_log_LRs, all_log_weights)
    M = length(all_log_weights)

    l_w_bar = get_l_w_bar(all_log_weights)
    l_wR_bar = get_l_wR_bar(all_log_LRs, all_log_weights)

    l_sigma2_n = get_l_w_var(all_log_weights)
    l_sigma2_d = get_l_wR_var(all_log_LRs, all_log_weights)
    l_sigma_nd = get_l_cov(all_log_LRs, all_log_weights)

    log_numerator = get_log_numerator(l_w_bar, l_wR_bar, l_sigma2_n, l_sigma2_d, l_sigma_nd)
    bad_log_numerator = bad_get_log_numerator(l_w_bar, l_wR_bar, l_sigma2_n, l_sigma2_d, l_sigma_nd)
    log_denominator = get_log_denominator(M, l_w_bar, l_wR_bar)

    return log_numerator - log_denominator
end

log_var_hat_log_LR(all_log_LRs, all_log_weights)





# ---------------------------------------------------------------------------- #
#                  Estimate SE of log LR estimate empirically                  #
# ---------------------------------------------------------------------------- #

M = M_lik_rat # Number of MC samples to generate for each estimate
B = 200  # Number of times to repeat estimaton (i.e. number of times to repeat MC)

Random.seed!(1)

all_log_LR_hats = zeros(B)

this_prog = Progress(B)

Threads.@threads for i in 1:B
# @showprogress for i in 1:B

    # Generate data and weights
    Xs_LR, unnorm_untrunc_log_wts_LR = get_importance_sample(theta_LR, Y_vec, M_lik_rat, norm_weights=false);
    unnorm_log_wts_LR = truncate_weights(unnorm_untrunc_log_wts_LR)
    wts_LR = normalize_weights(unnorm_log_wts_LR, truncate=false)

    # Compute log-likelihoods and log-weights
    ref_log_liks = [complete_data_log_lik(theta_LR, Y_vec, this_X_vec) for this_X_vec in Xs_LR]
    opt_log_liks = [complete_data_log_lik(final_theta_hat_MCEM, Y_vec, this_X_vec) for this_X_vec in Xs_LR]
    log_wts = log.(wts_LR)

    # Compute terms of our estimator
    all_terms = opt_log_liks - ref_log_liks + log_wts

    # Compute estimator
    log_LR_hat = logsumexp(all_terms)
    all_log_LR_hats[i] = log_LR_hat

    next!(this_prog)
end

SE_emp = std(all_log_LR_hats)