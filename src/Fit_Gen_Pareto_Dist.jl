
using ParetoSmooth
using Random
using Distributions
using LogExpFunctions

Random.seed!(1234)

M = 1000


raw_wts = exp.(rand(Normal(0, 10), M))

minimum(raw_wts)
maximum(raw_wts)


# --------------------------- Extract large weights -------------------------- #
M_keep = Int(floor(min(M/5, 3 * sqrt(M))))
large_wts = sort(raw_wts, rev=true)[1:M_keep]
large_wts_centered = large_wts .- minimum(large_wts)


# ------------------------- Fit GPD to large weights ------------------------- #
ParetoSmooth.gpd_fit(large_wts_centered, 1.0)


"""
Estimate parameters of the generalized Pareto distribution (GPD) using the method from Zhang and Stephens (2009).
"""
function fit_GPD(X)
    n = length(X)
    
    # ---------------------------- Compute theta terms --------------------------- #
    m = 20 + Int(floor(sqrt(n)))
    X_star = quantile(X, 0.25)
    X_max = maximum(X)

    all_theta_terms = 1/X_max .+ [1 - sqrt(m / (j-0.5)) for j in 1:m] ./ (3 * X_star)

    # -------------------- Compute weight for each theta term -------------------- #
    
    # Optimal k for each theta term
    all_ks = - [mean(log.(1 .- theta.*X)) for theta in all_theta_terms]

    # Log-likelihood for each theta term
    all_ls = zeros(m)
    for i in eachindex(all_ls)
        this_k = all_ks[i]
        this_theta = all_theta_terms[i]

        this_l = n * ( log(this_theta / this_k) + this_k - 1)
        all_ls[i] = this_l
    end

    # Likelihood weight for each theta term. Computed via log-sum-exp
    log_norm_const = logsumexp(all_ls)
    all_log_wts = all_ls .- log_norm_const
    all_wts = exp.(all_log_wts)

    # Estimate theta
    theta_hat = sum(all_wts .* all_theta_terms)

    # Estimate k based on theta_hat
    k_hat = - mean(log.(1 .- theta_hat .* X))

    # Estimate sigma based on theta_hat and k_hat
    sigma_hat = k_hat / theta_hat

    return k_hat, sigma_hat
end


fit_GPD(large_wts_centered)