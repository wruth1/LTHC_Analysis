
export cov2cor
export get_ESS, get_proposal_dist
export truncate_weights, normalize_weights
export get_proposal_r, get_proposal_p
export check_active
export get_phi
export time_to_next_case
export check_compatibility
export expand_grid, unfold

export get_omega_during_traj, get_omega_after_traj
export get_omega, get_omega_traj, omega_traj_many_Xs, omega_traj_many_Ys, omega_traj_one_sample


# ---------------------------------------------------------------------------- #
#                             MISC Helper Functions                            #
# ---------------------------------------------------------------------------- #



"""
Converts a covariance matrix to the corresponding correlation matrix.
"""
function cov2cor(cov_mat)
    D = diagm(1 ./ sqrt.(abs.(diag(cov_mat))))
    return D * cov_mat * D
end




# ---------------------------------------------------------------------------- #
#                             Proposal Distribution                            #
# ---------------------------------------------------------------------------- #

"""
Compute effective sample size from the given set of weights.
"""
function get_ESS(weights)
    return 1 / sum(weights.^2)
end

# As proposal, we use a negative binomial random variable (plus 1 to match the support of X) 
# Its mean is the same as X (i.e. 1/lambda), and its variance is eta^2 times that of X (i.e. eta^2 * (1-lambda) / lambda^2)



"""
Return the r parameter used to generate proposals for X.
"""
function get_proposal_r(lambda, eta)
    return (1 - lambda) / (eta^2 - lambda)
end


"""
Return the p parameter used to generate proposals for X.
"""
function get_proposal_p(lambda, eta)
    return lambda / eta^2
end


"""
Return the distribution used to generate proposals for X.
"""
function get_proposal_dist(lambda, eta)
    r = get_proposal_r(lambda, eta)
    p = get_proposal_p(lambda, eta)

    return ProposalDistribution(r, p)
end


# """
# Normalize and exponentiate the given log weights.
# """
# function normalize_weights(all_log_weights)
#     log_norm_const = logsumexp(all_log_weights)
#     return exp.(all_log_weights .- log_norm_const)
# end

"""
Truncate the log-weights based on the global recommendation of Ionides (2008).
"""
function truncate_weights(all_log_weights)
    log_norm_const = logsumexp(all_log_weights)

    this_M = length(all_log_weights)
    tau = log_norm_const - log(this_M)/2

    all_log_trunc_weights = [min(w, tau) for w in all_log_weights]
    
    return all_log_trunc_weights
end

"""
Truncate, then normalize and exponentiate the given log weights.
"""
function normalize_weights(all_log_weights; truncate=true)

    log_norm_const = logsumexp(all_log_weights)
    if truncate
        all_log_trunc_weights = truncate_weights(all_log_weights)
        log_trunc_norm_const = logsumexp(all_log_trunc_weights)

        return exp.(all_log_trunc_weights .- log_trunc_norm_const)
    else
        return exp.(all_log_weights .- log_norm_const)
    end
end




"""
Check if the individual reprensented by duration x is still active after the specified number of days.
Note: For compatibility with the rest of the algorithm, return is an integer.
"""
function check_active(x, days)
    if (x - days >= 0)
        return 1
    else
        return 0
    end
end



"""
Compute the offspring parameter, phi, at the specified time point.
"""
function get_phi(phi_0, gamma, t)
    phi = phi_0 * exp(- gamma * t)
    return phi
end


"""
Gets the number of days from day t until the next observed case.
"""
function time_to_next_case(Y, t)
    t_next = findfirst(x -> x>0, Y[(t+1):end])
    return t_next
end



"""
Check if the outbreak goes extinct before generating all observed cases.
"""
function check_compatibility(Y, X)
    t_max = length(Y)

    if t_max >= 2 # Problem can only arise if outrbeak lasts longer than 1 day
        for t in 2:t_max
            if Y[t] > 0 # Problem can only arise on days with cases
                # Check that there are active infections to generate new cases today
                omega = get_omega(Y, X, t)
                if omega == 0
                    return false    # If no active infections, then extinction has occurred
                end
            end
        end
    end
    # If end of trajectory is reached without detecting extinction, then we're good
    return true
end


function expand_grid(X...)
    vec(collect(Iterators.product(X...)))
end

"""
Turn a nested array into a 1d vector.
"""
function unfold(X)
    collect(Iterators.flatten(X))
end

# ---------------------------------------------------------------------------- #
#         Compute omega, the number of active cases on a particular day        #
# ---------------------------------------------------------------------------- #

"""
Get the number of active infections at the start of day `t`. Only works with `t <= length(Y)`.
"""
function get_omega_during_traj(X, t)
    omega = 0
    for s in 1:(t-1)
        for i in eachindex(X[s])
            omega += check_active(X[s][i], t - s)
        end
    end
    return omega
end

"""
Get the number of active infections at the start of day t_max. Only works with `t_max > length(Y)`.
"""
function get_omega_after_traj(Y, X, t)
    omega = 0
    for s in 1:length(Y)
        for i in 1:length(X[s])
            omega += check_active(X[s][i], t - s)
        end
    end
        
    return omega
end

"""
Get the number of active infections at the start of day t.
"""
function get_omega(Y, X, t)
    if t <= length(Y)
        return get_omega_during_traj(X, t)
    else
        return get_omega_after_traj(Y, X, t)
    end
end

"""
Get the number of active infections at the end of the provided trajectory.
"""
function get_omega(Y, X)
    return get_omega(Y, X, length(Y) + 1)
end

"""
Get the number of active infections at each time point. Stops when the number of active infections is 0.
Note: The omega trajectory will be longer than the (Y, X) trajectory.
NB: There is no omega on day 1. For consistent indexing, we put "nothing" in the first position.
"""
function get_omega_traj(Y, X)
    omega_traj = Vector{Any}(nothing, 1)
    t = 2
    omega = 1
    while omega > 0
        omega = get_omega(Y, X, t)
        push!(omega_traj, omega)
        t += 1
    end
    return omega_traj
end

"""
Get the active infection trajectory for each provided duration trajectory.
"""
function omega_traj_many_Xs(Y, Xs)
    all_omega_trajs = []
    for X in Xs
        push!(all_omega_trajs, get_omega_traj(Y, X))
    end
    return all_omega_trajs
end

"""
Vector of active infection trajectories for each provided Y.
"""
function omega_traj_many_Ys(Y_vec, all_X_vecs)
    return [omega_traj_many_Xs(Y_vec[i], all_X_vecs[i]) for i in eachindex(Y_vec)];
end

function omega_traj_one_sample(Y_vec, X_traj_vec)
    return [get_omega_traj(y, x) for (y, x) in zip(Y_vec, X_traj_vec)]
end




