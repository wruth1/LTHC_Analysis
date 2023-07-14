

export one_X_given_Y, one_MC_draw
export multiple_MC_draws, get_importance_sample

export Q_MCEM, complete_data_log_lik_increment, Q_MCEM_increment
export score_MCEM, Hessian_MCEM

export MCEM_update

export MC_complete_cond_info, MC_expect_sq_score, MC_score_cov
export MCEM_obs_data_info_formula, MCEM_COV_formula, MCEM_SE_formula






# ---------------------------------------------------------------------------- #
#                              Conditional Sampler                             #
# ---------------------------------------------------------------------------- #


#! This function prints an indicator every time it needs to reset due to a prematurely extinct trajectory. When prioritizing performance over checking stability, comment out the indicated lines.
#* Lines to comment are indicated in this color
"""
Generate a sequence of durations conditional on a sequence of case counts. Returns the sequence and its probability weight (obtained from the conditioning event).
"""
function one_X_given_Y(Y, phi_0, gamma, lambda)
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
                    if jump_count % 50 == 0
                        println("Jump to time_1 number $jump_count at time $t of $t_max")   #**************************
                    end
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
function one_MC_draw(Y_vec, phi_0, gamma, lambda)
    n = length(Y_vec)
    X_vec = Vector(undef, n)
    log_weight = 0.0

    for i in 1:n
        Y = Y_vec[i]
        output = one_X_given_Y(Y, phi_0, gamma, lambda)
        X_vec[i] = output["X"]
        log_weight += output["log_weight"]
    end

    return X_vec, log_weight
end

"""
Generate a sample of M iid duration trajectories for each Y conditional on that Y. Returns an Mxn array of X trajectories, and a vector of the MC samples' (standardized and non-log) weights.
Optionally, can return the un-normalized log-weights. Caution: This is rarely a good idea.
"""
function multiple_MC_draws(M, Y_vec, theta; norm_weights=true)
    n = length(Y_vec)

    # all_X_vecs = Array{Any}(undef, M, n)
    all_X_vecs = Vector{Any}(undef, M)
    all_log_weights_raw = Vector(undef, M)

    for i in 1:M
        all_X_vecs[i], all_log_weights_raw[i] = one_MC_draw(Y_vec, theta...)
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
function get_importance_sample(theta, Y_vec, M; norm_weights=true)
    return multiple_MC_draws(M, Y_vec, theta, norm_weights=norm_weights)
end


# ---------------------------------------------------------------------------- #
#                           Objective Function, etc.                           #
# ---------------------------------------------------------------------------- #


"""
Objective function of MCEM.
Supply the a pre-constructed MC sample.
"""
function Q_MCEM(theta, Y_vec, all_X_vecs, all_weights)
    all_lik = [complete_data_log_lik(theta, Y_vec, X) for X in all_X_vecs]

    return dot(all_lik, all_weights)
end

# Q_MCEM(theta1, Y, all_Xs1, all_weights1)

"""
Objective function of MCEM.
MC sample generated inside the function.
"""
function Q_MCEM(theta, Y, theta_old, M, return_X)
    all_Xs, all_weights = get_importance_sample(theta_old, Y, M)

    if !return_X
        return Q_MCEM(theta, Y, all_Xs, all_weights)
    else
        return Q_MCEM(theta, Y, all_Xs, all_weights), all_Xs, all_weights
    end
end

"""
Compute the difference in log-likelihoods between two values of theta.
"""
function complete_data_log_lik_increment(theta_new, theta_old, Y, X)
    A = complete_data_log_lik(theta_new, Y, X)
    B = complete_data_log_lik(theta_old, Y, X)
    return A - B
end


"""
Improvement in MCEM objective function.
"""
function Q_MCEM_increment(theta_new, theta_old, Y_vec, all_X_vecs, all_weights)
    all_lik_increments = [complete_data_log_lik_increment(theta_new, theta_old, Y_vec, X) for X in all_X_vecs]

    return dot(all_lik_increments, all_weights)
end


function score_MCEM(theta, Y_vec, all_X_vecs, all_weights)
    all_scores = [complete_data_score(theta, Y_vec, X) for X in all_X_vecs]

    return sum(all_scores .* all_weights)
end

function Hessian_MCEM(theta, Y_vec, all_X_vecs, all_weights)
    all_Hessians = [complete_data_Hessian(theta, Y_vec, X) for X in all_X_vecs]

    return sum(all_Hessians .* all_weights)
end


# ---------------------------------------------------------------------------- #
#                             MCEM Update Functions                            #
# ---------------------------------------------------------------------------- #

"""
Compute the next estimate of theta using MCEM.
MC sample is provided as an argument.
max_time is the (approximate) maximum time (in seconds) to run BFGS when Newton's method is not appropriate.
"""
function MCEM_update(theta_init::Vector, Y_vec::Vector, all_X_vecs::Vector, all_weights::Vector; max_time=10)

    # ------------------------ Build functions to optimize ----------------------- #
    #Note: Optim only does minimization. We therefore multiply our likelihood functions by -1
    function objective(theta)
        return -Q_MCEM(theta, Y_vec, all_X_vecs, all_weights)
    end

    function grad!(G, theta)
        G[:] = -score_MCEM(theta, Y_vec, all_X_vecs, all_weights)
    end

    function hess!(H, theta)
        H[:,:] = -Hessian_MCEM(theta, Y_vec, all_X_vecs, all_weights)
    end


    # println("Optimizing MCEM objective function...")


    # Check whether Hessian is positive definite. If not, use BFGS instead.
    p = length(theta_init)
    this_hess = hess!(zeros(p, p), theta_init)
    pos_def_check = isposdef(this_hess)

    if pos_def_check
        # Newton alone, custom stopping criterion
        Newton_iterate = optimize(objective, grad!, hess!, par_lower, par_upper, theta_init, IPNewton(),
        Optim.Options(show_trace=false, x_tol = 0, f_tol = 0, g_tol = 1e-6, iterations=1))# time_limit = 30))
        optim_iterate = Newton_iterate
    else
        println("Hessian is not positive definite. Using BFGS instead.")
        BFGS_iterate = optimize(objective, grad!, par_lower, par_upper, theta_init, Fminbox(BFGS()), 
        Optim.Options(show_trace=false, x_tol = 0, f_tol = 0, g_tol = 1e-6, time_limit = max_time))# iterations=1
        optim_iterate = BFGS_iterate
    end

    # Newton alone, default stopping criterion
    # Newton_iterate = optimize(objective, grad!, par_lower, par_upper, theta_old, IPNewton(),
    # Optim.Options(show_trace=false, x_tol = 0, f_tol = 0, g_tol = 1e-6))#, time_limit = 2))

    

    # # --------------- Check for descent. If not, run BFGS instead. --------------- #
    # objective_init = objective(theta_init)
    # if Newton_iterate.minimum > objective_init
    #     # Run BFGS with fixed runtime. Keep re-running from endpoint until improvement is detected.
    #     println("Newton failed to improve. Trying BFGS...")
    #     max_time = 10   # seconds
    #     theta_hat_BFGS = theta_init
    #     improved = false

    #     while !improved
    #         println("Running BFGS"...)
    #         BFGS_iterate = optimize(objective, grad!, par_lower, par_upper, theta_hat_BFGS, Fminbox(BFGS()),
    #         Optim.Options(show_trace=false, x_tol = 0, f_tol = 0, g_tol = 1e-6, time_limit = max_time))# iterations=1

    #         theta_hat_BFGS = Optim.minimizer(BFGS_iterate)

    #         improved = BFGS_iterate.minimum < objective_init
    #     end
    #     optim_iterate = BFGS_iterate
    # end
    
    # Compound BFGS, then Newton
    # BFGS_iterate = optimize(objective, grad!, par_lower, par_upper, theta_init, Fminbox(BFGS()),
    # Optim.Options(show_trace=false, x_tol = 0, f_tol = 0, g_tol = 1e-6, time_limit = 25))
    # BFGS_estimate = Optim.minimizer(BFGS_iterate)
    # Newton_iterate = optimize(objective, grad!, hess!, par_lower, par_upper, BFGS_estimate, IPNewton(),
    # Optim.Options(show_trace=false, x_tol = 0, f_tol = 0, g_tol = 1e-6, time_limit = 5))
    # optim_iterate = Newton_iterate

    # # Nelder-Mead alone, custom stopping criterion
    # Nelder_Mead_iterate = optimize(objective, par_lower, par_upper, theta_init, Fminbox(NelderMead()),
    # Optim.Options(show_trace=false, x_tol = 0, f_tol = 0, g_tol = 1e-12), time_limit = 30)
    # optim_iterate = Nelder_Mead_iterate
    
    # Newton_theta_hat = Optim.minimizer(Newton_iterate)
    # NM_theta_hat = Optim.minimizer(Nelder_Mead_iterate)
    
    # Netwon_min_val = Optim.minimum(Newton_iterate)
    # NM_min_val = Optim.minimum(Nelder_Mead_iterate)

    # BFGS alone, custom stopping criterion
    # BFGS_iterate = optimize(objective, grad!, par_lower, par_upper, theta_init, Fminbox(BFGS()),
    # Optim.Options(show_trace=false, x_tol = 0, f_tol = 0, g_tol = 1e-6, time_limit = 30))
    # optim_iterate = BFGS_iterate

    theta_hat = Optim.minimizer(optim_iterate)
    return theta_hat
end



"""
Compute the next estimate of theta using MCEM.
MC sample is generated internally and optionally returned.
"""
function MCEM_update(theta_old::Vector, Y_vec::Vector, M::Number; return_X=false)
    all_X_vecs, all_weights = get_importance_sample(theta_old, Y_vec, M)
    theta_hat = MCEM_update(theta_old, Y_vec, all_X_vecs, all_weights)

    if !return_X
        return theta_hat
    else
        return theta_hat, all_X_vecs, all_weights
    end
end









# ---------------------------------------------------------------------------- #
#                                Standard error                                #
# ---------------------------------------------------------------------------- #

"""
Estimate the conditional expectation of the complete data observed information, given Y=y.
MC sample provided as argument.
"""
function MC_complete_cond_info(theta, Y_vec, all_X_vecs, all_weights)
    all_infos = -[complete_data_Hessian(theta, Y_vec, X_vec) for X_vec in all_X_vecs]
    return sum(all_infos .* all_weights)
end

"""
Estimate the conditional expectation of the complete data observed information, given Y=y.
MC sample generated internally.
"""
function MC_complete_cond_info(theta, Y_vec, theta_old, M, return_X)
    all_X_vecs, all_weights = get_importance_sample(theta_old, Y_vec, M)
    
    if !return_X
        return MC_complete_cond_info(theta, Y_vec, all_X_vecs, all_weights)
    else
        return MC_complete_cond_info(theta, Y_vec, all_X_vecs, all_weights), all_X_vecs, all_weights
    end
end


"""
Estimate the conditional expectation of the outer product between the complete data score and itself, given Y=y.
MC sample provided as argument.
"""
function MC_expect_sq_score(theta, Y_vec, all_X_vecs, all_weights)
    all_scores = [complete_data_score(theta, Y_vec, X_vec) for X_vec in all_X_vecs]
    all_sq_scores = [score * Transpose(score) for score in all_scores]
    return sum(all_sq_scores .* all_weights)
end

"""
Estimate the conditional expectation of the outer product between the complete data score and itself, given Y=y.
MC sample generated internally.
"""
function MC_expect_sq_score(theta, Y_vec, theta_old, M, return_X)
    all_X_vecs, all_weights = get_importance_sample(theta_old, Y_vec, M)
    
    if !return_X
        return MC_expect_sq_score(theta, Y_vec, all_X_vecs, all_weights)
    else
        return MC_expect_sq_score(theta, Y_vec, all_X_vecs, all_weights), all_X_vecs, all_weights
    end
end


"""
Estimate the conditional covariance of the score vector, given Y=y.
MC sample provided as argument.
"""
function MC_score_cov(theta, Y_vec, all_X_vecs, all_weights)
    all_scores = [complete_data_score(theta, Y_vec, X_vec) for X_vec in all_X_vecs]
    
    mean_score = sum(all_scores .* all_weights)
    sq_mean_score = mean_score * mean_score'


    all_sq_scores = [score * score' for score in all_scores]
    mean_sq_score = sum(all_sq_scores .* all_weights)

    cov_hat = mean_sq_score - sq_mean_score
    return cov_hat
end

"""
Estimate the conditional covariance of the score vector, given Y=y.
MC sample generated internally.
"""
function MC_score_cov(theta, Y_vec, theta_old, M, return_X)
    all_X_vecs, all_weights = get_importance_sample(theta_old, Y_vec, M)
    
    if !return_X
        return MC_score_cov(theta, Y_vec, all_X_vecs, all_weights)
    else
        return MC_score_cov(theta, Y_vec, all_X_vecs, all_weights), all_X_vecs, all_weights
    end
end



"""
Estimate the observed data information using MCEM.
MC sample provided as argument.
Note: Formula is only valid at a stationary point of EM.
"""
function MCEM_obs_data_info_formula(theta, Y_vec, all_X_vecs, all_weights)
    # println("Computing expected complete data information...")
    A = MC_complete_cond_info(theta, Y_vec, all_X_vecs, all_weights)
    # println("Computing score covariance...")
    B = MC_score_cov(theta, Y_vec, all_X_vecs, all_weights)

    output = A - B
    return output
end


"""
Estimate the observed data information using MCEM.
MC sample generated internally.
Note: Formula is only valid at a stationary point of EM.
"""
function MCEM_obs_data_info_formula(theta, Y_vec, theta_old, M, return_X)
    all_X_vecs, all_weights = get_importance_sample(theta_old, Y_vec, M)
    
    if !return_X
        return MCEM_obs_data_info_formula(theta, Y_vec, all_X_vecs, all_weights)
    else
        return MCEM_obs_data_info_formula(theta, Y_vec, all_X_vecs, all_weights), all_X_vecs, all_weights
    end
end


"""
Estimate the covariance matrix of the MCEM estimate of theta.
MC sample provided as argument.
Note: Formula is only valid at a stationary point of EM.
"""
function MCEM_COV_formula(theta, Y_vec, all_X_vecs, all_weights)
    return inv(MCEM_obs_data_info_formula(theta, Y_vec, all_X_vecs, all_weights))
end

"""
Estimate the covariance matrix of the MCEM estimate of theta.
MC sample generated internally.
Note: Formula is only valid at a stationary point of EM.
"""
function MCEM_COV_formula(theta, Y_vec, theta_old, M, return_X)
    all_X_vecs, all_weights = get_importance_sample(theta_old, Y_vec, M)
    
    if !return_X
        return MCEM_COV_formula(theta, Y_vec, all_X_vecs, all_weights)
    else
        return MCEM_COV_formula(theta, Y_vec, all_X_vecs, all_weights), all_X_vecs, all_weights
    end
end


"""
Estimate the standard error of the MCEM estimate of theta. Return a vector of the same length as theta.
MC sample provided as argument.
Note: Formula is only valid at a stationary point of EM.
"""
function MCEM_SE_formula(theta, Y_vec, all_X_vecs, all_weights)
    return sqrt.(diag(MCEM_COV_formula(theta, Y_vec, all_X_vecs, all_weights)))
end


"""
Estimate the standard error of the MCEM estimate of theta. Return a vector of the same length as theta.
MC sample generated internally.
Note: Formula is only valid at a stationary point of EM.
"""
function MCEM_SE_formula(theta, Y_vec, theta_old, M, return_X)
    all_X_vecs, all_weights = get_importance_sample(theta_old, Y_vec, M)

    if !return_X
        return MCEM_SE_formula(theta, Y_vec, all_X_vecs, all_weights)
    else
        return MCEM_SE_formula(theta, Y_vec, all_X_vecs, all_weights), all_X_vecs, all_weights
    end
end







# ---------------------------------------------------------------------------- #
#                         Estimate EM Convergence Rate                         #
# ---------------------------------------------------------------------------- #

# I'm going to compute the thing which gets larger as convergence is faster
# I.e. I define convergence rate as the inverse complete data information times the observed data information. See McL & K

function MC_EM_conv_rate_mat(theta, Y_vec, all_X_vecs, all_weights)
    A = MC_complete_cond_info(theta, Y_vec, all_X_vecs, all_weights)
    B = MC_score_cov(theta, Y_vec, all_X_vecs, all_weights)
    
    return I - inv(A)*B
end


function MC_EM_conv_rate_mat(theta, Y_vec, theta_old, M, return_X)
    all_X_vecs, all_weights = get_importance_sample(theta_old, Y_vec, M)
    
    if !return_X
        return MC_EM_conv_rate_mat(theta, Y_vec, all_X_vecs, all_weights)
    else
        return MC_EM_conv_rate_mat(theta, Y_vec, all_X_vecs, all_weights), all_X_vecs, all_weights
    end
end


# theta_hat = all_theta_hats_MCEM[4]
# q = MC_EM_conv_rate_mat(theta_hat, Y_vec, theta_hat, 50, false)

# eigen(q)