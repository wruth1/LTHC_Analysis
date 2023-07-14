export get_ASE, check_ascent, check_for_termination, get_next_MC_size
export Ascent_MCEM_Control
export ascent_MCEM_update, full_ascent_MCEM_iteration
export run_ascent_MCEM

#* Note: To perform a single run, un-comment the lines commented in this color


# ---------------------------------------------------------------------------- #
#                 Asymptotic SE of increment in MCEM objective                 #
# ---------------------------------------------------------------------------- #



"""
Estimate the asymptotic standard error of the increment in MCEM objective function.
"""
function get_ASE(theta_new, theta_old, Y_vec, all_X_vecs, all_weights)
    M = length(all_X_vecs)

    # theta_new = theta_0
    # theta_old = theta_init1

    all_lik_increments = [complete_data_log_lik_increment(theta_new, theta_old, Y_vec, X_vec) for X_vec in all_X_vecs]

    A = dot(all_weights, all_lik_increments)^2 
    
    B1 = dot(all_weights.^2, all_lik_increments.^2)
    B2 = dot(all_weights, all_lik_increments)^2
    B = B1 / B2

    C1 = dot(all_weights.^2, all_lik_increments)
    C2 = dot(all_weights, all_lik_increments)
    C = 2 * C1 / C2

    D = sum(all_weights.^2)

    ASE2 = A * (B - C + D)

    # Deal with possible negative standard error
    if ASE2 < sqrt(eps())
        ASE2 = sqrt(eps())
    end

    ASE = sqrt(ASE2)
    return ASE
end


# """
# Estimate the asymptotic standard error of the increment in MCEM objective function directly from the log-likelihood increments.
# """
# function get_resamp_ASE(theta_new, theta_old, Y, all_resamp_Xs)

#     # test_all_Xs = wsample(all_Xs, all_weights, M, replace = true)

#     M = length(all_resamp_Xs)
#     test_lik_increments = [complete_data_log_lik_increment(theta_new, theta_old, Y, X) for X in all_resamp_Xs]
#     return std(test_lik_increments)/sqrt(M)

# end

# ---------------------------------------------------------------------------- #
#                    Investigate ASE formula for IID sample                    #
# ---------------------------------------------------------------------------- #

# Random.seed!(1)

# all_Xs_iid = sample_X_given_Y(theta1, Y, M)
# all_weights_iid = ones(M) / M
# theta_new = MCEM_update(Y, all_Xs_iid, all_weights_iid)
# theta_old = theta1


# all_lik_increments = [complete_data_log_lik_increment(theta_new, theta_old, Y, X) for X in all_Xs_iid]

# M = length(all_Xs_iid)

# A = dot(all_weights_iid, all_lik_increments)^2 
# A_iid = mean(all_lik_increments)^2

# B1 = dot(all_weights_iid.^2, all_lik_increments.^2)
# B2 = dot(all_weights_iid, all_lik_increments)^2
# B = B1 / B2
# B_iid = mean(all_lik_increments.^2) / (M * mean(all_lik_increments)^2)

# C1 = dot(all_weights_iid.^2, all_lik_increments)
# C2 = dot(all_weights_iid, all_lik_increments)
# C = 2 * C1 / C2
# C_iid = 2/M

# D = sum(all_weights_iid.^2)
# D_iid = 1/M

# ASE2 = A * (B - C + D) / M
# ASE2_iid = A_iid * (B_iid - C_iid + D_iid) / M

# ASE = sqrt(ASE2)
# ASE_iid = sqrt(ASE2_iid)



# get_resamp_ASE(theta_new, theta1, Y, all_Xs_iid)
# get_ASE(theta_new, theta1, Y, all_Xs_iid, all_weights_iid)


# ---------------------------------------------------------------------------- #
#             Various functions for within an ascent MCEM iteration            #
# ---------------------------------------------------------------------------- #

"""
Construct a lower confidence bound for improvement in the EM objective. Return true if this bound is positive.
Optionally returns the computed lower confidence bound.
"""
function check_ascent(theta_new, theta_old, Y_vec, all_X_vecs, all_weights, alpha; return_lcl = false)
    Q_increment = Q_MCEM_increment(theta_new, theta_old, Y_vec, all_X_vecs, all_weights)
    # if Q_increment < 0
    #     error("Theta hat decreases MCEM objective.")
    # end
    ASE = get_ASE(theta_new, theta_old, Y_vec, all_X_vecs, all_weights)
    multiplier = quantile(Normal(), 1 - alpha)

    lcl = Q_increment - multiplier*ASE

    if !return_lcl
        return lcl > 0
    else
        return lcl > 0, lcl
    end
end


"""
Construct an upper confidence bound for the EM increment. If smaller than the specified absolute tolerance, return true.
Optionally returns the computed upper confidence bound.
"""
function check_for_termination(theta_new, theta_old, Y_vec, all_X_vecs, all_weights, alpha, atol; diagnostics = false)
    Q_increment = Q_MCEM_increment(theta_new, theta_old, Y_vec, all_X_vecs, all_weights)
    ASE = get_ASE(theta_new, theta_old, Y_vec, all_X_vecs, all_weights)
    multiplier = quantile(Normal(), 1 - alpha)

    ucl = Q_increment + multiplier*ASE

    if !diagnostics
        return ucl < atol
    else
        return ucl < atol, ucl, ASE
    end
end



"""
Compute sample size for next iteration of MCEM.
Intermediate quantities are pre-computed.
"""
function get_next_MC_size(MCEM_increment, M_old, ASE, alpha1, alpha2)
    multiplier1 = quantile(Normal(), 1 - alpha1)
    multiplier2 = quantile(Normal(), 1 - alpha2)

    M_proposed = ceil(ASE^2 * (multiplier1 + multiplier2)^2 / MCEM_increment^2)

    M_new = Int(max(M_proposed, M_old))

    return M_new
end


"""
Compute sample size for next iteration of MCEM.
Intermediate quantities are computed inside the function.
"""
function get_next_MC_size(theta_new, theta_old, Y_vec, all_X_vecs, all_weights, M_old, alpha1, alpha2)
    Q_increment = Q_MCEM_increment(theta_new, theta_old, Y_vec, all_X_vecs, all_weights)
    ASE = get_ASE(theta_new, theta_old, Y_vec, all_X_vecs, all_weights)

    return get_next_MC_size(Q_increment, M_old, ASE, alpha1, alpha2)
end





# ---------------------------------------------------------------------------- #
#                      Single ascent-based MCEM iteration                      #
# ---------------------------------------------------------------------------- #


"""
Object which contains the static parameters for MCEM
Note: M is excluded because it varies across iterations.
"""
mutable struct Ascent_MCEM_Control
    alpha1::Float64 # confidence level for checking whether to augment MC sample size
    alpha2::Float64 # confidence level for computing next step's initial MC sample size
    alpha3::Float64 # confidence level for checking whether to terminate MCEM
    k::Float64      # when augmenting MC sample, add M/k new points
    atol::Float64   # absolute tolerance for checking whether to terminate MCEM
end



"""
Perform a single iteration of ascent-based MCEM. Uses a level-alpha1 confidence bound to check for ascent. If not, augments the MC sample with M/k new points and tries again.
Options for diagnostics are included to check whether the MC sample was augmented.
"""
function ascent_MCEM_update(theta_old, Y_vec, M, alpha1, k; return_MC_size = false, return_X=false, diagnostics = false)
    
    
    
    diagnostics = true
        


    
    all_X_vecs, all_raw_weights = get_importance_sample(theta_old, Y_vec, M; norm_weights=false);
    all_weights = normalize_weights(all_raw_weights);
    # get_ESS(all_weights)

    theta_new = MCEM_update(theta_old, Y_vec, all_X_vecs, all_weights);

    all_lcls = []

    if diagnostics
        ascended, lcl = check_ascent(theta_new, theta_old, Y_vec, all_X_vecs, all_weights, alpha1; return_lcl=true)
        push!(all_lcls, lcl)
    else
        ascended = check_ascent(theta_new, theta_old, Y_vec, all_X_vecs, all_weights, alpha1)
    end

    iter_count = 0

    while !ascended
        iter_count += 1
        this_samp_size = length(all_X_vecs)
        this_lcl = lcl
        Q_increment = Q_MCEM_increment(theta_new, theta_old, Y_vec, all_X_vecs, all_weights)
        this_ESS = round(get_ESS(all_weights), sigdigits = 3)
        M_new = Int(ceil(M/k))                  # Augment by a fraction of the previous iteration's sample size (recommendation from paper)
        
        println("Inner Iteration: $iter_count, M = $(this_samp_size+M_new), ESS = $this_ESS, Q Increment = $(round(Q_increment, sigdigits=3)), lcl = $(round(this_lcl, sigdigits=3))")

        # ----------------------------- Augment MC sample ---------------------------- #
        # M_new = Int(ceil(this_samp_size/k))   # Augment by a fraction of the current sample size
        
        # Every 10 iterations, add a bunch more points
        # if iter_count % 10 == 0
        #     M_new *= 10
        # end

        new_X_vecs, new_raw_weights = get_importance_sample(theta_old, Y_vec, M_new, norm_weights=false)
        all_X_vecs = vcat(all_X_vecs, new_X_vecs)
        all_raw_weights = vcat(all_raw_weights, new_raw_weights)
        all_weights = normalize_weights(all_raw_weights)    # Automatically incorporates truncation
        
        # ----------------------------- Re-Estimate theta ---------------------------- #
        #! Might be better to start from theta_old
        theta_new = MCEM_update(theta_new, Y_vec, all_X_vecs, all_weights)

        

        # --------------------- Check for sufficient improvement --------------------- #
        if diagnostics
            ascended, lcl = check_ascent(theta_new, theta_old, Y_vec, all_X_vecs, all_weights, alpha1; return_lcl=true)
            push!(all_lcls, lcl)
        else
            ascended = check_ascent(theta_new, theta_old, Y_vec, all_X_vecs, all_weights, alpha1)
        end

        # ------------------------------- Check MC size ------------------------------ #
        #! Stop increasing MC size if > 1000
        if length(all_X_vecs) > 1000
            println("MC sample size exceeded 1000. Terminating.")
            ascended = true
        end
    end

    M_end = size(all_X_vecs, 1)



    diagnostics = false


    if diagnostics
        return M_end, all_lcls
    elseif !return_X && !return_MC_size
        return theta_new
    elseif return_X && !return_MC_size
        return theta_new, all_X_vecs, all_weights
    elseif !return_X && return_MC_size
        return theta_new, M_end
    else
        return theta_new, all_X_vecs, all_weights, M_end
    end
end


# # Check whether ascent-based MCEM algorithm augments the MC sample size.
# # theta_hat_EM = run_EM(theta1, Y)

# M = 10
# alpha = 0.4
# k = 3

# theta_hat, X, W = ascent_MCEM_update(theta, Y_vec, M, alpha, k; return_X = true)




# alpha1 = 0.2
# alpha2 = 0.2
# alpha3 = 0.2
# k = 3
# atol = 1e-3


"""
Perform a single iteration of ascent-based MCEM. Augments the MC size as necessary. Returns the initial MC size for next iteration, as well as a check for whether to terminate MCEM.
Note:   alpha1 - confidence level for checking whether to augment MC sample size
        alpha2 - confidence level for computing next step's initial MC sample size
        alpha3 - confidence level for checking whether to terminate MCEM

        k - when augmenting MC sample, add M/k new points
        atol - absolute tolerance for checking whether to terminate MCEM
"""
function full_ascent_MCEM_iteration(theta_old, Y_vec, M, alpha1, alpha2, alpha3, k, atol; return_X=false, diagnostics=false)

    theta_hat, all_X_vecs, all_weights = ascent_MCEM_update(theta_old, Y_vec, M, alpha1, k; return_X=true)
    M_end = size(all_X_vecs, 1)
    
    this_ESS = get_ESS(all_weights)

    M_next = get_next_MC_size(theta_hat, theta_old, Y_vec, all_X_vecs, all_weights, M_end, alpha1, alpha2)

    ready_to_terminate, ucl, this_ASE = check_for_termination(theta_hat, theta_old, Y_vec, all_X_vecs, all_weights, alpha3, atol, diagnostics=true)

    if !return_X && !diagnostics
        return theta_hat, M_next, ready_to_terminate
    elseif return_X && !diagnostics
        return theta_hat, M_next, ready_to_terminate, all_X_vecs, all_weights
    elseif !return_X && diagnostics
        return theta_hat, M_next, ready_to_terminate, ucl, this_ASE, this_ESS
    else
        return theta_hat, M_next, ready_to_terminate, all_X_vecs, all_weights, ucl, this_ASE, this_ESS
    end
end




"""
Perform a single iteration of ascent-based MCEM. Augments the MC size as necessary. Returns the initial MC size for next iteration, as well as a check for whether to terminate MCEM.
Note:   alpha1 - confidence level for checking whether to augment MC sample size
        alpha2 - confidence level for computing next step's initial MC sample size
        alpha3 - confidence level for checking whether to terminate MCEM

        k - when augmenting MC sample, add M/k new points
        atol - absolute tolerance for checking whether to terminate MCEM
"""
function full_ascent_MCEM_iteration(theta_old, Y_vec, M, MCEM_control; return_X=false, diagnostics=false)

    # Unpack ascent-based MCEM parameters
    alpha1 = MCEM_control.alpha1
    alpha2 = MCEM_control.alpha2
    alpha3 = MCEM_control.alpha3
    k = MCEM_control.k
    atol = MCEM_control.atol

    return full_ascent_MCEM_iteration(theta_old, Y_vec, M, alpha1, alpha2, alpha3, k, atol; return_X=return_X, diagnostics=diagnostics)

end

# theta_hat, M_next, terminate, X, W = full_ascent_MCEM_iteration(theta1, Y_vec, 10, alpha1, alpha2, alpha3, k, atol; return_X=true)

# theta_hat, M_next, terminate, X, W = full_ascent_MCEM_iteration(theta_hat, Y_vec, M, control; return_X=true)


# ---------------------------------------------------------------------------- #
#                       Full ascent-based MCEM algorithm                       #
# ---------------------------------------------------------------------------- #

# # 
# # --------------- Set control parameters for ascent-based MCEM --------------- #
# alpha1 = 0.3
# alpha2 = 0.3
# alpha3 = 0.3
# k = 3
# atol = 1e-3 # Absolute tolerance for convergence. 

# ascent_MCEM_control = Ascent_MCEM_Control(alpha1, alpha2, alpha3, k, atol)
# theta_init = theta_init1
# M_init = 10

"""
Run ascent-based MCEM algorithm.
Returns the final estimate of theta.
"""
function run_ascent_MCEM(theta_init, Y_vec, M_init, ascent_MCEM_control; diagnostics = false)

    #! START HERE

    start_time = time()

    # Initialize MCEM
    theta_hat = theta_init
    # theta_hat = theta_hat_SAEM    # Start from SAEM estimate
    # theta_hat = theta_hat_MCEM      # Start from terminal value
    # theta_hat = theta_0             # Start from true value
    M = M_init
    ready_to_terminate = false

    all_theta_hats = []

    iteration_count = 0

    # Run MCEM
    while !ready_to_terminate
        theta_hat, M, ready_to_terminate, ucl, this_ASE, this_ESS = full_ascent_MCEM_iteration(theta_hat, Y_vec, M, ascent_MCEM_control, diagnostics=true)
        push!(all_theta_hats, theta_hat)
        #! Remove this after figuring out a better importance sampling scheme
        # if M > 1000
        #     println("MC size exceeded 1000. Terminating MCEM.")
        #     ready_to_terminate = true
        # end
        iteration_count += 1
        # println(iteration_count)
        println("Outer Iteration: $iteration_count, MC size: $M, ESS: $(round(this_ESS, sigdigits=3)), UCL = $(round(ucl, sigdigits=2)), ASE = $(round(this_ASE, sigdigits=2))")

        # Stop if runtime exceeds 10 minutes
        elapsed_time = time() - start_time
        if elapsed_time > 600
            println("MCEM took more than 10 minutes. Terminating MCEM.")
            ready_to_terminate = true
        end
    end

    # hcat(theta_init, theta_hat, theta_0)


    
    # if size(theta_hat, 1) == 1
    #     theta_hat = theta_hat
    # end

    if diagnostics
        return all_theta_hats, M, iteration_count
    else
        return theta_hat
    end
end
