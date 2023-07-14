
export get_ASE, check_ascent, check_for_termination, get_next_MC_size
export Ascent_MCEM_Control
export ascent_MCEM_update, full_ascent_MCEM_iteration
export run_ascent_MCEM


# ---------------------------------------------------------------------------- #
#                 Asymptotic SE of increment in MCEM objective                 #
# ---------------------------------------------------------------------------- #



"""
Estimate the asymptotic standard error of the increment in MCEM objective function.
"""
function get_ASE(theta_val_new, theta_val_old, Y_vec, all_X_vecs, all_weights, par_ind, theta_0)
    M = length(all_X_vecs)

    # theta_new = theta_0
    # theta_old = theta_init1

    all_lik_increments = [complete_data_log_lik_increment(theta_val_new, theta_val_old, Y_vec, X_vec, par_ind, theta_0) for X_vec in all_X_vecs]

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
function check_ascent(theta_val_new, theta_val_old, Y_vec, all_X_vecs, all_weights, alpha, par_ind, theta_0; return_lcl = false)
    Q_increment = Q_MCEM_increment(theta_val_new, theta_val_old, Y_vec, all_X_vecs, all_weights, par_ind, theta_0)
    if Q_increment < 0
        error("Theta hat decreases MCEM objective.")
    end
    ASE = get_ASE(theta_val_new, theta_val_old, Y_vec, all_X_vecs, all_weights, par_ind, theta_0)
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
function check_for_termination(theta_val_new, theta_val_old, Y_vec, all_X_vecs, all_weights, alpha, atol, par_ind, theta_0; diagnostics = false)
    Q_increment = Q_MCEM_increment(theta_val_new, theta_val_old, Y_vec, all_X_vecs, all_weights, par_ind, theta_0)
    ASE = get_ASE(theta_val_new, theta_val_old, Y_vec, all_X_vecs, all_weights, par_ind, theta_0)
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
    # println("Proposed M: ", M_proposed)

    M_new = Int(max(M_proposed, M_old))

    return M_new
end


"""
Compute sample size for next iteration of MCEM.
Intermediate quantities are computed inside the function.
"""
function get_next_MC_size(theta_new, theta_old, Y_vec, all_X_vecs, all_weights, M_old, alpha1, alpha2, par_ind, theta_0)
    Q_increment = Q_MCEM_increment(theta_new, theta_old, Y_vec, all_X_vecs, all_weights, par_ind, theta_0)
    ASE = get_ASE(theta_new, theta_old, Y_vec, all_X_vecs, all_weights, par_ind, theta_0)

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
    k::Float64        # when augmenting MC sample, add M/k new points
    atol::Float64   # absolute tolerance for checking whether to terminate MCEM
end



"""
Perform a single iteration of ascent-based MCEM. Uses a level-alpha confidence bound to check for ascent. If not, augments the MC sample with M/k new points and tries again.
Options for diagnostics are included to check whether the MC sample was augmented.
"""
function ascent_MCEM_update(theta_val_old, Y_vec, M, alpha, k, par_ind, theta_0; return_MC_size = false, return_X=false, diagnostics = false)
    
    theta_old = get_theta(theta_val_old, par_ind, theta_0)
    
    
    diagnostics = true
    
    
    
    
    
    all_X_vecs, all_raw_weights = get_importance_sample(theta_old, Y_vec, M; norm_weights=false)
    all_weights = normalize_weights(all_raw_weights)
    # get_ESS(all_weights)

    theta_val_new = MCEM_update(theta_val_old, Y_vec, all_X_vecs, all_weights, par_ind, theta_0)
    theta_new = get_theta(theta_val_new, par_ind, theta_0)

    all_lcls = []

    if diagnostics
        ascended, lcl = check_ascent(theta_val_new, theta_val_old, Y_vec, all_X_vecs, all_weights, alpha, par_ind, theta_0; return_lcl=true)
        push!(all_lcls, lcl)
    else
        ascended = check_ascent(theta_val_new, theta_val_old, Y_vec, all_X_vecs, all_weights, alpha, par_ind, theta_0)
    end

    iter_count = 0

    while !ascended
        iter_count += 1
        this_samp_size = length(all_X_vecs)
        this_lcl = lcl
        Q_increment = Q_MCEM_increment(theta_val_new, theta_val_old, Y_vec, all_X_vecs, all_weights, par_ind, theta_0)
        this_ESS = round(get_ESS(all_weights), sigdigits = 3)
        println("Inner Iteration: $iter_count, M = $this_samp_size, ESS = $this_ESS, Q Increment = $(round(Q_increment, sigdigits=3)), lcl = $(round(this_lcl, sigdigits=3)), Estimate = $theta_val_new")

        # ----------------------------- Augment MC sample ---------------------------- #
        M_new = Int(ceil(M/k))
        new_X_vecs, new_raw_weights = get_importance_sample(theta_new, Y_vec, M_new, norm_weights=false)
        all_X_vecs = vcat(all_X_vecs, new_X_vecs)
        all_raw_weights = vcat(all_raw_weights, new_raw_weights)
        all_weights = normalize_weights(all_raw_weights)    # Automatically incorporates truncation
        
        # ----------------------------- Re-Estimate theta ---------------------------- #
        # Might be better to start from theta_old
        theta_val_new = MCEM_update(theta_val_new, Y_vec, all_X_vecs, all_weights, par_ind, theta_0)

        # --------------------- Check for sufficient improvement --------------------- #
        if diagnostics
            ascended, lcl = check_ascent(theta_val_new, theta_val_old, Y_vec, all_X_vecs, all_weights, alpha, par_ind, theta_0; return_lcl=true)
            push!(all_lcls, lcl)
        else
            ascended = check_ascent(theta_val_new, theta_val_old, Y_vec, all_X_vecs, all_weights, alpha, par_ind, theta_0)
        end
    end

    M_end = size(all_X_vecs, 1)



    diagnostics = false


    if diagnostics
        return M_end, all_lcls
    elseif !return_X && !return_MC_size
        return theta_val_new
    elseif return_X && !return_MC_size
        return theta_val_new, all_X_vecs, all_weights
    elseif !return_X && return_MC_size
        return theta_val_new, M_end
    else
        return theta_val_new, all_X_vecs, all_weights, M_end
    end
end


# #* Check whether ascent-based MCEM algorithm augments the MC sample size.
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
# MCEM_control = Ascent_MCEM_Control(alpha1, alpha2, alpha3, k, atol)


"""
Perform a single iteration of ascent-based MCEM. Augments the MC size as necessary. Returns the initial MC size for next iteration, as well as a check for whether to terminate MCEM.
Note:   alpha1 - confidence level for checking whether to augment MC sample size
        alpha2 - confidence level for computing next step's initial MC sample size
        alpha3 - confidence level for checking whether to terminate MCEM

        k - when augmenting MC sample, add M/k new points
        atol - absolute tolerance for checking whether to terminate MCEM
"""
function full_ascent_MCEM_iteration(theta_val_old, Y_vec, M, alpha1, alpha2, alpha3, k, atol, par_ind, theta_0; return_X=false, diagnostics=false)

    theta_val_hat, all_X_vecs, all_weights = ascent_MCEM_update(theta_val_old, Y_vec, M, alpha1, k, par_ind, theta_0; return_X=true)
    M_end = size(all_X_vecs, 1)

    M_next = get_next_MC_size(theta_val_hat, theta_val_old, Y_vec, all_X_vecs, all_weights, M_end, alpha1, alpha2, par_ind, theta_0)

    ready_to_terminate, ucl, this_ASE = check_for_termination(theta_val_hat, theta_val_old, Y_vec, all_X_vecs, all_weights, alpha3, atol, par_ind, theta_0, diagnostics=true)

    if !return_X && !diagnostics
        return theta_val_hat, M_next, ready_to_terminate
    elseif return_X && !diagnostics
        return theta_val_hat, M_next, ready_to_terminate, all_X_vecs, all_weights
    elseif !return_X && diagnostics
        return theta_val_hat, M_next, ready_to_terminate, ucl, this_ASE
    else
        return theta_val_hat, M_next, ready_to_terminate, all_X_vecs, all_weights, ucl, this_ASE
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
function full_ascent_MCEM_iteration(theta_val_old, Y_vec, M, MCEM_control, par_ind, theta_0; return_X=false, diagnostics=false)

    # Unpack ascent-based MCEM parameters
    alpha1 = MCEM_control.alpha1
    alpha2 = MCEM_control.alpha2
    alpha3 = MCEM_control.alpha3
    k = MCEM_control.k
    atol = MCEM_control.atol

    return full_ascent_MCEM_iteration(theta_val_old, Y_vec, M, alpha1, alpha2, alpha3, k, atol, par_ind, theta_0; return_X=return_X, diagnostics=diagnostics)

end


# theta_hat, M_next, terminate, X, W = full_ascent_MCEM_iteration(theta1, Y_vec, 10, alpha1, alpha2, alpha3, k, atol; return_X=true)

# theta_hat, M_next, terminate, X, W = full_ascent_MCEM_iteration(theta_hat, Y_vec, M, control; return_X=true)


# ---------------------------------------------------------------------------- #
#                       Full ascent-based MCEM algorithm                       #
# ---------------------------------------------------------------------------- #

# 
# --------------- Set control parameters for ascent-based MCEM --------------- #
alpha1 = 0.2
alpha2 = 0.1
alpha3 = 0.1
k = 1
atol = 1e-3 # Absolute tolerance for convergence. 

ascent_MCEM_control = Ascent_MCEM_Control(alpha1, alpha2, alpha3, k, atol)
theta_init = theta_init1
M_init = 100

"""
Run ascent-based MCEM algorithm.
Returns the final estimate of theta.
"""
function run_ascent_MCEM(theta_val_init, Y_vec, M_init, ascent_MCEM_control, par_ind, theta_0; diagnostics = false)

    #! START HERE


    # Initialize MCEM
    theta_val_hat = theta_val_init
    M = M_init
    ready_to_terminate = false

    iteration_count = 0

    # Run MCEM
    while !ready_to_terminate
        theta_val_hat, M, ready_to_terminate, ucl, this_ASE = full_ascent_MCEM_iteration(theta_val_hat, Y_vec, M, ascent_MCEM_control, par_ind, theta_0, diagnostics=true)
        iteration_count += 1
        # println(iteration_count)
        println("Outer Iteration: $iteration_count, MC size: $M, UCL = $(round(ucl, sigdigits=2)), ASE = $(round(this_ASE, sigdigits=2)), Estimate = $theta_val_hat")
    end

    convergence_info = DataFrame(init = theta_val_init, hat = theta_hat, truth = theta_0)

    # if size(theta_hat, 1) == 1
    #     theta_hat = theta_hat
    # end

    if diagnostics
        return theta_hat, M
    else
        return theta_hat
    end
end



# control.atol = 1e-3
# theta_hat = run_ascent_MCEM(theta_init1, Y_vec, 10, control)

# """
# Run Ascent-Based MCEM algorithm for B replications.
# """
# function run_many_ascent_MCEMs(B, theta_init, theta_true, M_init, ascent_MCEM_control)
        
#     # Unpack theta_true
#     beta_0, sigma_0 = theta_true

    

#     # Container to store estimates of theta
#     all_theta_hat_MCEMs = Vector{Vector{Float64}}(undef, B)

#     # all_SE_hat_MCEMs = Vector{Vector{Float64}}(undef, B_MCEM)


#     @showprogress for b in eachindex(all_theta_hat_MCEMs)
#         # Generate data
#         Random.seed!(b^2)
#         this_X = rand(Normal(mu_0, tau_0), n)
#         this_epsilon = rand(Normal(0, sigma_0), n)
#         this_Y = beta_0 * this_X + this_epsilon

#         # Estimate theta
#         Random.seed!(b^2)
#         this_theta_hat = run_ascent_MCEM(theta_init, this_Y, M_init, ascent_MCEM_control)
#         all_theta_hat_MCEMs[b] = this_theta_hat

#         # # Estimate SE
#         # #! Fix-SE
#         # Random.seed!(b^2)
#         # this_SE_hat = MCEM_SE_formula(this_theta_hat, this_Y, this_theta_hat, M_SE)
#         # all_SE_hat_MCEMs[b] = this_SE_hat

#     end

#     return all_theta_hat_MCEMs
# end