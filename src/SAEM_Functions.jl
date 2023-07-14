

export importance_sample_SAEM

export get_alpha, get_all_zetas
export SAEM_objective, SAEM_gradient, get_SAEM_estimate
export run_SAEM

export get_alpha_adaptive, get_all_zetas_adaptive
export SAEM_objective_adaptive, SAEM_gradient_adaptive, get_SAEM_estimate_adaptive
export run_SAEM_adaptive



"""
Return the Xs and normalized, untruncated, non-log-scale weights for an MC sample of size M.
"""
function importance_sample_SAEM(theta, Y_vec, M)
    all_X_vecs = Vector{Any}(undef, M)
    all_log_weights_raw = Vector(undef, M)

    for i in 1:M
        all_X_vecs[i], all_log_weights_raw[i] = one_MC_draw(Y_vec, theta...)
    end

    # -------- Normalize weights using highly stable log-sum-exp function -------- #
    # --------- Also exponentiates away the log in the way you need it to -------- #
    all_weights = normalize_weights(all_log_weights_raw, truncate=false)
    return all_X_vecs, all_weights
end


# ---------------------------------------------------------------------------- #
#                             SAEM with fixed alpha                            #
# ---------------------------------------------------------------------------- #

"""
Weight on the new sample at each iteration.
"""
SA_step_size(k, alpha) = k^(-alpha)

"""
Compute the weight on sample j at iteration k.
"""
function get_zeta(j, k, alpha)
    gamma = SA_step_size(k, alpha)
    if j==1
        return (1-gamma)^(k-j)
    else
        return gamma * (1-gamma)^(k-j)
    end
end


"""
Compute the SAEM objective function at the given theta. Iteration number and MC size are inferred from the supplied MC sample.
"""
function SAEM_objective(theta, Y_vec, all_Xs_list, all_normed_weights_list, alpha)
    k = length(all_Xs_list)
    M = length(all_Xs_list[1])

    all_Q_MCEMs = [Q_MCEM(theta, Y_vec, all_Xs_list[i], all_normed_weights_list[i]) for i in 1:k]
    all_zetas = [get_zeta(i, k, alpha) for i in 1:k]

    return dot(all_Q_MCEMs, all_zetas)
end

"""
Gradient of the SAEM objective function. Iteration number and MC size are inferred from the supplied MC sample.
"""
function SAEM_gradient(theta, Y_vec, all_Xs_list, all_normed_weights_list, alpha)
    k = length(all_Xs_list)
    M = length(all_Xs_list[1])

    all_Q_MCEM_gradients = [score_MCEM(theta, Y_vec, all_Xs_list[i], all_normed_weights_list[i]) for i in 1:k]
    all_zetas = [get_zeta(i, k, alpha) for i in 1:k]

    return sum(all_Q_MCEM_gradients .* all_zetas)
end



"""
Maximize the SAEM objective function.
"""
function get_SAEM_estimate(Y_vec, all_Xs_list, all_normed_weights_list, theta_init, alpha)

    function objective(theta)
        return - SAEM_objective(theta, Y_vec, all_Xs_list, all_normed_weights_list, alpha)
    end

    function grad!(G, theta)
        G[:] = - SAEM_gradient(theta, Y_vec, all_Xs_list, all_normed_weights_list, alpha)
    end

    # Newton alone, custom stopping criterion
    BFGS_iterate = optimize(objective, grad!, par_lower, par_upper, theta_init, Fminbox(BFGS()),
    Optim.Options(show_trace=false, x_tol = 0, f_tol = 0, g_tol = 1e-6, time_limit = 30))
    optim_iterate = BFGS_iterate

    theta_hat = Optim.minimizer(optim_iterate)
    return theta_hat
end








function run_SAEM(theta_init, Y_vec, B, M, alpha)

    all_Xs_list = Vector{Any}(undef, B)
    all_normed_weights_list = Vector{Any}(undef, B)
    all_theta_hats = Vector{Any}(undef, B)

    theta_hat = theta_init

    @showprogress for i in 1:B
        all_Xs, all_weights = importance_sample_SAEM(theta_hat, Y_vec, M)
        all_Xs_list[i] = all_Xs
        all_normed_weights_list[i] = all_weights
        # push!(all_Xs_list, all_Xs)
        # push!(all_normed_weights_list, all_weights)

        theta_hat = get_SAEM_estimate(Y_vec, all_Xs_list[1:i], all_normed_weights_list[1:i], theta_init, alpha)
        all_theta_hats[i] = theta_hat
        # push!(all_theta_hats, theta_hat)
    end

    return all_theta_hats
end






# B = 50
# M = 10
# alpha = 0.6

# all_theta_hats = run_SAEM(theta_init, Y_vec, B, M, alpha)
# # run_SAEM_JuMP(theta_init, Y_vec, B, M, alpha)




# all_phi_0_hats = getindex.(all_theta_hats, 1)
# all_gamma_hats = getindex.(all_theta_hats, 2)
# all_lambda_hats = getindex.(all_theta_hats, 3)


# phi_0_plot = plot(all_phi_0_hats, label="phi_0", title="M=$M, alpha=$alpha");
# gamma_plot = plot(all_gamma_hats, label="gamma");
# lambda_plot = plot(all_lambda_hats, label="lambda");

# plot(phi_0_plot, gamma_plot, lambda_plot, layout=(3,1), size=(1200, 1000))






# ---------------------------------------------------------------------------- #
#                        Repeat above process with JuMP                        #
# ---------------------------------------------------------------------------- #

#! I haven't been able to get the JuMP optimizer to work. It might be worth exploring more, but all I'm going to get is a speedup and, for now, I'm better off not spending a bunch of time on it.


# using JuMP, Ipopt

# function get_SAEM_estimate_JuMP(Y_vec, all_Xs_list, all_normed_weights_list, alpha; theta_init = [1, 1, 0.5])
#     model = Model(Ipopt.Optimizer)
#     # set_silent(model)
#     @variable(model, 0 <= phi_0_jump, start = theta_init[1])
#     @variable(model, 0 <= gamma_jump, start = theta_init[2])
#     @variable(model, 0 <= lambda_jump <= 1, start = theta_init[3])
#     this_obj_fun(phi_0_jump, gamma_jump, lambda_jump) = SAEM_objective([phi_0_jump, gamma_jump, lambda_jump], Y_vec, all_Xs_list, all_normed_weights_list, alpha)
#     function this_grad!(G, phi_0_jump, gamma_jump, lambda_jump)
#         G[:] = SAEM_gradient([phi_0_jump, gamma_jump, lambda_jump], Y_vec, all_Xs_list, all_normed_weights_list, alpha)
#     end
#     register(model, :this_obj_fun, 3, this_obj_fun, this_grad!)
#     @NLobjective(model, Max, this_obj_fun(phi_0_jump, gamma_jump, lambda_jump))
#     optimize!(model)

#     phi_0_hat = value(phi_0_jump)
#     gamma_hat = value(gamma_jump)
#     lambda_hat = value(lambda_jump)
#     return phi_0_hat, gamma_hat, lambda_hat
# end



# function run_SAEM_JuMP(theta_init, Y_vec, B, M, alpha)

#     all_Xs_list = Vector{Any}(undef, B)
#     all_normed_weights_list = Vector{Any}(undef, B)
#     all_theta_hats = Vector{Any}(undef, B)

#     theta_hat = theta_init

#     @showprogress for i in 1:B
#         all_Xs, all_weights = importance_sample_SAEM(theta_hat, Y_vec, M)
#         all_Xs_list[i] = all_Xs
#         all_normed_weights_list[i] = all_weights
#         # push!(all_Xs_list, all_Xs)
#         # push!(all_normed_weights_list, all_weights)

#         theta_hat = get_SAEM_estimate_JuMP(Y_vec, all_Xs_list[1:i], all_normed_weights_list[1:i], alpha)
#         all_theta_hats[i] = theta_hat
#         push!(all_theta_hats, theta_hat)
        
#     end

#     return all_theta_hats
# end



# ---------------------------------------------------------------------------- #
#            Repeat the above process with step sizes varying with M           #
# ---------------------------------------------------------------------------- #


"""
Power on 1/k at current iteration
"""
function get_alpha_adaptive(k, eta)
    return 1 - eta^-k
end

"""
Weight on new MCEM objective function at iteration k. Satisfies the following:
    - gamma_adaptive(1) = 1
        - Convenient for writing the code
    - gamma_adaptive(2) = 1/2
    - gamma_adaptive(k) -> 0 as k -> Inf 
        - I.e. Starts with large step sizes and gets smaller as k increases
    - sum gamma_adaptive(k) = Inf
    - sum gamma_adaptive(k)^2 < Inf
        - Valid SA scheme
"""
function get_gamma_adaptive(k, eta)
    this_alpha = get_alpha_adaptive(k, eta)
    return k^(-this_alpha)
end


"""
Compute the weight for all MCEM objectives at iteration k. 
"""
function get_all_zetas_adaptive(k, eta)
    all_gammas = get_gamma_adaptive.(1:k, eta)

    all_zetas = all_gammas
    for i in 1:(k-1)
        for j in (i+1):k
            all_zetas[i] *= (1 - all_gammas[j])
        end
    end

    return all_zetas
end  

# all_alphas = get_alpha_adaptive.(1:20)
# all_gammas = get_gamma_adaptive.(1:20)
# all_zetas = get_all_zetas(20)


# ### There is a large jump from zeta_1 to zeta_2. This is actually fine.
# ### To see why, note that zeta_i differs from zeta_(i+1) in that the former contains gamma_i * (1 - gamma_(i+1)), while the latter contains gamma_(i+1). The rest of the zetas is identical (a product of 1 - gamma_j for j=i+2,...,k). For i=1, gamma_i * (1 - gamma_(i+1)) contains only a single term (gamma_1 = 1), while for every other i, this product contains two terms. The effect of having two terms is enough to offset the effect of gamma_(i+1) being closer to 0 than to 1.
# ### See black notebook for mathematical details.
# all_gammas[2] * (1 - all_gammas[3])
# all_gammas[3]

# test1 = zeros(19)
# test2 = zeros(19)
# for i in 1:19
#     test1[i] = all_gammas[i] * (1 - all_gammas[i+1])
#     test2[i] = all_gammas[i+1]
# end

"""
Compute the SAEM objective function at the given theta. Iteration number and MC size are inferred from the supplied MC sample.
"""
function SAEM_objective_adaptive(theta, Y_vec, all_Xs_list, all_normed_weights_list, eta)
    k = length(all_Xs_list)

    all_Q_MCEMs = [Q_MCEM(theta, Y_vec, all_Xs_list[i], all_normed_weights_list[i]) for i in 1:k]
    all_zetas = get_all_zetas_adaptive(k, eta)

    return dot(all_Q_MCEMs, all_zetas)
end

"""
Gradient of the SAEM objective function. Iteration number and MC size are inferred from the supplied MC sample.
"""
function SAEM_gradient_adaptive(theta, Y_vec, all_Xs_list, all_normed_weights_list, eta)
    k = length(all_Xs_list)

    all_Q_MCEM_gradients = [score_MCEM(theta, Y_vec, all_Xs_list[i], all_normed_weights_list[i]) for i in 1:k]
    all_zetas = get_all_zetas_adaptive(k, eta)

    return sum(all_Q_MCEM_gradients .* all_zetas)
end

"""
Hessian of the SAEM objective function. Iteration number and MC size are inferred from the supplied MC sample.
"""
function SAEM_hessian_adaptive(theta, Y_vec, all_Xs_list, all_normed_weights_list, eta)
    k = length(all_Xs_list)

    all_Q_MCEM_hessians = [Hessian_MCEM(theta, Y_vec, all_Xs_list[i], all_normed_weights_list[i]) for i in 1:k]
    all_zetas = get_all_zetas_adaptive(k, eta)

    return sum(all_Q_MCEM_hessians .* all_zetas)
end


"""
Maximize the SAEM objective function.
"""
function get_SAEM_estimate_adaptive(Y_vec, all_Xs_list, all_normed_weights_list, eta)

    function objective(theta)
        return - SAEM_objective_adaptive(theta, Y_vec, all_Xs_list, all_normed_weights_list, eta)
    end

    function grad!(G, theta)
        G[:] = - SAEM_gradient_adaptive(theta, Y_vec, all_Xs_list, all_normed_weights_list, eta)
    end

    function hess!(H, theta)
        H[:] = - SAEM_hessian_adaptive(theta, Y_vec, all_Xs_list, all_normed_weights_list, eta)
    end

    # # BFGS alone, custom stopping criterion
    # BFGS_iterate = optimize(objective, grad!, par_lower, par_upper, theta_init, Fminbox(BFGS()),
    # Optim.Options(show_trace=false, x_tol = 0, f_tol = 0, g_tol = 1e-6, time_limit = 30))
    # optim_iterate = BFGS_iterate

    # Newton alone, custom stopping criterion
    Newton_iterate = optimize(objective, grad!, hess!, par_lower, par_upper, theta_init, IPNewton(),
    Optim.Options(show_trace=false, x_tol = 0, f_tol = 0, g_tol = 1e-6, time_limit = 5))
    optim_iterate = Newton_iterate

    theta_hat = Optim.minimizer(optim_iterate)
    return theta_hat
end








function run_SAEM_adaptive(theta_init, Y_vec, B, M, eta)

    all_Xs_list = Vector{Any}(undef, B)
    all_normed_weights_list = Vector{Any}(undef, B)
    all_theta_hats = Vector{Any}(undef, B)

    theta_hat = theta_init

    @showprogress for i in 1:B
        all_Xs, all_weights = importance_sample_SAEM(theta_hat, Y_vec, M)
        all_Xs_list[i] = all_Xs
        all_normed_weights_list[i] = all_weights
        # push!(all_Xs_list, all_Xs)
        # push!(all_normed_weights_list, all_weights)

        theta_hat = get_SAEM_estimate_adaptive(Y_vec, all_Xs_list[1:i], all_normed_weights_list[1:i], eta)
        all_theta_hats[i] = theta_hat
        # push!(all_theta_hats, theta_hat)
    end

    return all_theta_hats
end







# # --------------------------- Algorithm parameters --------------------------- #
# W = 1  # Number of datasets to generate and analyze
# N = 50  # Number of trajectories to include in each dataset
# M_SAEM = 20    # Size of Monte Carlo sample
# B = 500   # Number of MCEM or SAEM iterations
# R = 10    # Number of times to repeat MCEM on the same dataset
# L = 3     # Number of different points to use for the initial guesses
# t_max_raw = 25  # Number of time steps to simulate. Trailing zeros will be dropped.


# # A number greater than 1 which governs how quickly the step sizes decay.
# # Note: Julia requires that this be a float, not an integer. E.g. use 2.0, not 2
# eta = 1.5

# Random.seed!(10)
# all_Ys, all_Xs = make_complete_data(t_max_raw, theta_0, Y_1, N, W)
# Y_vec = all_Ys[1]

# # theta_hat_MCEM = theta_hat
# all_theta_hats_adaptive = run_SAEM_adaptive(theta_init, Y_vec, B, M_SAEM, eta);
# # all_theta_hats_adaptive = run_SAEM_adaptive(theta_0, Y_vec, B, M_SAEM, eta);
# # all_theta_hats_adaptive = run_SAEM_adaptive(theta_hat_MCEM, Y_vec, B, M_SAEM, eta);   # Start from MCEM estimate
# # run_SAEM_JuMP(theta_init, Y_vec, B, M, alpha)


# theta_hat_adaptive = all_theta_hats_adaptive[end]

# all_phi_0_hats_adaptive = getindex.(all_theta_hats_adaptive, 1);
# all_gamma_hats_adaptive = getindex.(all_theta_hats_adaptive, 2);
# all_lambda_hats_adaptive = getindex.(all_theta_hats_adaptive, 3);


# phi_0_plot_adaptive = plot(all_phi_0_hats_adaptive, label="phi_0", title="SAEM: M=$M_SAEM, Adaptive Step Size - eta=$eta");
# # hline!(phi_0_plot_adaptive, [theta_hat[1]], label = "MCEM Estimate");
# gamma_plot_adaptive = plot(all_gamma_hats_adaptive, label="gamma");
# # hline!(gamma_plot_adaptive, [theta_hat[2]], label = "MCEM Estimate");
# lambda_plot_adaptive = plot(all_lambda_hats_adaptive, label="lambda");
# # hline!(lambda_plot_adaptive, [theta_hat[3]], label = "MCEM Estimate");

# plot(phi_0_plot_adaptive, gamma_plot_adaptive, lambda_plot_adaptive, layout=(3,1), size=(1200, 1000))



# # all_estimates = [all_theta_hats_adaptive[end]]
# push!(all_estimates, all_theta_hats_adaptive[end])


