
export increm_dens_Y, one_day_log_lik, post_trajectory_log_lik, complete_data_log_lik_one_Y, complete_data_log_lik
export complete_data_score
export complete_data_Hessian



# ---------------------------------------------------------------------------- #
#                                Log-Likelihood                                #
# ---------------------------------------------------------------------------- #


"""
Distribution of the next Y given the number of active cases.
"""
function increm_dens_Y(y, omega, phi)
    offspring_dist = OffspringDistribution(omega * phi)
    density = pdf(offspring_dist, y)

    return density
end

"""
Compute the log-likelihood for day t of the provided trajectory with the provided parameter values.
"""
function one_day_log_lik(t, Y, X, omega_traj, phi_0, gamma, lambda)
    duration_dist = DurationDistribution(lambda)

    # On day 1, only durations contribute to the likelihood.
    if t==1
        log_lik = sum(logpdf.(duration_dist, X[t]))
    else
        phi = get_phi(phi_0, gamma, t)

        cases_contrib = log(increm_dens_Y(Y[t], omega_traj[t], phi))
        durations_contrib = sum(logpdf.(duration_dist, X[t]))

        log_lik = cases_contrib + durations_contrib
    end

    return log_lik
end

"""
Incorporate zero observed cases into the likelihood on every day until no active infections remain.
Vector of active case counts provided.
"""
function post_trajectory_log_lik(Y, X, omega_traj, phi_0, gamma, lambda; log_scale=true)
    t = length(Y) + 1
    omega = 0.5 # Initialize to a value that won't end the while loop
    lik = 0

    while(omega > 0)
        omega = omega_traj[t]
        phi = get_phi(phi_0, gamma, t)
        offspring_dist = OffspringDistribution(omega * phi)
        
        this_lik = logpdf(offspring_dist, 0)
        lik += this_lik

        t += 1
    end

    if !log_scale
        lik = exp(lik)
    end

    return lik
end

"""
Incorporate zero observed cases into the likelihood on every day until no active infections remain.
Number of active cases computed as necessary
"""
function post_trajectory_log_lik(Y, X, phi_0, gamma, lambda; log_scale=true)
    t = length(Y) + 1
    omega = 0.5 # Initialize to a value that won't end the while loop
    lik = 0

    while(omega > 0)
        omega = get_omega(Y, X, t)
        phi = get_phi(phi_0, gamma, t)
        offspring_dist = OffspringDistribution(omega * phi)
        
        this_lik = logpdf(offspring_dist, 0)
        lik += this_lik

        t += 1
    end

    if !log_scale
        lik = exp(lik)
    end

    return lik
end

"""
Full data log likelihood for an observed trajectory. 
"""
function complete_data_log_lik_one_Y(theta, Y, X, omega_traj)

    # Unpack parameters
    phi_0, gamma, lambda = theta

    t_max = length(Y)

    # ------------- Compute log-likelihood contribution for each day ------------- #
    log_lik_obs = 0
    for t in 1:t_max
        log_lik_obs += one_day_log_lik(t, Y, X, omega_traj, phi_0, gamma, lambda)
    end


    log_lik_post_obs = post_trajectory_log_lik(Y, X, omega_traj, phi_0, gamma, lambda)

    log_lik = log_lik_obs + log_lik_post_obs
    return log_lik
end

"""
Full data log likelihood for an observed trajectory. Compute the trajectory of active case counts internally.
"""
function complete_data_log_lik_one_Y(theta, Y, X)
    omega_traj = get_omega_traj(Y, X)
    
    log_lik = complete_data_log_lik_one_Y(theta, Y, X, omega_traj)
    return log_lik
end

"""
Full data log likelihood for an observed trajectory.
X_traj_vec and omega_traj_vec are vectors of vectors
"""
function complete_data_log_lik(theta, Y_vec, X_traj_vec, omega_traj_vec)

    full_sample_log_lik = 0
    for i in eachindex(Y_vec)
        Y = Y_vec[i]
        X_traj = X_traj_vec[i]
        omega_traj = omega_traj_vec[i]

        full_sample_log_lik += complete_data_log_lik_one_Y(theta, Y, X_traj, omega_traj)
    end

    return full_sample_log_lik
end

"""
Full data log likelihood for an observed trajectory.
X_traj_vec and omega_traj_vec are vectors of vectors
"""
function complete_data_log_lik(theta, Y_vec, X_traj_vec)

    omega_traj_vec = omega_traj_one_sample(Y_vec, X_traj_vec)
    return complete_data_log_lik(theta, Y_vec, X_traj_vec, omega_traj_vec)

end


# ---------------------------------------------------------------------------- #
#                                     Score                                    #
# ---------------------------------------------------------------------------- #



function d1(Y, X, omega_traj, phi_0, gamma, lambda)
    if length(Y) ==1 
        term_1 = 0
    else
        term_1 = sum(Y[2:end]) / phi_0
    end

    term_2 = 0
    for t in 2:length(omega_traj)
        omega = omega_traj[t]
        term_2 -= omega * exp(-gamma * t)
    end

    return term_1 + term_2
end

function d2(Y, X, omega_traj, phi_0, gamma, lambda)
    if length(Y) ==1 
        term_1 = 0
    else
        T_max = length(Y)

        term_1 = @pipe map(*, 2:T_max, Y[2:end]) |> sum |> *(_, -1)
    end

    term_2 = 0
    for t in 2:length(omega_traj)
        omega = omega_traj[t]
        term_2 += t * omega * exp(-gamma * t)
    end
    term_2 *= phi_0

    return term_1 + term_2 
end

function d3(Y, X, omega_traj, phi_0, gamma, lambda)
    term_1 = sum(Y) / (lambda * (1 - lambda))

    term_2 = - sum(sum.(X)) / (1 - lambda)

    return term_1 + term_2
end


function score_vector_one_Y(Y, X, omega_traj, phi_0, gamma, lambda)
    return [d1(Y, X, omega_traj, phi_0, gamma, lambda), d2(Y, X, omega_traj, phi_0, gamma, lambda), d3(Y, X, omega_traj, phi_0, gamma, lambda)]
end

function score_vector_one_X(Y_vec, X_traj_vec, omega_traj_vec, phi_0, gamma, lambda)
    output = zeros(3)
    for j in eachindex(Y_vec)
        output += score_vector_one_Y(Y_vec[j], X_traj_vec[j], omega_traj_vec[j], phi_0, gamma, lambda)
    end
    return output
end


function complete_data_score(theta, Y_vec, X_traj_vec)
    omega_traj_vec = omega_traj_one_sample(Y_vec, X_traj_vec)
    return score_vector_one_X(Y_vec, X_traj_vec, omega_traj_vec, theta...)
end





# ---------------------------------------------------------------------------- #
#                                    Hessian                                   #
# ---------------------------------------------------------------------------- #

function d11(Y, phi_0)
    if length(Y) ==1 
        return 0
    else
        return -sum(Y[2:end]) / phi_0^2
    end
    
    
end

function d12(Y, omega_traj, gamma)
    output = 0
    for t in 2:length(omega_traj)
        omega = omega_traj[t]
        output += t * omega * exp(-gamma * t)
    end

    return output
end

function d13(Y, X, omega_traj, phi_0, gamma, lambda)
    return 0
end



function d21(Y, omega_traj, gamma)
    return d12(Y, omega_traj, gamma)
end

function d22(Y, X, omega_traj, phi_0, gamma)
    output = 0
    for t in 2:length(omega_traj)
        omega = omega_traj[t]
        output += t^2 * omega * exp(-gamma * t)
    end
    output *= -phi_0

    return output
end

function d23(Y, X, omega_traj, phi_0, gamma, lambda)
    return 0
end

function d31(Y, X, omega_traj, phi_0, gamma, lambda)
    return 0
end

function d32(Y, X, omega_traj, phi_0, gamma, lambda)
    return 0
end

function d33(Y, X, lambda)
    term_1 = sum(Y) * (2*lambda - 1) / (lambda * (1 - lambda))^2

    term_2 = - sum(sum.(X)) / (1 - lambda)^2

    return term_1 + term_2
end


function hessian_matrix_one_Y(Y, X, omega_traj, phi_0, gamma, lambda)
    return [d11(Y, phi_0) d12(Y, omega_traj, gamma) 0;
            d21(Y, omega_traj, gamma) d22(Y, X, omega_traj, phi_0, gamma) 0;
            0 0 d33(Y, X, lambda)]
end

function hessian_matrix_one_X(Y_vec, X_traj_vec, omega_traj_vec, phi_0, gamma, lambda)
    output = zeros(3,3)
    for j in eachindex(Y_vec)
        output += hessian_matrix_one_Y(Y_vec[j], X_traj_vec[j], omega_traj_vec[j], phi_0, gamma, lambda)
    end
    return output
end


function complete_data_Hessian(theta, Y_vec, X_traj_vec)
    omega_traj_vec = omega_traj_one_sample(Y_vec, X_traj_vec)
    return hessian_matrix_one_X(Y_vec, X_traj_vec, omega_traj_vec, theta...)
end


