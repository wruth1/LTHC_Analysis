export Q_EM_term, Q_EM, Q_EM_increment
export EM_update_beta, EM_update_sigma, EM_update, run_EM
export conditional_info1, conditional_info2, conditional_info3, complete_data_conditional_information
export cond_exp_beta_grad, cond_exp_sigma_grad
export cond_exp_beta_grad2, cond_exp_prod, cond_exp_sigma_grad2
export cond_exp_sq_score1, cond_exp_sq_score2, cond_exp_sq_score3
export expect_sq_score
export EM_obs_data_information_formula, EM_COV_formula, EM_SE_formula

# ---------------------------------------------------------------------------- #
#                          EM objective and maximizer                          #
# ---------------------------------------------------------------------------- #


function Q_EM_term(theta, y, theta_old)

    # Unpack theta
    beta, sigma = theta

    # Compute Q
    A = -log(sigma)
    B = -y^2 / (2 * sigma^2)
    C = beta * y * mu_X_given_Y(theta_old, y) / sigma^2
    D = - beta^2 * mu2_X_given_Y(theta_old, y) / (2 * sigma^2)

    output = A + B + C + D
    return output
end


function Q_EM(theta, Y, theta_old)
    output = 0
    for i in eachindex(Y)
        output += Q_EM_term(theta, Y[i], theta_old)
    end
    return output
end


function Q_EM_increment(theta_new, Y, theta_old)
    A = Q_EM(theta_new, Y, theta_old)
    B = Q_EM(theta_old, Y, theta_old)

    return A - B
end


# ---------------------------------------------------------------------------- #
#                              EM Update Functions                             #
# ---------------------------------------------------------------------------- #

"""
Compute the EM update for beta.
"""
function EM_update_beta(theta_old, Y)
    all_mu_hats = [mu_X_given_Y(theta_old, y) for y in Y]
    all_mu2_hats = [mu2_X_given_Y(theta_old, y) for y in Y]

    A = dot(Y, all_mu_hats)
    B = sum(all_mu2_hats)

    return A / B
end


"""
Compute the EM update for sigma. Depends on the updated value of beta.
"""
function EM_update_sigma(beta_new, theta_old, Y)
    all_mu_hats = [mu_X_given_Y(theta_old, y) for y in Y]
    all_mu2_hats = [mu2_X_given_Y(theta_old, y) for y in Y]

    n = length(Y)

    A = dot(Y, Y)
    B = -2 * beta_new * dot(Y, all_mu_hats)
    C =  sum(all_mu2_hats) * beta_new^2 

    return sqrt((A + B + C)/n)
end


"""
Perform one iteration of EM.
"""
function EM_update(theta_old, Y)
    beta_hat = EM_update_beta(theta_old, Y)
    sigma_hat = EM_update_sigma(beta_hat, theta_old, Y)

    return [beta_hat, sigma_hat]
end


"""
Iterate EM from the specified intial value of theta until convergence within the specified relative tolerance.
"""
function run_EM(theta_init, Y; rtol = 1e-6, return_iteration_count = false, return_trajectory = false)
    theta_old = theta_init
    theta_new = EM_update(theta_old, Y)

    if return_trajectory
        trajectory = [theta_old, theta_new]
    end

    count = 1

    while (norm(theta_new - theta_old)/norm(theta_old)) > rtol
        count += 1
        theta_old = theta_new
        theta_new = EM_update(theta_old, Y)

        if return_trajectory
            push!(trajectory, theta_new)
        end
    end


    if return_trajectory && return_iteration_count
        return theta_new, count, trajectory
    elseif return_trajectory && !return_iteration_count
        return theta_new, trajectory
    elseif !return_trajectory && return_iteration_count
        return theta_new, count
    else
        return theta_new
    end
end



# ---------------------------------------------------------------------------- #
#                            Standard error formula                            #
# ---------------------------------------------------------------------------- #


# ---------------------- Conditional Information Matrix ---------------------- #

"""
Negative conditional expectation of second-order beta derivative of complete data log-lik, given Y.
"""
function conditional_info1(theta, Y)
    # Unpack theta
    beta, sigma = theta

    all_mu2s = [mu2_X_given_Y(theta, y) for y in Y]

    output = sum(all_mu2s) / sigma^2
    return output
end

"""
Negative conditional expectation of derivative of complete data log-lik wrt beta and sigma, given Y.
"""
function conditional_info2(theta, Y)
    # Unpack theta
    beta, sigma = theta

    all_mus = [mu_X_given_Y(theta, y) for y in Y]
    all_mu2s = [mu2_X_given_Y(theta, y) for y in Y]

    A = 2 / sigma^3
    B = dot(Y, all_mus)
    C = - beta * sum(all_mu2s)

    return A * (B + C)
end

"""
Negative conditional expectation of second-order sigma derivative of complete data log-lik, given Y.
"""
function conditional_info3(theta, Y)
    # Unpack theta
    beta, sigma = theta

    n = length(Y)
    all_mus = [mu_X_given_Y(theta, y) for y in Y]
    all_mu2s = [mu2_X_given_Y(theta, y) for y in Y]

    A = 3 / sigma^4
    B = dot(Y, Y)
    C = -2 * beta * dot(Y, all_mus)
    D = beta^2 * sum(all_mu2s)
    E = -n / sigma^2
    return A * (B + C + D) + E
end


"""
Conditional expectation of the complete data observed information, given Y.
"""
function complete_data_conditional_information(theta, Y)
    A = conditional_info1(theta, Y)
    B = conditional_info2(theta, Y)
    C = conditional_info3(theta, Y)

    output = [A B; B C]
    return output
end





# ------ Conditional expectation of S_complete * S_complete^T, given Y. ------ #


### First, we need to define functions to compute the necessary conditional expectations.


"""
Conditional expectation of the beta derivative of the complete data log-likelihood, given Y.
"""
function cond_exp_beta_grad(theta, y)
    # Unpack parameters
    beta, sigma = theta

    mu = mu_X_given_Y(theta, y)
    mu2 = mu2_X_given_Y(theta, y)

    A = mu * y
    B = - beta * mu2

    output = (A + B) / sigma^2
    return output
end

"""
Conditional expectation of the sigma derivative of the complete data log-likelihood, given Y.
"""
function cond_exp_sigma_grad(theta, y)
    # Unpack parameters
    beta, sigma = theta
    

    mu = mu_X_given_Y(theta, y)
    mu2 = mu2_X_given_Y(theta, y)

    A = -1/sigma
    B = y^2
    C = -2 * beta * y * mu
    D = beta^2 * mu2

    output = A + (B + C + D) / sigma^3
    return output
end



"""
Conditional expectation of the squared beta derivative of the complete data log-likelihood, given Y.
"""
function cond_exp_beta_grad2(theta, y)
    # Unpack parameters
    beta, sigma = theta
    

    mu = mu_X_given_Y(theta, y)
    mu2 = mu2_X_given_Y(theta, y)
    mu3 = mu3_X_given_Y(theta, y)
    mu4 = mu4_X_given_Y(theta, y)

    A = beta^2 * mu4
    B = -2 * beta * mu3 * y
    C = mu2 * y^2

    return (A + B + C) / sigma^4
end


"""
Conditional expectation of the product between the beta and sigma derivatives of the complete data log-likelihood, given Y.
"""
function cond_exp_prod(theta, y)
    # Unpack parameters
    beta, sigma = theta
    

    mu = mu_X_given_Y(theta, y)
    mu2 = mu2_X_given_Y(theta, y)
    mu3 = mu3_X_given_Y(theta, y)
    mu4 = mu4_X_given_Y(theta, y)

    A = beta * sigma^2 * mu2
    B = - y * mu * sigma^2
    C = -beta^3 * mu4
    D = 3 * beta^2 * mu3 * y
    E = -3 * beta * mu2 * y^2
    F = mu * y^3

    return (A + B + C + D + E + F) / sigma^5
end


"""
Conditional expectation of the squared sigma derivative of the complete data log-likelihood, given Y.
"""
function cond_exp_sigma_grad2(theta, y)
    # Unpack parameters
    beta, sigma = theta
    

    mu = mu_X_given_Y(theta, y)
    mu2 = mu2_X_given_Y(theta, y)
    mu3 = mu3_X_given_Y(theta, y)
    mu4 = mu4_X_given_Y(theta, y)

    # Group terms based on power of sigma in the numerator
    A = sigma^4

    B = -2 * beta^2 * sigma^2 * mu2
    C = 4 * sigma^2 * beta * y * mu
    D = -2 * sigma^2 * y^2
    
    E = beta^4 * mu4
    F = -4 * beta^3 * mu3 * y
    G = 6 * beta^2 * mu2 * y^2
    H = -4 * beta * mu * y^3
    I = y^4

    return (A + B + C + D + E + F + G + H + I) / sigma^6
end



### We are now equipped to compute the conditional expectation of S_complete * S_complete^T, given Y. We will do so one entry at a time.

"""
Conditional expectation of the (1,1) entry of the squared score matrix, given Y.
"""
function cond_exp_sq_score1(theta, Y)
    all_expects = [cond_exp_beta_grad(theta, y) for y in Y]
    all_expect2s = [cond_exp_beta_grad2(theta, y) for y in Y]

    sum_cross_prods = 0
    for i in eachindex(Y)
        for j in eachindex(Y)
            if i != j
                sum_cross_prods += all_expects[i] * all_expects[j]
            end
        end
    end

    sum_expect2s = sum(all_expect2s)

    return sum_expect2s + sum_cross_prods
end

"""
Conditional expectation of the (1,2) and (2,1) entries of the squared score matrix, given Y.
"""
function cond_exp_sq_score2(theta, Y)
    all_expects_beta = [cond_exp_beta_grad(theta, y) for y in Y]
    all_expects_sigma = [cond_exp_sigma_grad(theta, y) for y in Y]
    all_expect_prod = [cond_exp_prod(theta, y) for y in Y]

    sum_cross_prods = 0
    for i in eachindex(Y)
        for j in eachindex(Y)
            if i != j
                sum_cross_prods += all_expects_beta[i] * all_expects_sigma[j]
            end
        end
    end

    sum_expect_prod = sum(all_expect_prod)

    return sum_expect_prod + sum_cross_prods
end

"""
Conditional expectation of the (2,2) entry of the squared score matrix, given Y.
"""
function cond_exp_sq_score3(theta, Y)
    all_expects = [cond_exp_sigma_grad(theta, y) for y in Y]
    all_expect2s = [cond_exp_sigma_grad2(theta, y) for y in Y]

    sum_cross_prods = 0
    for i in eachindex(Y)
        for j in eachindex(Y)
            if i != j
                sum_cross_prods += all_expects[i] * all_expects[j]
            end
        end
    end

    sum_expect2s = sum(all_expect2s)

    return sum_expect2s + sum_cross_prods
end


"""
Conditional expectation of the squared score (outer product with itself), given Y. 
"""
function expect_sq_score(theta, Y)
    A = cond_exp_sq_score1(theta, Y)
    B = cond_exp_sq_score2(theta, Y)
    C = cond_exp_sq_score3(theta, Y)

    return [A B; B C]
end


# -------------------------- Compute standard error -------------------------- #

"""
Evaluate the formula for the information matrix of the observed data log-likelihood using only quantities available to the EM algorithm.
Note: This formula is only valid at a fixed point of EM.
"""
function EM_obs_data_information_formula(theta, Y)
    A = complete_data_conditional_information(theta, Y)
    B = expect_sq_score(theta, Y)

    return A - B
end



"""
Evaluate the formula for the estimated covariance matrix of the EM estimator. That is, the inverse negative Hessian of the observed data log-likelihood.
Note: This formula is only valid at a fixed point of EM.
"""
function EM_COV_formula(theta, Y)
    return inv(EM_obs_data_information_formula(theta, Y))
end

"""
Evaluate the formula for the estimated standard error of the EM estimator.
Note: This formula is only valid at a fixed point of EM.
"""
function EM_SE_formula(theta, Y)
    return sqrt.(diag(EM_COV_formula(theta, Y)))
end



