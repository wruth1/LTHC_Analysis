
export obs_data_log_lik_term, obs_data_log_lik
export obs_data_score1, obs_data_score2, obs_data_score
export obs_data_Hessian1, obs_data_Hessian2, obs_data_Hessian3, obs_data_Hessian
export obs_data_MLE
export obs_data_obs_info, obs_data_MLE_Cov, obs_data_MLE_SE

# using Optim

# ---------------------------------------------------------------------------- #
#                                log-likelihood                                #
# ---------------------------------------------------------------------------- #
function obs_data_log_lik_term(theta, y)
    beta, sigma = theta
    

    eta = get_eta(theta)

    A = - log(2 * pi * eta) / 2
    B = - (y - beta * mu)^2 / (2 * eta)
    
    return A + B
end


function obs_data_log_lik(theta, Y)
    output = 0
    for y in Y
        output += obs_data_log_lik_term(theta, y)
    end
    return output
end



# ---------------------------------------------------------------------------- #
#                                     score                                    #
# ---------------------------------------------------------------------------- #

"""
Beta derivative of observed data log-likelihood.
"""
function obs_data_score1(theta, Y)
    beta, sigma = theta
    

    n = length(Y)
    sY = sum(Y)
    sY2 = sum(Y.^2)

    eta = get_eta(theta)
    eta2 = eta^2

    A = - n * beta^3 * tau^4
    B = -beta^2 * mu * tau^2 * sY

    C1 = tau^2 * (sY2 - n*sigma^2)
    C2 = -n * mu^2 * sigma^2
    C = beta * (C1 + C2)

    D = mu * sigma^2 * sY

    output = (A + B + C + D) / eta2
    return output
end

"""
Sigma derivative of observed data log-likelihood.
"""
function obs_data_score2(theta, Y)
    beta, sigma = theta
    

    n = length(Y)
    sY = sum(Y)
    sY2 = sum(Y.^2)

    eta = get_eta(theta)
    eta2 = eta^2

    A = n * beta^2 * (mu^2 - tau^2)
    B = -2 * beta * mu * sY
    C = sY2
    D = - n * sigma^2

    output = sigma * (A + B + C + D) / eta2
    return output
end

"""
Gradient of observed data log-likelihood.
"""
function obs_data_score(theta, Y)
    output = [obs_data_score1(theta, Y), obs_data_score2(theta, Y)]
    return output
end


# ---------------------------------------------------------------------------- #
#                                    Hessian                                   #
# ---------------------------------------------------------------------------- #

"""
Second-order beta derivative of observed data log-likelihood.
"""
function obs_data_Hessian1(theta, Y)
    beta, sigma = theta
    

    n = length(Y)
    sY = sum(Y)
    sY2 = sum(Y.^2)

    eta = get_eta(theta)
    eta3 = eta^3

    A = n * beta^4 * tau^6

    B1 = 2 * tau^4 * beta^3 * mu * sY
    B2 = -3 * tau^4 * beta^2 * sY2
    B = B1 + B2

    C1 = 3 * n * beta^2 * mu^2
    C2 = -6 * beta * mu * sY
    C3 = sY2 - n* sigma^2
    C = sigma^2 * tau^2 * (C1 + C2 + C3)

    D = -n * mu^2 * sigma^4


    output = (A + B + C + D) / eta3
    return output
 end


"""
Derivative of observed data log-likelihood wrt beta and sigma.
"""
 function obs_data_Hessian2(theta, Y)
    beta, sigma = theta
    

    n = length(Y)
    sY = sum(Y)
    sY2 = sum(Y.^2)

    eta = get_eta(theta)
    eta3 = eta^3

    A = n * beta^3 * mu^2 * tau^2
    B = -n  * beta^3 * tau^4
    C = -3 * beta^2 * mu * tau^2 * sY
    D = -n * beta * mu^2 * sigma^2
    E = -n * beta * sigma^2 * tau^2
    F = 2 * beta * tau^2 * sY2
    G = mu * sigma^2 * sY

    output = -2 * sigma * (A + B + C + D + E + F + G) / eta3
    return output
 end



 """
Second-order sigma derivative of observed data log-likelihood.
"""
 function obs_data_Hessian3(theta, Y)
    beta, sigma = theta
    

    n = length(Y)
    sY = sum(Y)
    sY2 = sum(Y.^2)

    eta = get_eta(theta)
    eta3 = eta^3

    A = n * beta^4 * mu^2 * tau^2
    B = -n * beta^4 * tau^4
    C = -2 * beta^3 * mu * tau^2 * sY
    D = -3 * n * beta^2 * mu^2 * sigma^2
    E = beta^2 * tau^2 * sY2
    F = 6 * beta * mu * sigma^2 * sY
    G = n * sigma^4
    H = -3 * sigma^2 * sY2

    output = (A + B + C + D + E + F + G + H) / eta3
    return output
 end


"""
Hessian of observed data log-likelihood.
"""
function obs_data_Hessian(theta, Y)
    output = [obs_data_Hessian1(theta, Y) obs_data_Hessian2(theta, Y); obs_data_Hessian2(theta, Y) obs_data_Hessian3(theta, Y)]
    return output
end



# ---------------------------------------------------------------------------- #
#                                      MLE                                     #
# ---------------------------------------------------------------------------- #

"""
Find the value of theta (i.e. beta and sigma) which maximizes the observed data log-likelihood.
Performs numerical optimization using the BFGS algorithm. Optionally, supply an initial value of theta.
"""
function obs_data_MLE(Y; theta_init = [1.0, 1.0])
    
    # --------------------- Define functions for optimization -------------------- #
    function this_log_lik(theta)
        return -obs_data_log_lik(theta, Y)
    end
    
    function this_score!(G, theta)
        G[1] = -obs_data_score1(theta, Y)
        G[2] = -obs_data_score2(theta, Y)
    end

    lower_bds = [-Inf, 0.0]
    upper_bds = [Inf, Inf]

    # theta_init = [2.0, 2.0]

    # ----------------- Perform optimization and extract results ----------------- #
    optim_results = optimize(this_log_lik, this_score!, lower_bds, upper_bds, theta_init, Fminbox(Optim.BFGS()), Optim.Options(show_trace=false, iterations=1000))
    # optim_results = optimize(this_log_lik, this_score!, theta_init, GradientDescent(), Optim.Options(show_trace=false, iterations=1000))
    theta_hat = optim_results.minimizer

    return theta_hat
end




# --------------------------- Standard Error of MLE -------------------------- #

"""
Evaluate the observed information (negative Hessian) of the observed data likelihood at the MLE.
MLE provided as an argument.
"""
function obs_data_obs_info(theta_hat, Y)
    Hessian = obs_data_Hessian(theta_hat, Y)
    return -Hessian
end


"""
Estimate the covariance matrix of the MLE using the obs data information matrix.
"""
function obs_data_MLE_Cov(theta_hat, Y)
    obs_info = obs_data_obs_info(theta_hat, Y)
    return inv(obs_info)
end

"""
Estimate the standard error of the MLE using the obs data information matrix.
"""
function obs_data_MLE_SE(theta_hat, Y)
    MLE_Cov = obs_data_MLE_Cov(theta_hat, Y)
    return sqrt.(diag(MLE_Cov))
end

