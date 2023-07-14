
export r_traj_pair, r_traj_pair_clean
export make_complete_data, make_data

"""
Generate a (Y, X) pair of length `t_max` using the provided parameters and initial outbreak size `Y_1`.
"""
function r_traj_pair(t_max, phi_0, gamma, lambda, Y_1)
    # Initialize duration distribution
    duration_dist = DurationDistribution(lambda)
    
    # Generate durations for initial cases
    X_1 = rand(duration_dist, Y_1)

    # Initialize Y and X, and add day 1
    Y = Vector(undef, t_max)
    Y[1] = Y_1
    X = Vector(undef, t_max)
    X[1] = X_1

    for t in 2:t_max
        phi = get_phi(phi_0, gamma, t)
        omega = get_omega(Y, X, t)
        offspring_dist = OffspringDistribution(omega * phi)

        this_Y = rand(offspring_dist)
        Y[t] = this_Y

        this_X = rand(duration_dist, this_Y)
        X[t] = this_X
    end

    output = Dict("Y" => Y, "X" => X)
    return output
end


"""
Generate a (Y, X) pair of length at most `t_max_raw` using the provided parameters and initial outbreak size `Y_1`. Any trailing zeros are dropped.
"""
function r_traj_pair_clean(t_max_raw, phi_0, gamma, lambda, Y_1)
    # Generate a trajectory pair and extract Y, X
    traj_pair_raw = r_traj_pair(t_max_raw, phi_0, gamma, lambda, Y_1)
    Y_raw = traj_pair_raw["Y"]
    X_raw = traj_pair_raw["X"]

    # Remove trailing zeros
    t_max = maximum(findall(x -> x>0, Y_raw))
    if t_max == t_max_raw
        error("t_max_raw is too small.")
    end
    Y = Y_raw[1:t_max]
    X = X_raw[1:t_max]

    output = Dict("X" => X, "Y" => Y)
    return output
end

"""
Generate W datasets, each with N trajectories. Return both Y and X.
"""
function make_complete_data(t_max_raw, theta_0, Y_1, N, W)
    # Unpack parameters
    phi_0_0, gamma_0, lambda_0 = theta_0

    data_raw = [[r_traj_pair_clean(t_max_raw, phi_0_0, gamma_0, lambda_0, Y_1) for _ in 1:N] for _ in 1:W]
    all_Ys = [[data_raw[i][j]["Y"] for j in 1:N] for i in 1:W]
    all_Xs = [[data_raw[i][j]["X"] for j in 1:N] for i in 1:W]
    return all_Ys, all_Xs
end

"""
Generate W datasets, each with N trajectories, using the provided parameter settings.
"""
function make_data(t_max_raw, theta_0, Y_1, N, W)
    all_Ys, _ = make_complete_data(t_max_raw, theta_0, Y_1, N, W)
    return all_Ys
end

# """
# Generate W datasets, each with N trajectories, using the provided parameter settings.
# Executes in parallel.
# """
# function make_data_parallel(t_max_raw, phi_0_0, gamma_0, lambda_0, Y_1, N, W)
#     data = Vector(undef, W)
#     # data = fill(fill(undef, N), W)

#     #* prog_bar = Progress(W*N, desc="Generating data:")

#     Threads.@threads for i in 1:W
#         data[i] = Vector(undef, N)
#         for j in 1:N
#             data[i][j] = r_traj_pair_clean(t_max_raw, phi_0_0, gamma_0, lambda_0, Y_1)["Y"]
#             #* next!(prog_bar)
#         end
#     end
#     return data
# end