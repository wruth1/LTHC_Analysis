module Marginal_Optimization

using Revise    # Allows re-running "using SLR_Vector_Parameter" after changes to this file
using Optim     # For optimization
using Distributions # For multivariate normal distribution
using LinearAlgebra # For inverse and diag functions
using ProgressMeter
using LogExpFunctions   # For logsumexp function
using JLD2
using ReverseDiff
using Random
using Statistics
using Pipe
using RData
using DataFrames
using Dates
using BenchmarkTools

export par_lower, par_upper

include("Define_All_Distributions.jl")
include("Helper_Functions.jl")
include("Generate_Data.jl")
# include("Obs_Data_Likelihood_Functions.jl")
include("Complete_Data_Likelihood_Functions.jl")
# include("Conditional_Distribution_Functions.jl")
# include("EM_Functions.jl")
include("MCEM_Functions.jl")
include("Ascent_MCEM_Functions.jl")

# ----------------------------- Parameter Bounds ----------------------------- #
par_lower = [0.0, 0.0, 0.0]
par_upper = [Inf, Inf, 1]

eta = 1.5    

# todo - Fix this
#! Code from previous project passes arguments in a different way.
#! Parameters are passed pointwise instead of as a vector, there is no theta_fixed, and the order is different.
#* Done - Removed theta_fixed.


end # module LTHC_Analysis



# test = ProposalDistribution(2, 0.2)

# pdf(test, 1)