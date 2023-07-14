module LTHC_Analysis

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



include("Define_All_Distributions.jl")
include("Helper_Functions.jl")
include("Generate_Data.jl")
# include("Obs_Data_Likelihood_Functions.jl")
include("Complete_Data_Likelihood_Functions.jl")
# include("Conditional_Distribution_Functions.jl")
# include("EM_Functions.jl")
include("MCEM_Functions.jl")
include("Ascent_MCEM_Functions.jl")
include("SAEM_Functions.jl")

# ----------------------------- Parameter Bounds ----------------------------- #
par_lower = [0.0, 0.0, 0.0]
par_upper = [Inf, Inf, 1]

eta = 1.5    



end # module LTHC_Analysis



# test = ProposalDistribution(2, 0.2)

# pdf(test, 1)