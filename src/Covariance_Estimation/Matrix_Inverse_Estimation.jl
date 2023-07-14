
using Random
using Distributions


n = 1000

alpha = 1

epsilon = 1e-3

Random.seed!(1)
all_Zs = epsilon * randn(n)


# ----------------------- Directly evaluate (alpha + Z)^-1 ----------------------- #
all_Ws = alpha .+ all_Zs
all_W_invs = 1 ./ all_Ws


# ---------------- Evaluate (alpha + Z)^-1 via Woodbury's formula ---------------- #
all_As = (1 / alpha) .+ (1 ./ all_Zs)
all_Bs = 1 ./ (all_As)
all_Cs = all_Bs ./ (alpha^2)
all_Ds = 1/alpha .- all_Cs


all_Ds ≈ all_W_invs



all_As
# all_Bs