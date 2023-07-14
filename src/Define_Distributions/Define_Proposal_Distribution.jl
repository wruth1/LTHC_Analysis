using Distributions
import Base.rand
import Distributions.pdf
import Distributions.logpdf
import Distributions.mean
import Distributions.var

export ProposalDistribution
export pdf, rand, logpdf, mean, var

struct ProposalDistribution <: DiscreteUnivariateDistribution
    r::Float64
    p::Float64
    ProposalDistribution(r, p) = new(Float64(r), Float64(p))
end

pdf(d::ProposalDistribution, x::Int) = pdf(NegativeBinomial(d.r, d.p), x - 1)
rand(d::ProposalDistribution) = rand(NegativeBinomial(d.r, d.p)) + 1
rand(d::ProposalDistribution, n::Int) = rand(NegativeBinomial(d.r, d.p), n) .+ 1
# sampler(d::ProposalDistribution) = sampler(Geometric(d.lambda)) + 1
logpdf(d::ProposalDistribution, x::Real) = logpdf(NegativeBinomial(d.r, d.p), x - 1)
# cdf(d::ProposalDistribution, x::Real) = cdf(Normal(d.mu, 2*d.sigma), x)
# quantile(d::ProposalDistribution, q::Real) = quantile(Normal(d.mu, 2*d.sigma), q)
# minimum(d::ProposalDistribution) = -Inf
# maximum(d::ProposalDistribution) = Inf
# insupport(d::ProposalDistribution, x::Real) = insupport(Normal(d.mu, 2*d.sigma), x)
mean(d::ProposalDistribution) = mean(NegativeBinomial(d.r, d.p)) + 1
var(d::ProposalDistribution) = var(NegativeBinomial(d.r, d.p))
