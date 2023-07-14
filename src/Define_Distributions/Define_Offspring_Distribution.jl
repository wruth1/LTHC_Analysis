using Distributions
import Base.rand
import Distributions.pdf
import Distributions.logpdf

export OffspringDistribution
export pdf, rand, logpdf

struct OffspringDistribution <: DiscreteUnivariateDistribution
    phi::Float64
    OffspringDistribution(phi) = new(Float64(phi))
end

pdf(d::OffspringDistribution, x::Int) = pdf(Poisson(d.phi), x)
rand(d::OffspringDistribution) = rand(Poisson(d.phi))
rand(d::OffspringDistribution, n::Int) = rand(Poisson(d.phi), n)
# sampler(d::OffspringDistribution) = sampler(Poisson(d.lambda)) + 1
logpdf(d::OffspringDistribution, x::Int) = logpdf(Poisson(d.phi), x)
# cdf(d::OffspringDistribution, x::Real) = cdf(Normal(d.mu, 2*d.sigma), x)
# quantile(d::OffspringDistribution, q::Real) = quantile(Normal(d.mu, 2*d.sigma), q)
# minimum(d::OffspringDistribution) = -Inf
# maximum(d::OffspringDistribution) = Inf
# insupport(d::OffspringDistribution, x::Real) = insupport(Normal(d.mu, 2*d.sigma), x)
# mean(d::OffspringDistribution) = mean(Normal(d.mu, 2*d.sigma))
# var(d::OffspringDistribution) = var(Normal(d.mu, 2*d.sigma))