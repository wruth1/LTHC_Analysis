using Distributions
import Base.rand
import Distributions.pdf
import Distributions.logpdf
import Distributions.mean
import Distributions.var

export DurationDistribution
export pdf, rand, logpdf, mean, var

struct DurationDistribution <: DiscreteUnivariateDistribution
    lambda::Float64
    DurationDistribution(lambda) = new(Float64(lambda))
end

pdf(d::DurationDistribution, x::Int) = pdf(Geometric(d.lambda), x - 1)
rand(d::DurationDistribution) = rand(Geometric(d.lambda)) + 1
rand(d::DurationDistribution, n::Int) = rand(Geometric(d.lambda), n) .+ 1
# sampler(d::DurationDistribution) = sampler(Geometric(d.lambda)) + 1
logpdf(d::DurationDistribution, x::Real) = logpdf(Geometric(d.lambda), x - 1)
# cdf(d::DurationDistribution, x::Real) = cdf(Normal(d.mu, 2*d.sigma), x)
# quantile(d::DurationDistribution, q::Real) = quantile(Normal(d.mu, 2*d.sigma), q)
# minimum(d::DurationDistribution) = -Inf
# maximum(d::DurationDistribution) = Inf
# insupport(d::DurationDistribution, x::Real) = insupport(Normal(d.mu, 2*d.sigma), x)
mean(d::DurationDistribution) = mean(Geometric(d.lambda)) + 1
var(d::DurationDistribution) = var(Geometric(d.lambda))