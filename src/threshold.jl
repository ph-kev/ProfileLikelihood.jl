"""
    find_threshold(confidence::Real, nums_params::Integer, loss::Real)

This finds the threshold used for calculating likelihood-based confidence intervals. 

The threshold Δ_α is the α quantile of the χ^2 distribution with df=1 for pointwise 
confidence intervals and df=# of parameters for simulataneous confidence intervals.

# Arguments 
- `confidence::Real`: Level of confidence used for confidence interval. 
- `nums_params::Integer`: Number of parameters. ``df`` = 1 for pointwise confidence intervals and ``df`` = number of unknown parameters for simulataneous confidence intervals.
- `loss:Real`: Minimum loss according to negative log-likelihood function after optimization.

# Return
- `threshold`: Threshold used for likelihood-based confidence interval. 
"""
function find_threshold(confidence::Real, nums_params::Integer, loss::Real)
    threshold = loss + cquantile(Chisq(nums_params), 1 - confidence)
    return threshold
end

