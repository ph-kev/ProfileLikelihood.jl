function find_threshold(confidence::Real, nums_params::Integer, loss::Real)
    threshold = loss + cquantile(Chisq(nums_params), 1 - confidence)
    return threshold
end
