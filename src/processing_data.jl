"""
    find_roots(interpolation, theta::AbstractVector{<:Real}, threshold::Real)

This finds the intercepts of the threshold and ``χ^2`` which corresponds to the confidence 
intervals. 

`interpolation` is a function which takes in a single input 
# Arguments
- `interpolation`: Function which takes in a single input ``x`` and output ``χ^2(x)`` where ``χ^2`` is profile likelihood. This should be an interpolation of the points found using `PL`.
- `theta::AbstractVector{<:Real}`: Values of ``θ`` searched using `PL`.
- `threshold::Real`: `threshold` found using `find_threshold`. If `pl_const` was used in `PL`, then also add `pl_const` to `threshold`.

# Returns
- `res`: Intercepts of the threshold and ``χ^2`` which corresponds to the confidence interval.
"""
function find_roots(interpolation, theta::AbstractVector{<:Real}, threshold::Real)
    res = find_zeros((x) -> interpolation(x) - threshold, (theta[1], theta[end]))
    println("The roots are $(res).")
    return res
end 
