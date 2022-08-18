"""
    find_roots(interpolation, theta::AbstractVector{<:Real}, threshold::Real)
                  
# Arguments
- `interpolation`: the array to search
- `theta::AbstractVector{<:Real}`: 
- `threshold::Real`: 

# Returns
- `res`: 
"""
function find_roots(interpolation, theta::AbstractVector{<:Real}, threshold::Real)
    res = find_zeros((x) -> interpolation(x) - threshold, (theta[1], theta[end]))
    println("The roots are $(res).")
    return res
end 