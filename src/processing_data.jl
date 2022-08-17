function find_roots(interpolation, theta::AbstractVector{<:Real}, threshold::Real)
    res = find_zeros((x) -> interpolation(x) - threshold, (theta[1], theta[end]))
    println("The roots are $(res).")
    return res
end 