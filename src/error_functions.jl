"""
    relative_error(data::AbstractVector{T}, sol::AbstractVector{T}, noise_level::T) where T<:Real

The objective function for when the noise in the data follow relative error. 
    
The objective function is

``\\hat{\\theta} = \\arg \\min_{\\theta} \\frac{1}{\\eta^2} \\sum_{i=1}^{n} \\left( \\frac{y_i - g(t_i,\\theta)}{g(t_i,\\theta)} \\right)^2 + 2 \\sum_{i=1}^n \\log g(t_i,\\theta)``

where ``\\eta`` is the noise level, ``g(t_i,\\theta)`` is the ``i``th point of the predicted solution, and ``y_i`` is the ``i``th data point. 
"""
function relative_error(data::AbstractVector{T}, sol::AbstractVector{T}, noise_level::T) where T<:Real
    return (1 / noise_level^2) * sum(((abs.(sol) - data) ./ abs.(sol)) .^ 2) + 2.0 * sum(log.(abs.(sol)))
end

"""
    poisson_error(data::AbstractVector{T}, sol::AbstractVector{T}) where T<:Real

The objective function for when the data follow a Poisson distribution. 
    
The objective function is

``\\hat{\\theta} = \\arg \\min_{\\theta} -\\sum_{i=1}^n \\log (g(t_i,u,\\theta)^{y_i}) + \\sum_{i=1}^n g(t_i,u,\\theta)``

where ``g(t_i,\\theta)`` is the ``i``th point of the predicted solution and ``y_i`` is the ``i``th data point.
"""
function poisson_error(data::AbstractVector{T}, sol::AbstractVector{T}) where T<:Real
    return 2*(sum(abs.(sol)) - sum(data .* log.(abs.(sol))))
end

"""
    const_variance_error(data::AbstractVector{T}, sol::AbstractVector{T}, sigma::T) where T<:Real

The objective function for when the data follow a Normal distribution with mean 0 and known variance ``\\sigma^2``. 
    
The objective function is 

``\\hat{\\theta} = \\arg \\min_{\\theta} \\sum_{i=1}^n (y_i - g(t_i,u,\\theta))^2 / \\sigma^2``

where ``g(t_i,\\theta)`` is the ``i``th point of the predicted solution and ``y_i`` is the ``i``th data point.
"""
function const_variance_error(data::AbstractVector{T}, sol::AbstractVector{T}, sigma::T) where T<:Real
    return (1 / sigma^2) * sum((sol - data) .^ 2)
end

function likelihood_const(obj::String; noise_level=0.01, times=Vector{Real}(), data=Vector{Real}(), sigma = 1.0) 
    if obj == "relative_error"
        return length(times) * log(noise_level^2) + length(times) * log(2 * pi)
    elseif obj == "poisson_error"
        return 2*sum(log.(factorial.(big.(data))))
    elseif obj == "const_variance_error"
        return length(times) * log(2 * pi) + length(times) * log(sigma^2)
    else
        return 0.0
    end
end