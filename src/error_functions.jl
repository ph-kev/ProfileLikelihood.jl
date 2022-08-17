"""
    relative_error(data::AbstractVector{<:Real}, sol::AbstractVector{<:Real}, noise_level::Real)

The objective function for when the noise in the data follow relative error. 
    
The objective function is

``\\hat{θ} = \\arg \\min_{θ} \\frac{1}{η^2} \\sum_{i=1}^{n} \\left( \\frac{y_i - g(t_i,θ)}{g(t_i,θ)} \\right)^2 + 2 \\sum_{i=1}^n \\log g(t_i,θ)``

where η is the noise level, ``g(t_i,θ)`` is the ``i``th point of the predicted solution, and ``y_i`` is the ``i``th data point. 
"""
function relative_error(data::AbstractVector{<:Real}, sol::AbstractVector{<:Real}, noise_level::Real) 
    return (1 / noise_level^2) * sum(((sol - data) ./ sol) .^ 2) + 2.0 * sum(log.(sol))
end

"""
    poisson_error(data::AbstractVector{<:Real}, sol::AbstractVector{<:Real}) 

The objective function for when the data follow a Poisson distribution. 
    
The objective function is

``\\hat{θ} = \\arg \\min_{θ} -\\sum_{i=1}^n \\log (g(t_i,u,θ)^{y_i}) + \\sum_{i=1}^n g(t_i,u,θ)``

where ``g(t_i,θ)`` is the ``i``th point of the predicted solution and ``y_i`` is the ``i``th data point.
"""
function poisson_error(data::AbstractVector{<:Real}, sol::AbstractVector{<:Real}) 
    return 2*(sum(abs.(sol)) - sum(abs.(data) .* log.(abs.(sol))))
end

"""
    const_variance_error(data::AbstractVector{<:Real}, sol::AbstractVector{<:Real}, sigma::Real)

The objective function for when the data follow a normal distribution with mean 0 and known variance ``\\sigma^2``. 
    
The objective function is 

``\\hat{θ} = \\arg \\min_{θ} \\sum_{i=1}^n (y_i - g(t_i,u,θ))^2 / \\sigma^2``

where ``g(t_i,θ)`` is the ``i``th point of the predicted solution and ``y_i`` is the ``i``th data point.
"""
function const_variance_error(data::AbstractVector{<:Real}, sol::AbstractVector{<:Real}, sigma::Real)
    return (1 / sigma^2) * sum((sol - data) .^ 2)
end

"""
    likelihood_const(obj::String; noise_level::Real=0.01, times::AbstractVector{<:Real}=Vector{Real}(), data::AbstractVector{<:Real}=Vector{Real}(), sigma::Real = 1.0)

This gives the constant in the likelihood function when the error is assumed to follow relative error or constant variance error or when the data is assumed to follow a Poisson distribution.

The constant for the likelihood function is often dropped when finding the minimizer of the likelihood function. For relative error, the constant is 

TODO
"""
function likelihood_const(obj::String; noise_level::Real=0.0, times::AbstractVector{<:Real}=Vector{Real}(), data::AbstractVector{<:Real}=Vector{Real}(), sigma::Real = -1.0) 
    if obj == "relative_error"
        if length(times) == 0
            println("The keyword argument times is missing.")
        end
        if length(noise_level) == 0
            println("The keyword argument noise_level is missing.")
        end
        return length(times) * log(noise_level^2) + length(times) * log(2 * pi)
    elseif obj == "poisson_error"
        if length(data) == 0
            println("The keyword argument data is missing.")
        end
        return 2*sum(log.(factorial.(big.(data))))
    elseif obj == "const_variance_error"
        if length(times) == 0
            println("The keyword argument times is missing.")
        end
        if length(sigma) == 0
            println("The keyword argument sigma is missing.")
        end
        return length(times) * log(2 * pi) + length(times) * log(sigma^2)
    else
        return 0.0
    end
end