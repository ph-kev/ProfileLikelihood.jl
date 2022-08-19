"""
    generate_data(index::Integer, seed::Integer, dist::Function, 
                  prob::SciMLBase.AbstractDEProblem, 
                  alg::SciMLBase.AbstractDEAlgorithm, 
                  times::AbstractVector{<:Real}; 
                  incidence_obs_status::Bool = false, 
                  kwargs...)
                  
This generates perfect and noisy data and incidence data of the state variables of the 
system of differential equations. 

Let ``\\mathcal{I}:[t_0,t_1,\\dots,t_n] \\rightarrow \\mathbb{R}`` be the number of incidences defined by 
``0`` if ``t=t_0`` and ``\\mathcal{C}(t_i) - \\mathcal{C}(t_{i-1})`` if ``t \\neq t_0``
where ``\\mathcal{C}`` is cumulative data. Note that `times` is ``[t_0,t_1,\\dots,t_n]``.

# Arguments
- `index::Integer`: Index of the component of the state variables for data collection.
- `seed::Integer`: Seed of RNG generator. Set for reproducible results. 
- `dist::Function`: Function to generate noisy data. The inputs are the data points sampled at points in `times`.
- `prob::SciMLBase.AbstractDEProblem`: DE Problem (see `DifferentialEquations.jl` for more information).
- `alg::SciMLBase.AbstractDEAlgorithm`: DE Solver Algorithm (see `DifferentialEquations.jl` for more information).
- `times::AbstractVector{<:Real}`: Times that the data points will be sampled at.

# Keywords
- `incidence_obs_status::Bool = false`: Determine whether the data is incidence data or not. If `true`, the `index`th state variable must be cumulative data.
- `kwargs...`: Keyword arguments to be passed into the DE solver. See `DifferentialEquations.jl`'s Common Solver Options.

# Returns
- `perfect_data`: Data without noise.
- `noisy_data`: Data with noise according to `dist`.
"""
function generate_data(index::Integer, seed::Integer, dist::Function, 
                       prob::SciMLBase.AbstractDEProblem, 
                       alg::SciMLBase.AbstractDEAlgorithm, 
                       times::AbstractVector{<:Real}; 
                       incidence_obs_status::Bool = false, 
                       kwargs...)
    if incidence_obs_status == false # check if we are generating incidence data or not 
        sol = solve(
            prob,
            alg,
            saveat=times,
            save_idxs=index;
            kwargs...
        )
        perfect_data = sol[1,:]
    else
        perfect_data = generate_incidence_data(index, prob, alg, times; kwargs...)
    end
    # Set seed for reproducibility 
    Random.seed!(seed)
    # Use probability distribution to generate data 
    noisy_data_generator = [dist(i) for i in perfect_data]
    # Generate noisy data 
    noisy_data = Vector{Float64}()
    for i in eachindex(noisy_data_generator) 
        append!(noisy_data, Distributions.rand(noisy_data_generator[i], 1))
    end
    return perfect_data, noisy_data
end

"""
    generate_incidence_data(index::Integer, 
                            prob::SciMLBase.AbstractDEProblem, 
                            alg::SciMLBase.AbstractDEAlgorithm, 
                            times::AbstractVector{<:Real}; 
                            kwargs...)

This generates incidence data without noise by solving the DEs and computing incidence data from cumulative data according to the `index`th state variable.

Let ``\\mathcal{I}:[t_0,t_1,\\dots,t_n] \\rightarrow \\mathbb{R}`` be the number of incidences defined by 
``0`` if ``t=t_0`` and ``\\mathcal{C}(t_i) - \\mathcal{C}(t_{i-1})`` if ``t \\neq t_0``
where ``\\mathcal{C}`` is cumulative data. Note that `times` is ``[t_0,t_1,\\dots,t_n]``.

# Arguments
- `index::Integer`: Index of the component of the state variables for data collection.
- `prob::SciMLBase.AbstractDEProblem`: DE Problem (see `DifferentialEquations.jl` for more information).
- `alg::SciMLBase.AbstractDEAlgorithm`: DE Solver Algorithm (see `DifferentialEquations.jl` for more information).
- `times::AbstractVector{<:Real}`: Times that the data points will be sampled at.

# Keywords
- `kwargs...`: Keyword arguments to be passed into the DE solver. See `DifferentialEquations.jl`'s Common Solver Options.

# Returns
- `incidence_data`: Incidence data of the `index`th state variable which represents cumulative data.
"""
function generate_incidence_data(index::Integer, 
                                 prob::SciMLBase.AbstractDEProblem, 
                                 alg::SciMLBase.AbstractDEAlgorithm, 
                                 times::AbstractVector{<:Real}; 
                                 kwargs...)
    sol = solve(
        prob,
        alg,
        saveat=times,
        save_idxs=index;
        kwargs...
    )
    cumulative_data = sol[1,:] # Generate cumulative data 
    perfect_data = Vector{Float64}()
    for i in eachindex(cumulative_data)
        if i == 1
            # First data point of incidence data is always 0
            append!(perfect_data, 0) 
        else 
            # Incidence data is Δ(C(t) - C(t-1))
            append!(perfect_data, sol(times[i]) - sol(times[i-1])) 
        end
    end
    return perfect_data
end 

"""
    generate_incidence_data(sol::AbstractVector{<:Real}) 

This generates incidence data from cumulative data.

Let ``\\mathcal{I}:[t_0,t_1,\\dots,t_n] \\rightarrow \\mathbb{R}`` be the number of incidences defined by 
``0`` if ``t=t_0`` and ``\\mathcal{C}(t_i) - \\mathcal{C}(t_{i-1})`` if ``t \\neq t_0``
where ``\\mathcal{C}`` is cumulative data.

# Arguments
- `sol::AbstractVector{<:Real}`: Vector of cumulative data.

# Returns
- `incidence_data`: Incidence data calculated from cumulative data.
"""
function generate_incidence_data(sol::AbstractVector{<:Real}) 
    incidence_data = Vector{Float64}()
    for i in eachindex(sol)
        if i == 1
            # First data point of incidence data is always 0
            append!(incidence_data, 0) 
        else 
            # Incidence data is Δ(C(t) - C(t-1))
            append!(incidence_data, sol[i] - sol[i-1]) 
        end
    end
    return incidence_data
end