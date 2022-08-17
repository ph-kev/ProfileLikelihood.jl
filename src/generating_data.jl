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