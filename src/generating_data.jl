function generate_data(index::Integer, seed::Integer, dist::Function, prob, alg, times::AbstractVector{T}; incidence_obs_status = false, kwargs...) where T<:Real
    if incidence_obs_status == false
        # Solve ODE
        sol = solve(
            prob,
            alg,
            saveat=times,
            save_idxs=index;
            kwargs...
        )
        perfect_data = abs.(sol[1,:])
    else
        perfect_data = generate_incidence_data(index, prob, alg, times; kwargs...)
    end
    # Set seed for reproducibility 
    Random.seed!(seed)
    # Create probability distributions to generate noisy data
    noisy_data_generator = [dist(i) for i in perfect_data]
    # Generate noisy data 
    noisy_data = Vector{Float64}()
    for i in eachindex(noisy_data_generator) 
        append!(noisy_data, Distributions.rand(noisy_data_generator[i], 1))
    end
    return perfect_data, noisy_data
end

function generate_incidence_data(index::Integer, prob, alg, times::AbstractVector{T}; kwargs...) where T<:Real
    sol = solve(
        prob,
        alg,
        saveat=times,
        save_idxs=index;
        kwargs...
    )
    cumulativeData = sol[1,:]
    perfect_data = Vector{Float64}()
    for i in eachindex(cumulativeData)
        if i == 1
            append!(perfect_data, 0)
        else 
            append!(perfect_data, sol(times[i]) - sol(times[i-1]))
        end
    end
    return perfect_data
end 

function generate_incidence_data(sol)
    incidence_data = Vector{Float64}()
    for i in eachindex(sol)
        if i == 1
            append!(incidence_data, 0)
        else 
            append!(incidence_data, sol[i] - sol[i-1])
        end
    end
    return incidence_data
end