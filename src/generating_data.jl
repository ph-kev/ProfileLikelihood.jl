function generate_data(index, seed, dist, prob, alg, times; incidenceStatus = false, kwargs...)
    if incidenceStatus == false
        # Solve ODE
        sol = solve(
            prob,
            alg,
            saveat=times,
            save_idxs=index;
            kwargs...
        )
        # perfectData = [sol(t) for t in times]
        perfectData = abs.(sol[1,:])
    else
        perfectData = generate_incidence_data(index, prob, alg, times; kwargs...)
        perfectData = abs.(perfectData)
    end
    # Set seed for reproducibility 
    Random.seed!(seed)
    # Create probability distributions to generate noisy data
    noisyDataGenerator = [dist(i) for i in perfectData]
    # Generate noisy data 
    noisyData = Vector{Float64}()
    for i in eachindex(noisyDataGenerator) 
        append!(noisyData, Distributions.rand(noisyDataGenerator[i], 1))
    end
    return perfectData, noisyData
end

function generate_incidence_data(index, prob, alg, times; kwargs...)
    sol = solve(
        prob,
        alg,
        saveat=times,
        save_idxs=index;
        kwargs...
    )
    cumulativeData = sol[1,:]
    perfectData = Vector{Float64}()
    for i in eachindex(cumulativeData)
        if i == 1
            append!(perfectData, 0)
        else 
            append!(perfectData, sol(times[i]) - sol(times[i-1]))
        end
    end
    return perfectData
end 

function generate_incidence_data(sol)
    incidenceData = Vector{Float64}()
    for i in eachindex(sol)
        if i == 1
            append!(incidenceData, 0)
        else 
            append!(incidenceData, sol[i] - sol[i-1])
        end
    end
    return incidenceData
end