function likelihood(params::AbstractVector{<:Real}, 
                    data::AbstractVector{<:AbstractVector{<:Real}},
                    sol_obs::AbstractVector{<:Integer}, 
                    prob::SciMLBase.AbstractDEProblem, 
                    alg::SciMLBase.AbstractDEAlgorithm, 
                    times::AbstractVector{<:Real}, 
                    obj_arr::AbstractVector; 
                    incidence_obs::AbstractVector{<:Integer} = [], 
                    param_index::Integer=0, 
                    param_eval::Real=0.0, 
                    solver_diff_opts::Dict = Dict())
                    sol_obs_copy = vcat(sol_obs, incidence_obs)
    if param_index != 0 # check if the parameter is fixed or not
        params_copy = copy(params)
        insert!(params_copy, param_index, param_eval)
        prob_cur = remake(prob, p=params_copy)
    else
        prob_cur = remake(prob, p=params)
    end
    # solve ODEs
    sol = solve(
        prob_cur,
        alg,
        saveat=times,
        save_idxs=sol_obs_copy;
        solver_diff_opts...
    )
    loss = 0.0
    index = 0 
    for ind in eachindex(sol_obs)
        loss += obj_arr[ind](data[ind], sol[ind,:])
        index += 1
    end
    for ind in eachindex(incidence_obs)
        incidence_sol = generate_incidence_data(sol[ind + index,:])
        #= remove the first data point for cumulative data since it is always 0 and 
        taking the log of 0 is -Inf =# 
        loss += obj_arr[ind + index](data[ind + index][2:end], incidence_sol[2:end]) 
    end
    return loss
end