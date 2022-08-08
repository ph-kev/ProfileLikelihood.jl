function likelihood(params, data::Vector{Vector{T}}, sol_obs, prob, alg, times, obj_arr; incidence_obs = [], param_index=0, param_eval=0.0, solver_diff_opts = Dict()) where {T<:Real} 
    sol_obs_copy = vcat(sol_obs, incidence_obs)
    if param_index != 0
        params_copy = copy(params)
        insert!(params_copy, param_index, param_eval)
        prob_cur = remake(prob, p=params_copy)
    else
        prob_cur = remake(prob, p=params)
    end
    # solve odes
    sol = solve(
        prob_cur,
        alg,
        saveat=times,
        save_idxs=sol_obs_copy;
        solver_diff_opts...
    )
    # loss
    loss = 0.0
    index = 0 
    for ind in eachindex(sol_obs)
        loss += obj_arr[ind](data[ind], sol[ind,:])
        index += 1
    end
    for ind in eachindex(incidence_obs)
        incidence_sol = generate_incidence_data(sol[ind + index,:])
        loss += obj_arr[ind + index](data[ind + index][2:end], incidence_sol[2:end]) # remove the first data point for cumulative data since it is always 0 and taking the log of 0 is -Inf  
    end
    return loss
end