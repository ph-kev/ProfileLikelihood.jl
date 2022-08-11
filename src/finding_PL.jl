function min_point(param_index::Integer, loss, param_fitted::AbstractVector{T}) where {T<:Real}
    return [param_fitted[param_index]], [loss]
end

function go_right_PL(step_size::T, max_steps::Integer, param_index::Integer, param_fitted::AbstractVector{T}, data::Vector{Vector{T}}, sol_obs::AbstractVector{Any}, threshold::T, loss::T, prob, alg_diff, times::AbstractVector{T}, obj_arr::AbstractVector, alg_opti, lb::AbstractVector{T}, ub::AbstractVector{T}; incidence_obs=[], solver_diff_opts=Dict(), opti_prob_opts=Dict(), opti_solver_opts=Dict(), print_status=false) where {T<:Real}
    theta_right = Vector{Real}()
    sol_right = Vector{Real}()
    curr_loss = loss
    param_fitted_copy = copy(param_fitted)
    param_to_look_at = param_fitted_copy[param_index]
    param_guess = deleteat!(param_fitted_copy, param_index)
    iter = 0
    if param_to_look_at + step_size > ub[param_index]
        return theta_right, sol_right
    end
    while curr_loss < threshold && iter < max_steps && param_to_look_at < ub[param_index]
        param_to_look_at = param_to_look_at + step_size
        append!(theta_right, param_to_look_at)
        loss, param_guess = estimate_params(param_guess, data, sol_obs, prob, alg_diff, times, obj_arr, alg_opti, lb, ub; incidence_obs=incidence_obs, param_index=param_index, param_eval=param_to_look_at, solver_diff_opts=solver_diff_opts, opti_prob_opts=opti_prob_opts, opti_solver_opts=opti_solver_opts, print_status=print_status)
        append!(sol_right, loss)
        iter = iter + 1
        curr_loss = loss
    end
    return theta_right, sol_right
end

function go_left_PL(step_size::T, max_steps::Integer, param_index::Integer, param_fitted::AbstractVector{T}, data::Vector{Vector{T}}, sol_obs::AbstractVector{Any}, threshold::T, loss::T, prob, alg_diff, times::AbstractVector{T}, obj_arr::AbstractVector, alg_opti, lb::AbstractVector{T}, ub::AbstractVector{T}; incidence_obs=[], solver_diff_opts=Dict(), opti_prob_opts=Dict(), opti_solver_opts=Dict(), print_status=false) where {T<:Real}
    thetaLeft = Vector{Real}()
    solLeft = Vector{Real}()
    curr_loss = loss
    param_fitted_copy = copy(param_fitted)
    param_to_look_at = param_fitted_copy[param_index]
    param_guess = deleteat!(param_fitted_copy, param_index)
    iter = 0
    if param_to_look_at - step_size < lb[param_index]
        return reverse(thetaLeft), reverse(solLeft)
    end
    while curr_loss < threshold && iter < max_steps && param_to_look_at > lb[param_index]
        param_to_look_at = param_to_look_at - step_size
        append!(thetaLeft, param_to_look_at)
        loss, param_guess = estimate_params(param_guess, data, sol_obs, prob, alg_diff, times, obj_arr, alg_opti, lb, ub; incidence_obs=incidence_obs, param_index=param_index, param_eval=param_to_look_at, solver_diff_opts=solver_diff_opts, opti_prob_opts=opti_prob_opts, opti_solver_opts=opti_solver_opts, print_status=print_status)
        append!(solLeft, loss)
        iter = iter + 1
        curr_loss = loss
    end
    return reverse(thetaLeft), reverse(solLeft)
end

function find_profile_likelihood(step_size::T, max_steps::Integer, param_index::Integer, param_fitted::AbstractVector{T}, data::Vector{Vector{T}}, sol_obs::AbstractVector{Any}, threshold::T, loss::T, prob, alg_diff, times::AbstractVector{T}, obj_arr::AbstractVector, alg_opti, lb::AbstractVector{T}, ub::AbstractVector{T}; incidence_obs=[], solver_diff_opts=Dict(), opti_prob_opts=Dict(), opti_solver_opts=Dict(), print_status=false, pl_const=0.0) where {T<:Real}
    theta_right, sol_right = go_right_PL(step_size, max_steps, param_index, param_fitted, data, sol_obs, threshold, loss, prob, alg_diff, times, obj_arr, alg_opti, lb, ub; incidence_obs=incidence_obs, solver_diff_opts=solver_diff_opts, opti_prob_opts=opti_prob_opts, opti_solver_opts=opti_solver_opts, print_status=print_status)
    thetaLeft, solLeft = go_left_PL(step_size, max_steps, param_index, param_fitted, data, sol_obs, threshold, loss, prob, alg_diff, times, obj_arr, alg_opti, lb, ub; incidence_obs=incidence_obs, solver_diff_opts=solver_diff_opts, opti_prob_opts=opti_prob_opts, opti_solver_opts=opti_solver_opts, print_status=print_status)
    thetaPoint, solPoint = min_point(param_index, loss, param_fitted)
    theta = vcat(thetaLeft, thetaPoint, theta_right)
    sol = vcat(solLeft, solPoint, sol_right)
    sol = sol .+ pl_const
    return theta, sol
end