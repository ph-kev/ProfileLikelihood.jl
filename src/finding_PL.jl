function min_point(param_index::Integer, loss::Real, param_fitted::AbstractVector{<:Real}) 
    return [param_fitted[param_index]], [loss]
end

function go_right_PL(step_size::Real, max_steps::Integer, param_index::Integer, 
                     param_fitted::AbstractVector{<:Real}, 
                     data::AbstractVector{<:AbstractVector{<:Real}}, 
                     sol_obs::AbstractVector{<:Integer}, 
                     threshold::Real, loss::Real, 
                     prob::SciMLBase.AbstractDEProblem, 
                     alg_diff::SciMLBase.AbstractDEAlgorithm, 
                     times::AbstractVector{<:Real}, 
                     obj_arr::AbstractVector, 
                     alg_opti, 
                     lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}; 
                     incidence_obs::AbstractVector{<:Integer}=[], 
                     solver_diff_opts::Dict=Dict(), opti_prob_opts::Dict=Dict(), 
                     opti_solver_opts::Dict=Dict(), print_status::Bool=false) 
    theta_right = Vector{Real}()
    sol_right = Vector{Real}()
    curr_loss = loss
    param_fitted_copy = copy(param_fitted)
    param_to_look_at = param_fitted_copy[param_index]
    param_guess = deleteat!(param_fitted_copy, param_index)
    iter = 0
    # check if the first step is out of bound 
    if param_to_look_at + step_size > ub[param_index]
        return theta_right, sol_right
    end
    while curr_loss < threshold && iter < max_steps && param_to_look_at < ub[param_index]
        # take one step to the right, compute, and iterate 
        param_to_look_at = param_to_look_at + step_size
        append!(theta_right, param_to_look_at)
        loss, param_guess = estimate_params(param_guess, data, sol_obs, prob, alg_diff, 
        times, obj_arr, alg_opti, lb, ub; incidence_obs=incidence_obs, 
        param_index=param_index, param_eval=param_to_look_at, 
        solver_diff_opts=solver_diff_opts, opti_prob_opts=opti_prob_opts, 
        opti_solver_opts=opti_solver_opts, print_status=print_status)
        append!(sol_right, loss)
        iter = iter + 1
        curr_loss = loss
    end
    return theta_right, sol_right
end

function go_left_PL(step_size::Real, max_steps::Integer, param_index::Integer, 
                    param_fitted::AbstractVector{<:Real}, 
                    data::AbstractVector{<:AbstractVector{<:Real}}, 
                    sol_obs::AbstractVector{<:Integer}, 
                    threshold::Real, loss::Real, 
                    prob::SciMLBase.AbstractDEProblem, 
                    alg_diff::SciMLBase.AbstractDEAlgorithm, 
                    times::AbstractVector{<:Real}, 
                    obj_arr::AbstractVector, 
                    alg_opti, 
                    lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}; 
                    incidence_obs::AbstractVector{<:Integer}=[],  
                    solver_diff_opts::Dict=Dict(), opti_prob_opts::Dict=Dict(), 
                    opti_solver_opts::Dict=Dict(), print_status::Bool=false) 
    thetaLeft = Vector{Real}()
    solLeft = Vector{Real}()
    curr_loss = loss
    param_fitted_copy = copy(param_fitted)
    param_to_look_at = param_fitted_copy[param_index]
    param_guess = deleteat!(param_fitted_copy, param_index)
    iter = 0
    # check if the first step is out of bound 
    if param_to_look_at - step_size < lb[param_index]
        return reverse(thetaLeft), reverse(solLeft)
    end
    while curr_loss < threshold && iter < max_steps && param_to_look_at > lb[param_index]
        # take one step to the left, compute, and iterate 
        param_to_look_at = param_to_look_at - step_size
        append!(thetaLeft, param_to_look_at)
        loss, param_guess = estimate_params(param_guess, data, sol_obs, prob, alg_diff, 
        times, obj_arr, alg_opti, lb, ub; incidence_obs=incidence_obs, 
        param_index=param_index, param_eval=param_to_look_at, 
        solver_diff_opts=solver_diff_opts, opti_prob_opts=opti_prob_opts, 
        opti_solver_opts=opti_solver_opts, print_status=print_status)
        append!(solLeft, loss)
        iter = iter + 1
        curr_loss = loss
    end
    return reverse(thetaLeft), reverse(solLeft)
end

function find_profile_likelihood(step_size::Real, max_steps::Integer, param_index::Integer,
                                 param_fitted::AbstractVector{<:Real}, 
                                 data::AbstractVector{<:AbstractVector{<:Real}}, 
                                 sol_obs::AbstractVector{<:Integer}, 
                                 threshold::Real, loss::Real, 
                                 prob::SciMLBase.AbstractDEProblem, 
                                 alg_diff::SciMLBase.AbstractDEAlgorithm,
                                 times::AbstractVector{<:Real},
                                 obj_arr::AbstractVector, 
                                 alg_opti, 
                                 lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}; 
                                 incidence_obs=[], 
                                 solver_diff_opts::Dict=Dict(), opti_prob_opts::Dict=Dict(), 
                                 opti_solver_opts::Dict=Dict(), 
                                 print_status::Bool=false, pl_const::Real=0.0) 
    # find the right part of the profile likelihood graph 
    theta_right, sol_right = go_right_PL(step_size, max_steps, param_index, param_fitted, 
    data, sol_obs, threshold, loss, prob, alg_diff, times, obj_arr, alg_opti, lb, ub; 
    incidence_obs=incidence_obs, solver_diff_opts=solver_diff_opts, 
    opti_prob_opts=opti_prob_opts, opti_solver_opts=opti_solver_opts, 
    print_status=print_status)
    # find the left part of the profile likelihood graph 
    thetaLeft, solLeft = go_left_PL(step_size, max_steps, param_index, param_fitted, data, 
    sol_obs, threshold, loss, prob, alg_diff, times, obj_arr, alg_opti, lb, ub; 
    incidence_obs=incidence_obs, solver_diff_opts=solver_diff_opts, 
    opti_prob_opts=opti_prob_opts, opti_solver_opts=opti_solver_opts, 
    print_status=print_status)
    thetaPoint, solPoint = min_point(param_index, loss, param_fitted)
    # append the left part, midpoint, and right part 
    theta = vcat(thetaLeft, thetaPoint, theta_right)
    sol = vcat(solLeft, solPoint, sol_right)
    sol = sol .+ pl_const
    return theta, sol
end

function go_right_PL_multistart(step_size::Real, max_steps::Integer, param_index::Integer, 
                                param_fitted::AbstractVector{<:Real}, 
                                data::AbstractVector{<:AbstractVector{<:Real}}, 
                                sol_obs::AbstractVector{<:Integer}, 
                                threshold::Real, loss::Real, 
                                prob::SciMLBase.AbstractDEProblem, 
                                alg_diff::SciMLBase.AbstractDEAlgorithm, 
                                times::AbstractVector{<:Real}, 
                                obj_arr::AbstractVector, 
                                alg_opti, alg_opti_local, 
                                lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}; 
                                incidence_obs::AbstractVector{<:Integer}=[], 
                                solver_diff_opts::Dict=Dict(), opti_prob_opts::Dict=Dict(), 
                                opti_solver_opts::Dict=Dict(), print_status::Bool=false)
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
        loss, param_guess = estimate_params_multistart(param_guess, data, sol_obs, prob, 
        alg_diff, times, obj_arr, alg_opti, alg_opti_local, lb, ub; 
        incidence_obs=incidence_obs, param_index=param_index, param_eval=param_to_look_at, 
        solver_diff_opts=solver_diff_opts, opti_prob_opts=opti_prob_opts, 
        opti_solver_opts=opti_solver_opts, print_status=print_status)
        append!(sol_right, loss)
        iter = iter + 1
        curr_loss = loss
    end
    return theta_right, sol_right
end

function go_left_PL_multistart(step_size::Real, max_steps::Integer, param_index::Integer, 
                               param_fitted::AbstractVector{<:Real}, 
                               data::AbstractVector{<:AbstractVector{<:Real}}, 
                               sol_obs::AbstractVector{<:Integer}, 
                               threshold::Real, loss::Real, 
                               prob::SciMLBase.AbstractDEProblem, 
                               alg_diff::SciMLBase.AbstractDEAlgorithm, 
                               times::AbstractVector{<:Real}, 
                               obj_arr::AbstractVector, 
                               alg_opti, 
                               alg_opti_local, 
                               lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}; 
                               incidence_obs::AbstractVector{<:Integer}=[], 
                               solver_diff_opts=Dict(), opti_prob_opts=Dict(), 
                               opti_solver_opts=Dict(), print_status=false) 
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
        loss, param_guess = estimate_params_multistart(param_guess, data, sol_obs, prob, 
        alg_diff, times, obj_arr, alg_opti, alg_opti_local, lb, ub; 
        incidence_obs=incidence_obs, param_index=param_index, param_eval=param_to_look_at, 
        solver_diff_opts=solver_diff_opts, opti_prob_opts=opti_prob_opts, 
        opti_solver_opts=opti_solver_opts, print_status=print_status)
        append!(solLeft, loss)
        iter = iter + 1
        curr_loss = loss
    end
    return reverse(thetaLeft), reverse(solLeft)
end

function find_profile_likelihood_multistart(step_size::Real, max_steps::Integer, param_index::Integer, 
                                            param_fitted::AbstractVector{<:Real}, 
                                            data::AbstractVector{<:AbstractVector{<:Real}}, 
                                            sol_obs::AbstractVector{<:Integer}, 
                                            threshold::Real, loss::Real, 
                                            prob::SciMLBase.AbstractDEProblem, 
                                            alg_diff::SciMLBase.AbstractDEAlgorithm,  
                                            times::AbstractVector{<:Real}, 
                                            obj_arr::AbstractVector, 
                                            alg_opti, alg_opti_local, 
                                            lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}; 
                                            incidence_obs::AbstractVector{<:Integer}=[], 
                                            solver_diff_opts=Dict(), opti_prob_opts=Dict(), 
                                            opti_solver_opts=Dict(), 
                                            print_status::Bool=false, pl_const::Real=0.0) 
    theta_right, sol_right = go_right_PL_multistart(step_size, max_steps, param_index, 
    param_fitted, data, sol_obs, threshold, loss, prob, alg_diff, times, obj_arr, alg_opti, 
    alg_opti_local, lb, ub; incidence_obs=incidence_obs, solver_diff_opts=solver_diff_opts, 
    opti_prob_opts=opti_prob_opts, opti_solver_opts=opti_solver_opts, 
    print_status=print_status)
    thetaLeft, solLeft = go_left_PL_multistart(step_size, max_steps, param_index, 
    param_fitted, data, sol_obs, threshold, loss, prob, alg_diff, times, obj_arr, alg_opti, 
    alg_opti_local, lb, ub; incidence_obs=incidence_obs, solver_diff_opts=solver_diff_opts, 
    opti_prob_opts=opti_prob_opts, opti_solver_opts=opti_solver_opts, 
    print_status=print_status)
    thetaPoint, solPoint = min_point(param_index, loss, param_fitted)
    theta = vcat(thetaLeft, thetaPoint, theta_right)
    sol = vcat(solLeft, solPoint, sol_right)
    sol = sol .+ pl_const
    return theta, sol
end