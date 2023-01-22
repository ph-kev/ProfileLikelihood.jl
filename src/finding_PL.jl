"""
    min_point(param_index::Integer, loss::Real, param_fitted::AbstractVector{<:Real}) 

This returns the minimum point of the profile likelihood plot where the point is 
(`index`th element of `param_fitted`,`loss`).
"""
function min_point(param_index::Integer, loss::Real, param_fitted::AbstractVector{<:Real}) 
    return [param_fitted[param_index]], [loss]
end

"""
    go_right_PL(step_size::Real, max_steps::Integer, param_index::Integer, 
                param_fitted::AbstractVector{<:Real}, 
                data::AbstractVector{<:AbstractVector{<:Real}}, 
                sol_obs::AbstractVector{<:Integer}, 
                threshold::Real, loss::Real, 
                prob::SciMLBase.AbstractDEProblem, 
                alg_diff::SciMLBase.AbstractDEAlgorithm, 
                times::AbstractVector{<:Real}, 
                obj_arr::AbstractVector{<:Function}, 
                alg_opti, 
                lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}; 
                incidence_obs::AbstractVector{<:Integer}=Int64[], 
                solver_diff_opts::Dict=Dict(), opti_prob_opts::Dict=Dict(), 
                opti_solver_opts::Dict=Dict(), print_status::Bool=false) 

This compute the points to the right of the minimum point of the profile likelihood.

# Arguments 
- `step_size::Real`: Step size to take for computing points.
- `max_steps::Integer`: Maximum steps allowed.
- `param_index::Integer`: Index of the parameter vector that is fixed. 
- `param_fitted::AbstractVector{<:Real}`: Optimized parameters. Should be the return vector `minimizer` of `estimate_params` or `estimate_params_multistart`.
- `data::AbstractVector{<:AbstractVector{<:Real}}`: Vector of data used for optimizing parameters. `data` must be in the same order as in `sol_obs` and `incidence_obs`.  
- `sol_obs::AbstractVector{<:Integer}`: Indices of the state variables of the DEs to be used for sampling data points. 
- `threshold::Real`: Value for which the algorithm will stop computing points if the loss of the computed points is greater than the threshold. Does not necessarily need to be the same threshold as the `threshold` value in `find_threshold`.
- `loss::Real`: Loss of the fitted parameters according to the objective function(s). Should be the return value `minimum` of `estimate_params` or `estimate_params_multistart`.
- `prob::SciMLBase.AbstractDEProblem`: DE Problem (see `DifferentialEquations.jl` for more information).
- `alg_diff::SciMLBase.AbstractDEAlgorithm`: DE Solver Algorithm (see `DifferentialEquations.jl` for more information).
- `times::AbstractVector{<:Real}`: Times that the data points will be sampled at.
- `obj_arr::AbstractVector{<:Function}`: Vector of objective functions. 
- `alg_opti`: Optimization algorithm (see `Optimization.jl` for a list of algorithms that could be used).
- `lb::AbstractVector{<:Real}`: Lower bound (does not need to be changed if `param_index` and `param_eval` are used).
- `ub::AbstractVector{<:Real}`: Upper bound (does not need to be changed if `param_index` and `param_eval` are used).

# Keywords 
- `incidence_obs::AbstractVector{<:Integer}=Int64[]`: Indices of the state variables of the DEs to find incidence data of. The state variables must be cumulative data.  
- `solver_diff_opts::Dict=Dict()`: Keyword arguments to be passed into the DE solver. See `DifferentialEquations.jl`'s Common Solver Options.
- `opti_prob_opts::Dict=Dict()`: Keyword arguments to be passed into the optimization problem. See `Optimization.jl`'s Defining OptimizationProblems.
- `opti_solver_opts::Dict=Dict()`: Keyword arguments to be passed into the optimization solver. See `Optimization.jl`'s Common Solver Options.
- `print_status::Bool=false`: Determine whether the original output of the optimization algorithm is printed or not. 

# Returns
- `theta_right`: Values of the fixed parameter that is explored to the right of the minimum point.
- `sol_right`: Value of `likelihood` computed at the fixed parameter where the other parameters are optimized.
"""
function go_right_PL(step_size::Real, max_steps::Integer, param_index::Integer, 
                     param_fitted::AbstractVector{<:Real}, 
                     data::AbstractVector{<:AbstractVector{<:Real}}, 
                     sol_obs::AbstractVector{<:Integer}, 
                     threshold::Real, loss::Real, 
                     prob::SciMLBase.AbstractDEProblem, 
                     alg_diff::SciMLBase.AbstractDEAlgorithm, 
                     times::AbstractVector{<:Real}, 
                     obj_arr::AbstractVector{<:Function}, 
                     alg_opti, 
                     lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}; 
                     incidence_obs::AbstractVector{<:Integer}=Int64[], 
                     solver_diff_opts::Dict=Dict(), opti_prob_opts::Dict=Dict(), 
                     opti_solver_opts::Dict=Dict(), print_status::Bool=false) 
    theta_right = Vector{Float64}()
    sol_right = Vector{Float64}()
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

"""
    go_left_PL(step_size::Real, max_steps::Integer, param_index::Integer, 
               param_fitted::AbstractVector{<:Real}, 
               data::AbstractVector{<:AbstractVector{<:Real}}, 
               sol_obs::AbstractVector{<:Integer}, 
               threshold::Real, loss::Real, 
               prob::SciMLBase.AbstractDEProblem, 
               alg_diff::SciMLBase.AbstractDEAlgorithm, 
               times::AbstractVector{<:Real}, 
               obj_arr::AbstractVector{<:Function}, 
               alg_opti, 
               lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}; 
               incidence_obs::AbstractVector{<:Integer}=Int64[],  
               solver_diff_opts::Dict=Dict(), opti_prob_opts::Dict=Dict(), 
               opti_solver_opts::Dict=Dict(), print_status::Bool=false)

This compute the points to the left of the minimum point of the profile likelihood.

# Arguments 
- `step_size::Real`: Step size to take for computing points.
- `max_steps::Integer`: Maximum steps allowed.
- `param_index::Integer`: Index of the parameter vector that is fixed. 
- `param_fitted::AbstractVector{<:Real}`: Optimized parameters. Should be the return vector `minimizer` of `estimate_params` or `estimate_params_multistart`.
- `data::AbstractVector{<:AbstractVector{<:Real}}`: Vector of data used for optimizing parameters. `data` must be in the same order as in `sol_obs` and `incidence_obs`.  
- `sol_obs::AbstractVector{<:Integer}`: Indices of the state variables of the DEs to be used for sampling data points. 
- `threshold::Real`: Value for which the algorithm will stop computing points if the loss of the computed points is greater than the threshold. Does not necessarily need to be the same threshold as the `threshold` value in `find_threshold`.
- `loss::Real`: Loss of the fitted parameters according to the objective function(s). Should be the return value `minimum` of `estimate_params` or `estimate_params_multistart`.
- `prob::SciMLBase.AbstractDEProblem`: DE Problem (see `DifferentialEquations.jl` for more information).
- `alg_diff::SciMLBase.AbstractDEAlgorithm`: DE Solver Algorithm (see `DifferentialEquations.jl` for more information).
- `times::AbstractVector{<:Real}`: Times that the data points will be sampled at.
- `obj_arr::AbstractVector{<:Function}`: Vector of objective functions. 
- `alg_opti`: Optimization algorithm (see `Optimization.jl` for a list of algorithms that could be used).
- `lb::AbstractVector{<:Real}`: Lower bound (does not need to be changed if `param_index` and `param_eval` are used).
- `ub::AbstractVector{<:Real}`: Upper bound (does not need to be changed if `param_index` and `param_eval` are used).
           
# Keywords 
- `incidence_obs::AbstractVector{<:Integer}=Int64[]`: Indices of the state variables of the DEs to find incidence data of. The state variables must be cumulative data.  
- `solver_diff_opts::Dict=Dict()`: Keyword arguments to be passed into the DE solver. See `DifferentialEquations.jl`'s Common Solver Options.
- `opti_prob_opts::Dict=Dict()`: Keyword arguments to be passed into the optimization problem. See `Optimization.jl`'s Defining OptimizationProblems.
- `opti_solver_opts::Dict=Dict()`: Keyword arguments to be passed into the optimization solver. See `Optimization.jl`'s Common Solver Options.
- `print_status::Bool=false`: Determine whether the original output of the optimization algorithm is printed or not. 
           
# Returns
- `theta_left`: Values of the fixed parameter that is explored to the left of the minimum point.
- `sol_left`: Value of `likelihood` computed at the fixed parameter where the other parameters are optimized.
"""
function go_left_PL(step_size::Real, max_steps::Integer, param_index::Integer, 
                    param_fitted::AbstractVector{<:Real}, 
                    data::AbstractVector{<:AbstractVector{<:Real}}, 
                    sol_obs::AbstractVector{<:Integer}, 
                    threshold::Real, loss::Real, 
                    prob::SciMLBase.AbstractDEProblem, 
                    alg_diff::SciMLBase.AbstractDEAlgorithm, 
                    times::AbstractVector{<:Real}, 
                    obj_arr::AbstractVector{<:Function}, 
                    alg_opti, 
                    lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}; 
                    incidence_obs::AbstractVector{<:Integer}=Int64[],  
                    solver_diff_opts::Dict=Dict(), opti_prob_opts::Dict=Dict(), 
                    opti_solver_opts::Dict=Dict(), print_status::Bool=false) 
    thetaLeft = Vector{Float64}()
    solLeft = Vector{Float64}()
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

"""
    find_profile_likelihood(step_size::Real, max_steps::Integer, param_index::Integer,
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
                            incidence_obs::AbstractVector{<:Integer}=Int64[], 
                            solver_diff_opts::Dict=Dict(), opti_prob_opts::Dict=Dict(), 
                            opti_solver_opts::Dict=Dict(), 
                            print_status::Bool=false, pl_const::Real=0.0)

This find the profile likelihood of the `param_index`th parameter in `param_fitted`. 

This implements a simple fixed step size algorithm to compute profile likelihood. It start at 
the minimum point of the profile likelihood and take step in the right direction and left direction.
The algorithm stop taking steps if the maximum allowed steps `max_step` is reached or if the 
points being computed is above the `threshold`. 

# Arguments
- `step_size::Real`: Step size to take for computing points.
- `max_steps::Integer`: Maximum steps allowed.
- `param_index::Integer`: Index of the parameter vector that is fixed. 
- `param_fitted::AbstractVector{<:Real}`: Optimized parameters. Should be the return vector `minimizer` of `estimate_params` or `estimate_params_multistart`.
- `data::AbstractVector{<:AbstractVector{<:Real}}`: Vector of data used for optimizing parameters. `data` must be in the same order as in `sol_obs` and `incidence_obs`.  
- `sol_obs::AbstractVector{<:Integer}`: Indices of the state variables of the DEs to be used for sampling data points. 
- `threshold::Real`: Value for which the algorithm will stop computing points if the loss of the computed points is greater than the threshold. Does not necessarily need to be the same threshold as the `threshold` value in `find_threshold`.
- `loss::Real`: Loss of the fitted parameters according to the objective function(s). Should be the return value `minimum` of `estimate_params` or `estimate_params_multistart`.
- `prob::SciMLBase.AbstractDEProblem`: DE Problem (see `DifferentialEquations.jl` for more information).
- `alg_diff::SciMLBase.AbstractDEAlgorithm`: DE Solver Algorithm (see `DifferentialEquations.jl` for more information).
- `times::AbstractVector{<:Real}`: Times that the data points will be sampled at.
- `obj_arr::AbstractVector{<:Function}`: Vector of objective functions. 
- `alg_opti`: Optimization algorithm (see `Optimization.jl` for a list of algorithms that could be used).
- `lb::AbstractVector{<:Real}`: Lower bound (does not need to be changed if `param_index` and `param_eval` are used).
- `ub::AbstractVector{<:Real}`: Upper bound (does not need to be changed if `param_index` and `param_eval` are used).

# Keywords 
- `incidence_obs::AbstractVector{<:Integer}=Int64[]`: Indices of the state variables of the DEs to find incidence data of. The state variables must be cumulative data.  
- `solver_diff_opts::Dict=Dict()`: Keyword arguments to be passed into the DE solver. See `DifferentialEquations.jl`'s Common Solver Options.
- `opti_prob_opts::Dict=Dict()`: Keyword arguments to be passed into the optimization problem. See `Optimization.jl`'s Defining OptimizationProblems.
- `opti_solver_opts::Dict=Dict()`: Keyword arguments to be passed into the optimization solver. See `Optimization.jl`'s Common Solver Options.
- `print_status::Bool=false`: Determine whether the original output of the optimization algorithm is printed or not. 
- `pl_const::Real=0.0`: Constant that is added to loss of the computed points. The constant can be computed by `likelihood_const`.

# Returns
- `theta`: Values of the fixed parameter that is explored.
- `sol`: Value of `likelihood` computed at the fixed parameter where the other parameters are optimized.
"""
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
                                 incidence_obs::AbstractVector{<:Integer}=Int64[], 
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

"""
    go_right_PL_multistart(step_size::Real, max_steps::Integer, param_index::Integer, 
                           param_fitted::AbstractVector{<:Real}, 
                           data::AbstractVector{<:AbstractVector{<:Real}}, 
                           sol_obs::AbstractVector{<:Integer}, 
                           threshold::Real, loss::Real, 
                           prob::SciMLBase.AbstractDEProblem, 
                           alg_diff::SciMLBase.AbstractDEAlgorithm, 
                           times::AbstractVector{<:Real}, 
                           obj_arr::AbstractVector::{<:Function}, 
                           alg_opti, alg_opti_local
                           lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}; 
                           incidence_obs::AbstractVector{<:Integer}=Int64[], 
                           solver_diff_opts::Dict=Dict(), opti_prob_opts::Dict=Dict(), 
                           opti_solver_opts::Dict=Dict(), print_status::Bool=false) 

This compute the points to the right of the minimum point of the profile likelihood.

This is used by `find_profile_likelihood_multistart`.

# Arguments 
- `step_size::Real`: Step size to take for computing points.
- `max_steps::Integer`: Maximum steps allowed.
- `param_index::Integer`: Index of the parameter vector that is fixed. 
- `param_fitted::AbstractVector{<:Real}`: Optimized parameters. Should be the return vector `minimizer` of `estimate_params` or `estimate_params_multistart`.
- `data::AbstractVector{<:AbstractVector{<:Real}}`: Vector of data used for optimizing parameters. `data` must be in the same order as in `sol_obs` and `incidence_obs`.  
- `sol_obs::AbstractVector{<:Integer}`: Indices of the state variables of the DEs to be used for sampling data points. 
- `threshold::Real`: Value for which the algorithm will stop computing points if the loss of the computed points is greater than the threshold. Does not necessarily need to be the same threshold as the `threshold` value in `find_threshold`.
- `loss::Real`: Loss of the fitted parameters according to the objective function(s). Should be the return value `minimum` of `estimate_params` or `estimate_params_multistart`.
- `prob::SciMLBase.AbstractDEProblem`: DE Problem (see `DifferentialEquations.jl` for more information).
- `alg_diff::SciMLBase.AbstractDEAlgorithm`: DE Solver Algorithm (see `DifferentialEquations.jl` for more information).
- `times::AbstractVector{<:Real}`: Times that the data points will be sampled at.
- `obj_arr::AbstractVector{<:Function}`: Vector of objective functions. 
- `alg_opti`: Global optimization algorithm. Typically, `MultistartOptimization.TikTak(n)` where `n` is the number of starting points generated from the Sobol sequence.
- `alg_opti_local`: Local optimization algorithm. Must be an algorithm from `NLopt.jl`.
- `lb::AbstractVector{<:Real}`: Lower bound (does not need to be changed if `param_index` and `param_eval` are used).
- `ub::AbstractVector{<:Real}`: Upper bound (does not need to be changed if `param_index` and `param_eval` are used).

# Keywords 
- `incidence_obs::AbstractVector{<:Integer}=Int64[]`: Indices of the state variables of the DEs to find incidence data of. The state variables must be cumulative data.  
- `solver_diff_opts::Dict=Dict()`: Keyword arguments to be passed into the DE solver. See `DifferentialEquations.jl`'s Common Solver Options.
- `opti_prob_opts::Dict=Dict()`: Keyword arguments to be passed into the optimization problem. See `Optimization.jl`'s Defining OptimizationProblems.
- `opti_solver_opts::Dict=Dict()`: Keyword arguments to be passed into the optimization solver. See `Optimization.jl`'s Common Solver Options.
- `print_status::Bool=false`: Determine whether the original output of the optimization algorithm is printed or not. 

# Returns
- `theta_right`: Values of the fixed parameter that is explored to the right of the minimum point.
- `sol_right`: Value of `likelihood` computed at the fixed parameter where the other parameters are optimized.
"""
function go_right_PL_multistart(step_size::Real, max_steps::Integer, param_index::Integer, 
                                param_fitted::AbstractVector{<:Real}, 
                                data::AbstractVector{<:AbstractVector{<:Real}}, 
                                sol_obs::AbstractVector{<:Integer}, 
                                threshold::Real, loss::Real, 
                                prob::SciMLBase.AbstractDEProblem, 
                                alg_diff::SciMLBase.AbstractDEAlgorithm, 
                                times::AbstractVector{<:Real}, 
                                obj_arr::AbstractVector{<:Function}, 
                                alg_opti, alg_opti_local, 
                                lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}; 
                                incidence_obs::AbstractVector{<:Integer}=Int64[], 
                                solver_diff_opts::Dict=Dict(), opti_prob_opts::Dict=Dict(), 
                                opti_solver_opts::Dict=Dict(), print_status::Bool=false)
    theta_right = Vector{Float64}()
    sol_right = Vector{Float64}()
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

"""
    go_left_PL_multistart(step_size::Real, max_steps::Integer, param_index::Integer, 
                          param_fitted::AbstractVector{<:Real}, 
                          data::AbstractVector{<:AbstractVector{<:Real}}, 
                          sol_obs::AbstractVector{<:Integer}, 
                          threshold::Real, loss::Real, 
                          prob::SciMLBase.AbstractDEProblem, 
                          alg_diff::SciMLBase.AbstractDEAlgorithm, 
                          times::AbstractVector{<:Real}, 
                          obj_arr::AbstractVector{<:Function}, 
                          alg_opti, alg_opti_local
                          lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}; 
                          incidence_obs::AbstractVector{<:Integer}=Int64[], 
                          solver_diff_opts::Dict=Dict(), opti_prob_opts::Dict=Dict(), 
                          opti_solver_opts::Dict=Dict(), print_status::Bool=false) 

This compute the points to the left of the minimum point of the profile likelihood.

This is used by `find_profile_likelihood_multistart`.

# Arguments 
- `step_size::Real`: Step size to take for computing points.
- `max_steps::Integer`: Maximum steps allowed.
- `param_index::Integer`: Index of the parameter vector that is fixed. 
- `param_fitted::AbstractVector{<:Real}`: Optimized parameters. Should be the return vector `minimizer` of `estimate_params` or `estimate_params_multistart`.
- `data::AbstractVector{<:AbstractVector{<:Real}}`: Vector of data used for optimizing parameters. `data` must be in the same order as in `sol_obs` and `incidence_obs`.  
- `sol_obs::AbstractVector{<:Integer}`: Indices of the state variables of the DEs to be used for sampling data points. 
- `threshold::Real`: Value for which the algorithm will stop computing points if the loss of the computed points is greater than the threshold. Does not necessarily need to be the same threshold as the `threshold` value in `find_threshold`.
- `loss::Real`: Loss of the fitted parameters according to the objective function(s). Should be the return value `minimum` of `estimate_params` or `estimate_params_multistart`.
- `prob::SciMLBase.AbstractDEProblem`: DE Problem (see `DifferentialEquations.jl` for more information).
- `alg_diff::SciMLBase.AbstractDEAlgorithm`: DE Solver Algorithm (see `DifferentialEquations.jl` for more information).
- `times::AbstractVector{<:Real}`: Times that the data points will be sampled at.
- `obj_arr::AbstractVector{<:Function}`: Vector of objective functions. 
- `alg_opti`: Global optimization algorithm. Typically, `MultistartOptimization.TikTak(n)` where `n` is the number of starting points generated from the Sobol sequence.
- `alg_opti_local`: Local optimization algorithm. Must be an algorithm from `NLopt.jl`.
- `lb::AbstractVector{<:Real}`: Lower bound (does not need to be changed if `param_index` and `param_eval` are used).
- `ub::AbstractVector{<:Real}`: Upper bound (does not need to be changed if `param_index` and `param_eval` are used).

# Keywords 
- `incidence_obs::AbstractVector{<:Integer}=Int64[]`: Indices of the state variables of the DEs to find incidence data of. The state variables must be cumulative data.  
- `solver_diff_opts::Dict=Dict()`: Keyword arguments to be passed into the DE solver. See `DifferentialEquations.jl`'s Common Solver Options.
- `opti_prob_opts::Dict=Dict()`: Keyword arguments to be passed into the optimization problem. See `Optimization.jl`'s Defining OptimizationProblems.
- `opti_solver_opts::Dict=Dict()`: Keyword arguments to be passed into the optimization solver. See `Optimization.jl`'s Common Solver Options.
- `print_status::Bool=false`: Determine whether the original output of the optimization algorithm is printed or not. 

# Returns
- `theta_right`: Values of the fixed parameter that is explored to the left of the minimum point.
- `sol_right`: Value of `likelihood` computed at the fixed parameter where the other parameters are optimized.
"""
function go_left_PL_multistart(step_size::Real, max_steps::Integer, param_index::Integer, 
                               param_fitted::AbstractVector{<:Real}, 
                               data::AbstractVector{<:AbstractVector{<:Real}}, 
                               sol_obs::AbstractVector{<:Integer}, 
                               threshold::Real, loss::Real, 
                               prob::SciMLBase.AbstractDEProblem, 
                               alg_diff::SciMLBase.AbstractDEAlgorithm, 
                               times::AbstractVector{<:Real}, 
                               obj_arr::AbstractVector{<:Function}, 
                               alg_opti, alg_opti_local, 
                               lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}; 
                               incidence_obs::AbstractVector{<:Integer}=Int64[], 
                               solver_diff_opts=Dict(), opti_prob_opts=Dict(), 
                               opti_solver_opts=Dict(), print_status=false) 
    thetaLeft = Vector{Float64}()
    solLeft = Vector{Float64}()
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

"""
    find_profile_likelihood_multistart(step_size::Real, max_steps::Integer, param_index::Integer, 
                                       param_fitted::AbstractVector{<:Real}, 
                                       data::AbstractVector{<:AbstractVector{<:Real}}, 
                                       sol_obs::AbstractVector{<:Integer}, 
                                       threshold::Real, loss::Real, 
                                       prob::SciMLBase.AbstractDEProblem, 
                                       alg_diff::SciMLBase.AbstractDEAlgorithm,  
                                       times::AbstractVector{<:Real}, 
                                       obj_arr::AbstractVector{<:Function}, 
                                       alg_opti, alg_opti_local, 
                                       lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}; 
                                       incidence_obs::AbstractVector{<:Integer}=Int64[], 
                                       solver_diff_opts=Dict(), opti_prob_opts=Dict(), 
                                       opti_solver_opts=Dict(), 
                                       print_status::Bool=false, pl_const::Real=0.0)

This find the profile likelihood of the `param_index`th parameter in `param_fitted`. 

This implements a simple fixed step size algorithm to compute profile likelihood. It start at 
the minimum point of the profile likelihood and take step in the right direction and left direction.
The algorithm stop taking steps if the maximum allowed steps `max_step` is reached or if the 
points being computed is above the `threshold`. The difference between `find_profile_likelihood`
and `find_profile_likelihood_multistart` is that this function uses the multi-start optimization algorithm `alg_opti` from `MultiStartOptimization` and the local optimization algorithm `alg_opti_local` from the `NLopt.jl`. This is similar to the differences between `estimate_params` and `estimate_params_multistart`.

# Arguments
- `step_size::Real`: Step size to take for computing points.
- `max_steps::Integer`: Maximum steps allowed.
- `param_index::Integer`: Index of the parameter vector that is fixed. 
- `param_fitted::AbstractVector{<:Real}`: Optimized parameters. Should be the return vector `minimizer` of `estimate_params` or `estimate_params_multistart`.
- `data::AbstractVector{<:AbstractVector{<:Real}}`: Vector of data used for optimizing parameters. `data` must be in the same order as in `sol_obs` and `incidence_obs`.  
- `sol_obs::AbstractVector{<:Integer}`: Indices of the state variables of the DEs to be used for sampling data points. 
- `threshold::Real`: Value for which the algorithm will stop computing points if the loss of the computed points is greater than the threshold. Does not necessarily need to be the same threshold as the `threshold` value in `find_threshold`.
- `loss::Real`: Loss of the fitted parameters according to the objective function(s). Should be the return value `minimum` of `estimate_params` or `estimate_params_multistart`.
- `prob::SciMLBase.AbstractDEProblem`: DE Problem (see `DifferentialEquations.jl` for more information).
- `alg_diff::SciMLBase.AbstractDEAlgorithm`: DE Solver Algorithm (see `DifferentialEquations.jl` for more information).
- `times::AbstractVector{<:Real}`: Times that the data points will be sampled at.
- `obj_arr::AbstractVector{<:Function}`: Vector of objective functions. 
- `alg_opti`: Global optimization algorithm. Typically, `MultistartOptimization.TikTak(n)` where `n` is the number of starting points generated from the Sobol sequence.
- `alg_opti_local`: Local optimization algorithm. Must be an algorithm from `NLopt.jl`.
- `lb::AbstractVector{<:Real}`: Lower bound (does not need to be changed if `param_index` and `param_eval` are used).
- `ub::AbstractVector{<:Real}`: Upper bound (does not need to be changed if `param_index` and `param_eval` are used).

# Keywords 
- `incidence_obs::AbstractVector{<:Integer}=Int64[]`: Indices of the state variables of the DEs to find incidence data of. The state variables must be cumulative data.  
- `solver_diff_opts::Dict=Dict()`: Keyword arguments to be passed into the DE solver. See `DifferentialEquations.jl`'s Common Solver Options.
- `opti_prob_opts::Dict=Dict()`: Keyword arguments to be passed into the optimization problem. See `Optimization.jl`'s Defining OptimizationProblems.
- `opti_solver_opts::Dict=Dict()`: Keyword arguments to be passed into the optimization solver. See `Optimization.jl`'s Common Solver Options.
- `print_status::Bool=false`: Determine whether the original output of the optimization algorithm is printed or not. 
- `pl_const::Real=0.0`: Constant that is added to loss of the computed points. The constant can be computed by `likelihood_const`.

# Returns
- `theta`: Values of the fixed parameter that is explored.
- `sol`: Value of `likelihood` computed at the fixed parameter where the other parameters are optimized.
"""
function find_profile_likelihood_multistart(step_size::Real, max_steps::Integer, param_index::Integer, 
                                            param_fitted::AbstractVector{<:Real}, 
                                            data::AbstractVector{<:AbstractVector{<:Real}}, 
                                            sol_obs::AbstractVector{<:Integer}, 
                                            threshold::Real, loss::Real, 
                                            prob::SciMLBase.AbstractDEProblem, 
                                            alg_diff::SciMLBase.AbstractDEAlgorithm,  
                                            times::AbstractVector{<:Real}, 
                                            obj_arr::AbstractVector{<:Function}, 
                                            alg_opti, alg_opti_local, 
                                            lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}; 
                                            incidence_obs::AbstractVector{<:Integer}=Int64[], 
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