"""
    estimate_params(p0::AbstractVector{<:Real}, 
                    data::AbstractVector{<:AbstractVector{<:Real}}, 
                    sol_obs::AbstractVector{<:Integer}, 
                    prob::SciMLBase.AbstractDEProblem, 
                    alg_diff::SciMLBase.AbstractDEAlgorithm, 
                    times::AbstractVector{<:Real}, obj_arr::AbstractVector{<:Function}, 
                    alg_opti, 
                    lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}; 
                    incidence_obs::AbstractVector{<:Int}=[], param_index::Int=0, 
                    param_eval::Real=0.0, 
                    solver_diff_opts::Dict=Dict(), opti_prob_opts::Dict=Dict(), 
                    opti_solver_opts::Dict=Dict(), print_status::Bool=false) 

This estimate the parameters of the system of differential equations using the objective functions in `obj_arr` given `prob` using the optimization algorithm `alg_opti`. 

Estimation of parameters is done using the objective function array `obj_arr` where each objective function in the array corresponds to data described by `sol_obs` and `incidence_obs`. For instance, if `obj_arr = [obj1 obj2]`,`sol_obs = [2]`, `incidence_obs = [3]`, then `obj1` use data corresponding to the second state variable of the DEs and `obj2` use incidence data of the third state variable of the DEs. Noisy data or any other type of data should be passed to `data` and in the same order as in the arrays `sol_obs` and `incidence_obs`. Fixing a parameter can be done using `param_index`. Instead, this return the fitted parameters with `param_eval` for the fixed parameter and accordingly, the loss is the objective function evaluated at the fitted parameters with `param_eval` for the fixed parameter. 

# Arguments
- `p0::AbstractVector{<:Real}`: Starting guess for optimization. If `param_index` and `param_eval` are used, then `p0` should not contain the fixed parameter.
- `data::AbstractVector{<:AbstractVector{<:Real}}`: Vector of data used for optimizing parameters. `data` must be in the same order as in `sol_obs` and `incidence_obs`.
- `sol_obs::AbstractVector{<:Integer}`: Indices of the state variables of the DEs to be used for sampling data points.
- `prob::SciMLBase.AbstractDEProblem`: DE Problem (see `DifferentialEquations.jl` for more information).
- `alg_diff::SciMLBase.AbstractDEAlgorithm`: DE Solver Algorithm (see `DifferentialEquations.jl` for more information).
- `times::AbstractVector{<:Real}`: Times that the data points will be sampled at.
- `obj_arr::AbstractVector{Function}`: Vector of objective functions. 
- `alg_opti`: Optimization algorithm (see `Optimization.jl` for a list of algorithms that could be used).
- `lb::AbstractVector{<:Real}`: Lower bound (does not need to be changed if `param_index` and `param_eval` are used).
- `ub::AbstractVector{<:Real}`: Upper bound (does not need to be changed if `param_index` and `param_eval` are used).

# Keywords
- `incidence_obs::AbstractVector{<:Int}=[]`: Indices of the state variables of the DEs to find incidence data of. The state variables must be cumulative data.  
- `param_index::Int=0`: Index of the parameter vector that is fixed. 
- `param_eval::Real=0.0`: Value for the fixed parameter. 
- `solver_diff_opts::Dict=Dict()`: Keyword arguments to be passed into the DE solver. See `DifferentialEquations.jl`'s Common Solver Options.
- `opti_prob_opts::Dict=Dict()`: Keyword arguments to be passed into the optimization problem. See `Optimization.jl`'s Defining OptimizationProblems.
- `opti_solver_opts::Dict=Dict()`: Keyword arguments to be passed into the optimization solver. See `Optimization.jl`'s Common Solver Options.
- `print_status::Bool=false`: Determine whether the original output of the optimization algorithm is printed or not. 

# Returns
- `minimum`: Loss of the fitted parameters according to the objective function(s).
- `minimizer`: Fitted parameters.
"""
function estimate_params(p0::AbstractVector{<:Real}, 
                         data::AbstractVector{<:AbstractVector{<:Real}}, 
                         sol_obs::AbstractVector{<:Integer}, 
                         prob::SciMLBase.AbstractDEProblem, 
                         alg_diff::SciMLBase.AbstractDEAlgorithm, 
                         times::AbstractVector{<:Real}, obj_arr::AbstractVector{<:Function}, 
                         alg_opti, 
                         lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}; 
                         incidence_obs::AbstractVector{<:Int} = Int64[], param_index::Int=0, 
                         param_eval::Real=0.0, 
                         solver_diff_opts::Dict = Dict(), opti_prob_opts::Dict = Dict(), 
                         opti_solver_opts::Dict = Dict(), print_status::Bool = false)
    if param_index == 0 # check if the parameter is fixed or not 
        g = (params, params_func) -> likelihood(params, data, sol_obs, prob, alg_diff, times,
        obj_arr; incidence_obs = incidence_obs, param_index=param_index, param_eval=param_eval,
        solver_diff_opts = solver_diff_opts)
        prob_opt = OptimizationProblem(g, p0; lb = lb, ub = ub, opti_prob_opts...)
        sol = Optimization.solve(prob_opt, alg_opti; opti_solver_opts...)
    else
        g = (params, params_func) -> likelihood(params, data, sol_obs, prob, alg_diff, times,
        obj_arr; incidence_obs = incidence_obs, param_index=param_index, param_eval=param_eval,
        solver_diff_opts = solver_diff_opts)
        # adjust bounds for optimization problem 
        lb_copy = copy(lb)
        ub_copy = copy(ub)
        deleteat!(lb_copy, param_index)
        deleteat!(ub_copy, param_index)
        prob_opt = OptimizationProblem(g, p0; lb = lb_copy, ub = ub_copy, opti_prob_opts...)
        sol = Optimization.solve(prob_opt, alg_opti; opti_solver_opts...)
    end
    if print_status == true
        println(sol.original)
    else 
    end    
    return sol.minimum, sol.minimizer
end


"""
    estimate_params_multistart(p0::AbstractVector{<:Real}, 
                               data::AbstractVector{<:AbstractVector{<:Real}}, 
                               sol_obs::AbstractVector{<:Integer}, 
                               prob::SciMLBase.AbstractDEProblem, 
                               alg_diff::SciMLBase.AbstractDEAlgorithm, 
                               times::AbstractVector{<:Real}, 
                               obj_arr::AbstractVector{<:Function}, 
                               alg_opti, 
                               alg_opti_local,
                               lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}; 
                               incidence_obs::AbstractVector{<:Int}=[], param_index::Int=0, 
                               param_eval::Real=0.0, 
                               solver_diff_opts::Dict=Dict(), opti_prob_opts::Dict=Dict(), 
                               opti_solver_opts::Dict=Dict(), print_status::Bool=false) 

This estimate the parameters of the system of differential equations using the objective function `obj_arr` given `prob` using the multi-start optimization algorithm `alg_opti` from `MultiStartOptimization` and the local optimization algorithm `alg_opti_local` from the `NLopt.jl`. 

The algorithm `alg_opti` is always `MultistartOptimization.TikTak(n)` where `n` is
the number of starting points generated from the Sobol sequence. `MultistartOptimization.jl` only support optimization algorithms from `NLopt.jl`. The recommended local optimization algorithm is `NLopt.LN_NELDERMEAD()`. See the docstring for `estimate_params` for more information for what `estimate_params_multistart` do. 

# Arguments
- `p0::AbstractVector{<:Real}`: Starting guess for optimization. If `param_index` and `param_eval` are used, then `p0` should not contain the fixed parameter.
- `data::AbstractVector{<:AbstractVector{<:Real}}`: Vector of data used for optimizing parameters. `data` must be in the same order as in `sol_obs` and `incidence_obs`.
- `sol_obs::AbstractVector{<:Integer}`: Indices of the state variables of the DEs to be used for sampling data points.
- `prob::SciMLBase.AbstractDEProblem`: DE Problem (see `DifferentialEquations.jl` for more information).
- `alg_diff::SciMLBase.AbstractDEAlgorithm`: DE Solver Algorithm (see `DifferentialEquations.jl` for more information).
- `times::AbstractVector{<:Real}`: Times that the data points will be sampled at.
- `obj_arr::AbstractVector`: Vector of objective functions. 
- `alg_opti`: Global optimization algorithm. Typically, `MultistartOptimization.TikTak(n)` where `n` is the number of starting points generated from the Sobol sequence.
- `alg_opti_local`: Local optimization algorithm. Must be an algorithm from `NLopt.jl`.
- `lb::AbstractVector{<:Real}`: Lower bound (does not need to be changed if `param_index` and `param_eval` are used).
- `ub::AbstractVector{<:Real}`: Upper bound (does not need to be changed if `param_index` and `param_eval` are used).

# Keywords
- `incidence_obs::AbstractVector{<:Int}=[]`: Indices of the state variables of the DEs to find incidence data of which is used for sampling data points. The state variables must be cumulative data.  
- `param_index::Int=0`: Index of the parameter vector that is fixed. 
- `param_eval::Real=0.0`: Value for the fixed parameter. 
- `solver_diff_opts::Dict=Dict()`: Keyword arguments to be passed into the DE solver. See `DifferentialEquations.jl`'s Common Solver Options.
- `opti_prob_opts::Dict=Dict()`: Keyword arguments to be passed into the optimization problem. See `Optimization.jl`'s Defining OptimizationProblems.
- `opti_solver_opts::Dict=Dict()`: Keyword arguments to be passed into the optimization solver. See `Optimization.jl`'s Common Solver Options.
- `print_status::Bool=false`: Determine whether the original output of the optimization algorithm is printed or not. 

# Returns
- `minimum`: Loss of the fitted parameters according to the objective function(s).
- `minimizer`: Fitted parameters.
"""
function estimate_params_multistart(p0::AbstractVector{<:Real}, 
                                    data::AbstractVector{<:AbstractVector{<:Real}}, 
                                    sol_obs::AbstractVector{<:Integer}, 
                                    prob::SciMLBase.AbstractDEProblem, 
                                    alg_diff::SciMLBase.AbstractDEAlgorithm, 
                                    times::AbstractVector{<:Real}, 
                                    obj_arr::AbstractVector{<:Function}, 
                                    alg_opti, 
                                    alg_opti_local, 
                                    lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}; 
                                    incidence_obs::AbstractVector{<:Int} = [], 
                                    param_index::Int=0,
                                    param_eval::Real=0.0, 
                                    solver_diff_opts::Dict = Dict(), 
                                    opti_prob_opts::Dict = Dict(), 
                                    opti_solver_opts = Dict(), print_status::Bool = false) 
    if param_index == 0 # check if the parameter is fixed or not 
        g = (params, params_func) -> likelihood(params, data, sol_obs, prob, alg_diff, times,
        obj_arr; incidence_obs = incidence_obs, param_index=param_index, param_eval=param_eval,
        solver_diff_opts = solver_diff_opts)
        prob_opt = OptimizationProblem(g, p0; lb = lb, ub = ub, opti_prob_opts...)
        sol = Optimization.solve(prob_opt, alg_opti, alg_opti_local; opti_solver_opts...)
    else
        g = (params, params_func) -> likelihood(params, data, sol_obs, prob, alg_diff, times,
        obj_arr; incidence_obs = incidence_obs, param_index=param_index, param_eval=param_eval,
        solver_diff_opts = solver_diff_opts)
        # adjust bounds for optimization problem 
        lb_copy = copy(lb)
        ub_copy = copy(ub)
        deleteat!(lb_copy, param_index)
        deleteat!(ub_copy, param_index)
        prob_opt = OptimizationProblem(g, p0; lb = lb_copy, ub = ub_copy, opti_prob_opts...)
        sol = Optimization.solve(prob_opt, alg_opti, alg_opti_local; opti_solver_opts...)
    end
    if print_status == true
        println(sol.original)
    else 
    end    
    return sol.minimum, sol.minimizer
end