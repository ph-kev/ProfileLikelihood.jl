function estimate_params(p0::AbstractVector{<:Real}, 
                         data::AbstractVector{<:AbstractVector{<:Real}}, 
                         sol_obs::AbstractVector{Any}, 
                         prob::SciMLBase.AbstractDEProblem, 
                         alg_diff::SciMLBase.AbstractDEAlgorithm, 
                         times::AbstractVector{<:Real}, obj_arr::AbstractVector, 
                         alg_opti, 
                         lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}; 
                         incidence_obs::AbstractVector{<:Int} = [], param_index::Int=0, 
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

function estimate_params_multistart(p0::AbstractVector{<:Real}, 
                                    data::AbstractVector{<:AbstractVector{<:Real}}, 
                                    sol_obs::AbstractVector{Any}, 
                                    prob::SciMLBase.AbstractDEProblem, 
                                    alg_diff::SciMLBase.AbstractDEAlgorithm, 
                                    times::AbstractVector{<:Real}, 
                                    obj_arr::AbstractVector, 
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