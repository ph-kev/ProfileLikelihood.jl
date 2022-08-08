function estimate_params(p0, data, sol_obs, prob, alg_diff, times, obj_arr, alg_opti, lb, ub; incidence_obs = [], param_index=0, param_eval=0, solver_diff_opts = Dict(), opti_prob_opts = Dict(), opti_solver_opts = Dict(), print_status = false)
    if param_index == 0
        g = (params, params_func) -> likelihood(params, data, sol_obs, prob, alg_diff, times, obj_arr; incidence_obs = incidence_obs, param_index=param_index, param_eval=param_eval, solver_diff_opts = solver_diff_opts)
        prob_opt = OptimizationProblem(g, p0; lb = lb, ub = ub, opti_prob_opts...)
        sol = Optimization.solve(prob_opt, alg_opti; opti_solver_opts...)
    else
        g = (params, params_func) -> likelihood(params, data, sol_obs, prob, alg_diff, times, obj_arr; incidence_obs = incidence_obs, param_index=param_index, param_eval=param_eval, solver_diff_opts = solver_diff_opts)
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