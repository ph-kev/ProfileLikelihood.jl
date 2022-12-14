# Packages 
using ProfileLikelihood, DifferentialEquations, Distributions


# Define system of ODEs
# Constants 
const Pi_h = 0.004
const Pi_v = 90
const mu_h = 0.00004
const mu_v = 0.09

# Define SIS model
function sis!(du, u, p, t)
    let (I_h, I_v, S_h, S_v, C_h, beta_h, beta_v, gamma) = (u[1], u[2], u[3], u[4], u[5], p[1], p[2], p[3])
        du[1] = beta_h * S_h * I_v - (mu_h + gamma) * I_h
        du[2] = beta_v * S_v * I_h - mu_v * I_v
        du[3] = Pi_h - mu_h * S_h - beta_h * S_h * I_v + gamma * I_h
        du[4] = Pi_v - mu_v * S_v - beta_v * S_v * I_h
        du[5] = beta_h * S_h * I_v
        du[6] = beta_v * S_v * I_h
    end
end

# Timespan, true parameters and initial conditions for simulating data
tspan = (0.0, 40.0)
p0 = [0.0001, 0.001, 0.09]
u0 = [1.0, 1.0, 1000.0, 5000.0, 1.0, 1.0]

prob = ODEProblem(sis!, u0, tspan, p0);

# times
times = LinRange{Float64}(0.0, 30.0, 31)

# Generate data 
perfect_data_host, noisy_data_host = generate_data(5, 366, i -> truncated(Poisson(i), lower=-eps(Float64)), prob, Tsit5(), times; incidence_obs_status=true, abstol=1e-10, reltol=1e-5)

perfect_data_vector, noisy_data_vector = generate_data(6, 366, i -> truncated(Poisson(i), lower=-eps(Float64)), prob, Tsit5(), times; incidence_obs_status=true, abstol=1e-10, reltol=1e-5)

# Objective function 
obj = (data, sol) -> poisson_error(data, sol)

solver_diff_opts = Dict(
    :reltol => 1e-5,
    :abstol => 1e-10
)

opti_solver_opts = Dict(
    :time_limit => 240.0,
    :f_calls_limit => 10000000000,
    :iterations => 200000000
)

# Find optimal parameters 
loss, fitted_params = estimate_params([1.0, 1.0, 1.0], [noisy_data_host, noisy_data_vector], Int64[], prob, Tsit5(), times, [obj, obj], OptimizationMetaheuristics.DE(N=100), [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]; incidence_obs=[5, 6], solver_diff_opts=solver_diff_opts, opti_prob_opts=opti_prob_opts, opti_solver_opts=opti_solver_opts)
println("The minimum loss is $loss.")
println("The fitted parameters are $fitted_params.")

loss, fitted_params = estimate_params([1.0, 1.0, 1.0], [noisy_data_host, noisy_data_vector], Int64[], prob, Tsit5(), times, [obj, obj], OptimizationMetaheuristics.DE(N=30), [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]; incidence_obs=[5, 6], param_eval=1.5, param_index=1, solver_diff_opts=solver_diff_opts, opti_solver_opts=opti_solver_opts)
println("The minimum loss is $loss.")
println("The fitted parameters are $fitted_params.")

loss, fitted_params = estimate_params([1.0, 1.0], [noisy_data_host, noisy_data_vector], Int64[], prob, Tsit5(), times, [obj, obj], NOMADOpt(), [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]; incidence_obs=[5, 6], param_eval=1.5, param_index=1, solver_diff_opts=solver_diff_opts)
println("The minimum loss is $loss.")
println("The fitted parameters are $fitted_params.")

loss, fitted_params = estimate_params_multistart([1.0, 1.0], [noisy_data_host, noisy_data_vector], Int64[], prob, Tsit5(), times, [obj, obj], MultistartOptimization.TikTak(300), NLopt.LN_NELDERMEAD(), [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]; incidence_obs=[5, 6], param_eval=1.5, param_index=1, solver_diff_opts=solver_diff_opts)
println("The minimum loss is $loss.")
println("The fitted parameters are $fitted_params.")