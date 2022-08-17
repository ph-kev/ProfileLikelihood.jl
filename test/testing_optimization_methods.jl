# Packages 
using ProfileLikelihood, DifferentialEquations, Distributions


# Define system of ODEs
# Constants 
const Pi_h = 0.004
const Pi_v = 90

# Define SIS model
function sis!(du, u, p, t)
    let (I_h, I_v, S_h, S_v, C_h, beta_h, beta_v, gamma, mu_v, mu_h) = (u[1], u[2], u[3], u[4], u[5], p[1], p[2], p[3], p[4], p[5])
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
p0 = [0.0001, 0.001, 0.09, 0.09, 0.00004]
u0 = [1.0, 1.0, 1000.0, 5000.0, 1.0, 1.0]

prob = ODEProblem(sis!, u0, tspan, p0);

# times
times = LinRange{Float64}(0.0, 30.0, 31)

# Generate data 
perfectDataHost, noisyDataHost = generate_data(5, 366, i -> truncated(Poisson(i), lower=-eps(Float64)), prob, Tsit5(), times; incidence_obs_status=true, abstol=1e-10, reltol=1e-5)

perfectDataVector, noisyDataVector = generate_data(6, 366, i -> truncated(Poisson(i), lower=-eps(Float64)), prob, Tsit5(), times; incidence_obs_status=true, abstol=1e-10, reltol=1e-5)

# Objective function 
obj = (data, sol) -> poisson_error(data, sol)

# Options for solving differential equations 
solver_diff_opts = Dict(
    :reltol => 1e-5,
    :abstol => 1e-10
)

# Best loss found: ≈ -36049.832901472066

# Metaheuristics: Differential Evolution Algorithm 
opti_solver_opts = Dict(
    :time_limit => 200,
    :f_calls_limit => 100000000,
    :iterations => 2000000
)

# Optimization Benchmarks for a variety of algorithms which is not at all comprehensive 
println("Metaheuristics")
loss1, paramsFitted1 = @time estimate_params([12.0, 12.0, 12.0, 12.0, 12.0], [noisyDataHost, noisyDataVector], Int64[], prob, AutoVern7(Rodas4()), times, [obj, obj], OptimizationMetaheuristics.DE(), [eps(Float64), eps(Float64), eps(Float64), eps(Float64), eps(Float64)], [25.0, 25.0, 25.0, 25.0, 25.0]; incidence_obs=[5, 6], solver_diff_opts=solver_diff_opts, opti_solver_opts=opti_solver_opts)

# NOMAD 
println("NOMAD")
loss2, paramsFitted2 = @time estimate_params([12.0, 12.0, 12.0, 12.0, 12.0], [noisyDataHost, noisyDataVector], Int64[], prob, AutoVern7(Rodas4()), times, [obj, obj], NOMADOpt(), [eps(Float64), eps(Float64), eps(Float64), eps(Float64), eps(Float64)], [25.0, 25.0, 25.0, 25.0, 25.0]; incidence_obs=[5, 6], solver_diff_opts=solver_diff_opts)

# NOMAD (smaller bounds)
println("NOMAD (smaller bounds)")
loss, paramsFitted = @time estimate_params([2.0, 2.0, 2.0, 2.0, 2.0], [noisyDataHost, noisyDataVector], Int64[], prob, AutoVern7(Rodas4()), times, [obj, obj], NOMADOpt(), [eps(Float64), eps(Float64), eps(Float64), eps(Float64), eps(Float64)], [5.0, 5.0, 5.0, 5.0, 5.0]; incidence_obs=[5, 6], solver_diff_opts=solver_diff_opts)

# Evolutionary 
println("Evolutionary")
loss3, paramsFitted3 = @time estimate_params([12.0, 12.0, 12.0, 12.0, 12.0], [noisyDataHost, noisyDataVector], Int64[], prob, AutoVern7(Rodas4()), times, [obj, obj], Evolutionary.CMAES(), [eps(Float64), eps(Float64), eps(Float64), eps(Float64), eps(Float64)], [25.0, 25.0, 25.0, 25.0, 25.0]; incidence_obs=[5, 6])

# Evolutionary (smaller bounds)
println("Evolutionary (smaller bounds)")
loss4, paramsFitted4 = @time estimate_params([2.0, 2.0, 2.0, 2.0, 2.0], [noisyDataHost, noisyDataVector], Int64[], prob, AutoVern7(Rodas4()), times, [obj, obj], Evolutionary.CMAES(), [eps(Float64), eps(Float64), eps(Float64), eps(Float64), eps(Float64)], [3.0, 3.0, 3.0, 3.0, 3.0]; incidence_obs=[5, 6])

# MultiStartOptimization and NLOpt
println("MultiStartOptimization and NLOpt (LN_NELDERMEAD)")
loss5, paramsFitted5 = @time estimate_params_multistart([12.0, 12.0, 12.0, 12.0, 12.0], [noisyDataHost, noisyDataVector], Int64[], prob, AutoVern7(Rodas4()), times, [obj, obj], MultistartOptimization.TikTak(100), NLopt.LN_NELDERMEAD(), [eps(Float64), eps(Float64), eps(Float64), eps(Float64), eps(Float64)], [25.0, 25.0, 25.0, 25.0, 25.0]; incidence_obs=[5, 6])

# MultiStartOptimization and NLOpt
println("MultiStartOptimization and NLOpt (LN_SBPLX)")
loss6, paramsFitted6 = @time estimate_params_multistart([12.0, 12.0, 12.0, 12.0, 12.0], [noisyDataHost, noisyDataVector], Int64[], prob, AutoVern7(Rodas4()), times, [obj, obj], MultistartOptimization.TikTak(100), NLopt.LN_SBPLX(), [eps(Float64), eps(Float64), eps(Float64), eps(Float64), eps(Float64)], [25.0, 25.0, 25.0, 25.0, 25.0]; incidence_obs=[5, 6])

# BBO
opti_solver_opts = Dict(
    :maxtime => 500,
    :TraceInterval => 50.0
)
println("BBO")
loss7, paramsFitted7 = @time estimate_params([12.0, 12.0, 12.0, 12.0, 12.0], [noisyDataHost, noisyDataVector], Int64[], prob, AutoVern7(Rodas4()), times, [obj, obj], BBO_generating_set_search(), [eps(Float64), eps(Float64), eps(Float64), eps(Float64), eps(Float64)], [25.0, 25.0, 25.0, 25.0, 25.0]; incidence_obs=[5, 6], opti_solver_opts = opti_solver_opts)

# Optim
println("Optim")
loss8, paramsFitted8 = @time estimate_params([12.0, 12.0, 12.0, 12.0, 12.0], [noisyDataHost, noisyDataVector], Int64[], prob, AutoVern7(Rodas4()), times, [obj, obj], Optim.ParticleSwarm(), [eps(Float64), eps(Float64), eps(Float64), eps(Float64), eps(Float64)], [25.0, 25.0, 25.0, 25.0, 25.0]; incidence_obs=[5, 6])

# Optim (smaller bounds)
println("Optim (smaller bounds)")
loss9, paramsFitted9 = @time estimate_params([2.0, 2.0, 2.0, 2.0, 2.0], [noisyDataHost, noisyDataVector], Int64[], prob, AutoVern7(Rodas4()), times, [obj, obj], Optim.ParticleSwarm(), [eps(Float64), eps(Float64), eps(Float64), eps(Float64), eps(Float64)], [5.0, 5.0, 5.0, 5.0, 5.0]; incidence_obs=[5, 6])

# Print stuff 
println("Metaheuristics (DE): The minimum loss is $loss1.")
println("Metaheuristics (DE): The fitted parameters are $paramsFitted1.")
println("NOMAD (NOMADOpt): The minimum loss is $loss2.")
println("NOMAD (NOMADOpt): The fitted parameters are $paramsFitted2.")
println("NOMAD (NOMADOpt) (smaller bounds): The minimum loss is $loss.")
println("NOMAD (NOMADOpt) (smaller bounds): The fitted parameters are $paramsFitted.")
println("Evolutionary (CMAES): The minimum loss is $loss3.")
println("Evolutionary (CMAES): The fitted parameters are $paramsFitted3.")
println("Evolutionary (CMAES) (smaller bounds): The minimum loss is $loss4.")
println("Evolutionary (CMAES) (smaller bounds): The fitted parameters are $paramsFitted4.")
println("MultiStartOptimization (LN_NELDERMEAD): The minimum loss is $loss5.")
println("MultiStartOptimization (LN_NELDERMEAD): The fitted parameters are $paramsFitted5.")
println("MultiStartOptimization (LN_SBPLX): The minimum loss is $loss6.")
println("MultiStartOptimization (LN_SBPLX): The fitted parameters are $paramsFitted6.")
println("BBO (generating_set_search): The minimum loss is $loss7.")
println("BBO (generating_set_search): The fitted parameters are $paramsFitted7.")
println("Optim (ParticleSwarm): The minimum loss is $loss8.")
println("Optim (ParticleSwarm): The fitted parameters are $paramsFitted8.")
println("Optim (ParticleSwarm) (smaller bounds): The minimum loss is $loss9.")
println("Optim (ParticleSwarm) (smaller bounds): The fitted parameters are $paramsFitted9.")

