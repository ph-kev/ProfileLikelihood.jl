# Packages 
using ProfileLikelihood, DifferentialEquations, Plots, Distributions, LaTeXStrings, Measures

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

# solver algorithm, tolerances
solver_opts = Dict(
    :alg => Tsit5(),
    :reltol => 1e-5,
    :abstol => 1e-10,
)

# times
times = LinRange{Float64}(0.0, 30.0, 31)

# Generate data 
perfectDataHost, noisyDataHost = generate_data(5, 366, i -> truncated(Poisson(i), lower=-eps(Float64)), prob, Tsit5(), times; incidence_obs_status=true, abstol=1e-10, reltol=1e-5)

perfectDataVector, noisyDataVector = generate_data(6, 366, i -> truncated(Poisson(i), lower=-eps(Float64)), prob, Tsit5(), times; incidence_obs_status=true, abstol=1e-10, reltol=1e-5)

# Objective function 
obj = (data, sol) -> poisson_error(data, sol)

solver_diff_opts = Dict(
    :reltol => 1e-5,
    :abstol => 1e-10
)

opti_prob_opts = Dict(
)

opti_solver_opts = Dict(
)

# Find optimal parameters 
loss, paramsFitted = estimate_params([1.0, 1.0, 1.0], [noisyDataHost, noisyDataVector], Int64[], prob, Tsit5(), times, [obj, obj], NOMADOpt(), [eps(Float64), eps(Float64), eps(Float64)], [2.0, 2.0, 2.0]; incidence_obs=[5, 6], solver_diff_opts=solver_diff_opts, opti_prob_opts=opti_prob_opts, opti_solver_opts=opti_solver_opts)
println("The minimum loss is $loss.")
println("The fitted parameters are $paramsFitted.")

# Constants to add back 
pl_const_1 = likelihood_const("poissonError"; data=noisyDataHost)
pl_const_2 = likelihood_const("poissonError"; data=noisyDataVector)
pl_const = pl_const_1 + pl_const_2
println("The profile likelihood constant is ", pl_const)

# Threshold 
threshold_simu = find_threshold(0.95, 3, loss)
threshold_poin = find_threshold(0.95, 1, loss)

# beta_h
theta1, sol1 = find_profile_likelihood(8.333e-7, 60, 1, paramsFitted, [noisyDataHost, noisyDataVector], Int64[], threshold_simu + 3, loss, prob, Tsit5(), times, [obj, obj], NOMADOpt(), [eps(Float64), eps(Float64), eps(Float64)], [2.0, 2.0, 2.0]; incidence_obs=[5, 6], solver_diff_opts=solver_diff_opts, opti_prob_opts=opti_prob_opts, opti_solver_opts=opti_solver_opts, print_status=false, pl_const=pl_const)
PLbeta_h = plot(theta1, [sol1, (x) -> (threshold_simu + pl_const), (x) -> (threshold_poin + pl_const)], xlabel=L"\beta_h", ylabel=L"\chi^2_{\rm PL}", yformatter=:plain, legend=:topright, labels=[L"\chi^2_{\rm PL}" "Simultaneous Threshold" "Pointwise Threshold"], right_margin=5mm, dpi=400)
scatter!([paramsFitted[1]], [loss + pl_const], color="orange", labels="Fitted Parameter")