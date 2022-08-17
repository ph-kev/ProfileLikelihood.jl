# Packages 
using ProfileLikelihood, DifferentialEquations, Plots, Distributions, Interpolations, LaTeXStrings, Measures, JLD2

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
    end
end

# Timespan, true parameters and initial conditions for simulating data
tspan = (0.0, 40.0)
p0 = [0.0001, 0.001, 0.09] 
u0 = [1.0, 1.0, 1000.0, 5000.0, 1.0]

prob = ODEProblem(sis!, u0, tspan, p0);

# times
times = LinRange{Float64}(0.0, 30.0, 31)

# Generate data 
perfect_data, noisy_data = generate_data(5, 366, i -> truncated(Poisson(i), lower = -eps(Float64)), prob, Tsit5(), times; incidence_obs_status=true, abstol=1e-10, reltol=1e-5)

# Plot solution 
plt = plot(times, perfect_data, dpi = 400, legend = :topleft, labels = "Perfect Incidence Data", xlabel = L"t")
plot!(times, noisy_data, labels = "Noisy Incidence Data")
display(plt)

# Settings for solving differential equations and optimization 
solver_diff_opts = Dict(
    :reltol => 1e-5,
    :abstol => 1e-10,
)

# Objective function 
obj = (data, sol) -> const_variance_error(data, sol, 5.7)

# True loss value 
trueLoss = likelihood(p0, [noisy_data], Int64[], prob, Tsit5(), times, [obj]; incidence_obs = [5], solver_diff_opts = solver_diff_opts)
println("The loss with the true parameters is $(trueLoss).")

# Initial guess 
p1 = [1., 1., 1.]

# Options for optimization solver 
opti_solver_opts = Dict(
    :time_limit => 200,
    :f_calls_limit => 100000000,
    :iterations => 2000000
)

# Find optimal parameters 
loss, fitted_params = estimate_params(p1, [noisy_data], Int64[], prob, Tsit5(), times, [obj], OptimizationMetaheuristics.DE(N=300), [eps(Float64), eps(Float64), eps(Float64)], [2.0, 2.0, 2.0]; incidence_obs = [5], solver_diff_opts=solver_diff_opts, opti_solver_opts = opti_solver_opts, print_status = true)
println("The minimum loss is $loss.")
println("The fitted parameters are $fitted_params.")

# Plot fitted parameters 
probCur = remake(prob, p=fitted_params)
sol1 = generate_incidence_data(5, probCur, Tsit5(), times, abstol=1e-10, reltol=1e-5)
plt1 = plot(times, sol1, legend=:topleft, labels = "Fitted Parameters", xlabel = L"t", dpi = 400)
scatter!(times, noisy_data, labels = "Noisy Incidence Data")
display(plt1)
savefig(plt1, "noisyIncidenceDataGraph")

# Find threshold 
threshold_simu = find_threshold(0.95, 3, loss)
threshold_poin = find_threshold(0.95, 1, loss)
println(threshold_simu)

# Constants to add back 
pl_const = likelihood_const("const_variance_error"; times = times, sigma = 5.7)
println("The profile likelihood constant is $(pl_const).")

# Finding profile likelihood 
# beta_h
theta1, sol1 = find_profile_likelihood(8.4e-7, 55, 1, fitted_params, [noisy_data], Int64[], threshold_simu + 3, loss, prob, Tsit5(), times, [obj], NOMADOpt(), [eps(Float64), eps(Float64), eps(Float64)], [4.0, 4.0, 10.0]; incidence_obs = [5], solver_diff_opts=solver_diff_opts, pl_const = pl_const, print_status = true)
PLbeta_h = plot(theta1, [sol1, (x) -> (threshold_simu + pl_const), (x) -> (threshold_poin + pl_const)], xlabel = L"\beta_h", ylabel = L"\chi^2_{\rm PL}", yformatter = :plain, legend=:topright, labels = [L"\chi^2_{\rm PL}" "Simultaneous Threshold" "Pointwise Threshold"], right_margin=5mm, dpi = 400)
scatter!([fitted_params[1]], [loss + pl_const], color = "orange", labels = "Fitted Parameter")
display(PLbeta_h)
savefig(PLbeta_h, "PLbeta_h.png")

# beta_v 
theta2, sol2 = find_profile_likelihood(1.27e-4, 95, 2, fitted_params, [noisy_data], Int64[], threshold_simu + 3, loss, prob, Tsit5(), times, [obj], NOMADOpt(), [eps(Float64), eps(Float64), eps(Float64)], [4.0, 4.0, 100.0]; incidence_obs = [5], solver_diff_opts=solver_diff_opts, pl_const = pl_const, print_status = false)
PLbeta_v = plot(theta2, [sol2, (x) -> (threshold_simu + pl_const), (x) -> (threshold_poin + pl_const)], xlabel = L"\beta_v", ylabel = L"\chi^2_{\rm PL}", yformatter = :plain, legend=:topright, labels = [L"\chi^2_{\rm PL}" "Simultaneous Threshold" "Pointwise Threshold"], right_margin=5mm, dpi = 400)
scatter!([fitted_params[2]], [loss + pl_const], color = "orange", labels = "Fitted Parameter")
display(PLbeta_v)
savefig(PLbeta_v, "PLbeta_v.png")

# gamma
theta3, sol3 = find_profile_likelihood(0.04, 100, 3, fitted_params, [noisy_data], Int64[], threshold_simu + 3, loss, prob, Tsit5(), times, [obj], NOMADOpt(), [eps(Float64), eps(Float64), eps(Float64)], [4.0, 4.0, 1000.0]; incidence_obs = [5], solver_diff_opts=solver_diff_opts, pl_const = pl_const)
PLgamma = plot(theta3, [sol3, (x) -> (threshold_simu + pl_const), (x) -> (threshold_poin + pl_const)], xlabel = L"\gamma", ylabel = L"\chi^2_{\rm PL}", yformatter = :plain, legend=:topright, labels = [L"\chi^2_{\rm PL}" "Simultaneous Threshold" "Pointwise Threshold"], right_margin=5mm, dpi = 400)
scatter!([fitted_params[3]], [loss + pl_const], color = "orange", labels = "Fitted Parameter")
display(PLgamma)
savefig(PLgamma, "PLgamma.png")

# Interpolations 
beta_h_inter = linear_interpolation(theta1, sol1)
beta_v_inter = linear_interpolation(theta2, sol2)
gamma_inter = linear_interpolation(theta3, sol3)

ci_simu_beta_h = find_roots(beta_h_inter, theta1, threshold_simu + pl_const)
ci_simu_beta_v = find_roots(beta_v_inter, theta2, threshold_simu + pl_const)
ci_simu_gamma = find_roots(gamma_inter, theta3, threshold_simu + pl_const)

ci_poin_beta_h = find_roots(beta_h_inter, theta1, threshold_poin + pl_const)
ci_poin_beta_v = find_roots(beta_v_inter, theta2, threshold_poin + pl_const)
ci_poin_gamma = find_roots(gamma_inter, theta3, threshold_poin + pl_const)

jldsave("sisIncidencePoissonMisfitConstantError0_30.jld2"; loss, fitted_params, threshold_simu, threshold_poin, pl_const, theta1, sol1, theta2, sol2, theta3, sol3, ci_simu_beta_h, ci_simu_beta_v, ci_simu_gamma, ci_poin_beta_h, ci_poin_beta_v, ci_poin_gamma)