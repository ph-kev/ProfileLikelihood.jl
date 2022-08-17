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
        du[6] = beta_v * S_v * I_h
    end
end

# Timespan, true parameters and initial conditions for simulating data
tspan = (0.0, 40.0)
p0 = [0.0001, 0.001, 0.09] 
u0 = [1.0, 1.0, 1000.0, 5000.0, 1.0, 1.0]

prob = ODEProblem(sis!, u0, tspan, p0);

# times
times = LinRange{Float64}(0.0, 15.0, 16)

# Generate data 
perfect_data_host, noisy_data_host = generate_data(5, 366, i -> truncated(Poisson(i), lower = -eps(Float64)), prob, Tsit5(), times; incidence_obs_status=true, abstol=1e-10, reltol=1e-5)
perfect_data_vector, noisy_data_vector = generate_data(6, 366, i -> truncated(Poisson(i), lower = -eps(Float64)), prob, Tsit5(), times; incidence_obs_status=true, abstol=1e-10, reltol=1e-5)

# Plot solution 
plt = plot(times, perfect_data_host, dpi = 400, legend = :topleft, labels = "Perfect Incidence Host Data", xlabel = L"t")
plot!(times, noisy_data_host, labels = "Noisy Incidence Host Data")
display(plt)
plt1 = plot(times, perfect_data_vector, dpi = 400, legend = :topleft, labels = "Perfect Incidence Vector Data", xlabel = L"t")
plot!(times, noisy_data_vector, labels = "Noisy Incidence Vector Data")
display(plt1)


# Settings for solving differential equations and optimization 
solver_diff_opts = Dict(
    :reltol => 1e-5,
    :abstol => 1e-10,
)

# Objective function 
obj = (data, sol) -> poisson_error(data, sol)

# True loss value 
trueLoss = likelihood(p0, [noisy_data_host, noisy_data_vector], Int64[], prob, Tsit5(), times, [obj, obj]; incidence_obs = [5, 6], solver_diff_opts = solver_diff_opts)
println("The loss with the true parameters is $(trueLoss).")

# Initial guess 
p1 = [2., 2., 2.]

# Find optimal parameters 
loss, fitted_params = estimate_params(p1, [noisy_data_host, noisy_data_vector], Int64[], prob, Tsit5(), times, [obj, obj], NOMADOpt(), [eps(Float64), eps(Float64), eps(Float64)], [2.0, 2.0, 2.0]; incidence_obs = [5, 6], solver_diff_opts=solver_diff_opts)
println("The minimum loss is $loss.")
println("The fitted parameters are $fitted_params.")

# Plot fitted parameters 
probCur = remake(prob, p=fitted_params)
sol1 = generate_incidence_data(5, probCur, Tsit5(), times, abstol=1e-10, reltol=1e-5)
plt1 = plot(times, sol1, legend=:topleft, labels = "Fitted Parameters", xlabel = L"t", dpi = 400)
scatter!(times, noisy_data_host, labels = "Noisy Incidence Host Data")
display(plt1)
savefig(plt1, "noisyIncidenceHostDataGraph")

sol2 = generate_incidence_data(6, probCur, Tsit5(), times, abstol=1e-10, reltol=1e-5)
plt2 = plot(times, sol2, legend=:topleft, labels = "Fitted Parameters", xlabel = L"t", dpi = 400)
scatter!(times, noisy_data_vector, labels = "Noisy Incidence Vector Data")
display(plt2)
savefig(plt2, "noisyIncidenceVectorDataGraph")

# Find threshold 
threshold_simu = find_threshold(0.95, 3, loss)
threshold_poin = find_threshold(0.95, 1, loss)
println(threshold_simu)

# Constants to add back 
pl_const_host = likelihood_const("poisson_error"; data = noisy_data_host)
pl_const_vector = likelihood_const("poisson_error"; data = noisy_data_vector)
pl_const = pl_const_host + pl_const_vector 
println("The profile likelihood constant is $(pl_const).")

# Finding profile likelihood 
# beta_h
theta1, sol1 = find_profile_likelihood(9.999e-7, 600, 1, fitted_params, [noisy_data_host, noisy_data_vector], Int64[], threshold_simu + 3, loss, prob, Tsit5(), times, [obj, obj], NOMADOpt(), [eps(Float64), eps(Float64), eps(Float64)], [4.0, 4.0, 4.0]; incidence_obs = [5, 6], solver_diff_opts=solver_diff_opts, pl_const = pl_const, print_status = true)
PLbeta_h = plot(theta1, [sol1, (x) -> (threshold_simu + pl_const), (x) -> (threshold_poin + pl_const)], xlabel = L"\beta_h", ylabel = L"\chi^2_{\rm PL}", yformatter = :plain, legend=:topright, labels = [L"\chi^2_{\rm PL}" "Simultaneous Threshold" "Pointwise Threshold"], right_margin=5mm, dpi = 400)
scatter!([fitted_params[1]], [loss + pl_const], color = "orange", labels = "Fitted Parameter")
display(PLbeta_h)
savefig(PLbeta_h, "PLbeta_h.png")

# beta_v 
theta2, sol2 = find_profile_likelihood(2e-5, 55, 2, fitted_params, [noisy_data_host, noisy_data_vector], Int64[], threshold_simu + 3, loss, prob, Tsit5(), times, [obj, obj], NOMADOpt(), [eps(Float64), eps(Float64), eps(Float64)], [4.0, 4.0, 4.0]; incidence_obs = [5, 6], solver_diff_opts=solver_diff_opts, pl_const = pl_const, print_status = false)
PLbeta_v = plot(theta2, [sol2, (x) -> (threshold_simu + pl_const), (x) -> (threshold_poin + pl_const)], xlabel = L"\beta_v", ylabel = L"\chi^2_{\rm PL}", yformatter = :plain, legend=:topright, labels = [L"\chi^2_{\rm PL}" "Simultaneous Threshold" "Pointwise Threshold"], right_margin=5mm, dpi = 400)
scatter!([fitted_params[2]], [loss + pl_const], color = "orange", labels = "Fitted Parameter")
display(PLbeta_v)
savefig(PLbeta_v, "PLbeta_v.png")

# gamma
theta3, sol3 = find_profile_likelihood(7.2e-3, 70, 3, fitted_params, [noisy_data_host, noisy_data_vector], Int64[], threshold_simu + 3, loss, prob, Tsit5(), times, [obj, obj], NOMADOpt(), [eps(Float64), eps(Float64), eps(Float64)], [4.0, 4.0, 4.0]; incidence_obs = [5, 6], solver_diff_opts=solver_diff_opts, pl_const = pl_const)
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

jldsave("sisIncidenceMoreObservationsPoisson0_15.jld2"; loss, fitted_params, threshold_simu, threshold_poin, pl_const, theta1, sol1, theta2, sol2, theta3, sol3, ci_simu_beta_h, ci_simu_beta_v, ci_simu_gamma, ci_poin_beta_h, ci_poin_beta_v, ci_poin_gamma)