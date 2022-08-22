# # Tutorial 

# This tutorial highlights how `ProfileLikelihood.jl` can be used to generate data, estimate parameters, and find the profile likelihood. Furthermore, profile likelihood can be used for identifiability analysis and confidence intervals estimation. It is recommend to read the documentation for more information.   

# ## Packages 

# We load some useful packages.

using ProfileLikelihood, DifferentialEquations, Distributions, # for finding PL
Plots, LaTeXStrings, Measures, # for plotting
Interpolations # for finding CI

# ## Setting up the problem 

# We will consider the simple ``SIR`` model. To set this up, we will use the `DifferentialEquations.jl` package.

function sir!(du, u, p, t)
    let (S, I, R, C, beta, gamma) = (u[1], u[2], u[3], u[4], p[1], p[2])
        du[1] = - beta * S * I
        du[2] = beta * S * I - gamma * I
        du[3] = gamma * I
        du[4] = beta * S * I
    end
end
nothing # hide

# Note that none of the parameters are known. If we want to make a parameter known, we can simply put it as a `const` global variable. For instance, if we want to make beta known as 0.01, we would add the line `const beta = 0.0001` before defining the function `sir!`. Also, the variable `C` is the cumulative data of the infected. 

# We now set up the time span, initial conditions, and true parameters for the differential equation solver.

tspan = (0.0, 40.0)
p0 = [0.0001, 0.1] 
u0 = [10000.0, 10.0, 0.0, 10.0]
prob = ODEProblem(sir!, u0, tspan, p0)
nothing # hide

# Let plot what the solutions to the ``SIR`` model look like using the `Plots.jl` package. 

sol = solve(prob, Tsit5(), reltol=1e-5, abstol=1e-10, dense=true)
sol_plot = plot(sol,plotdensity=1000, labels = ["Suspectible" "Infected" "Recovered" "Cumulative Infected"])

# ## Generating data 
# Let generate fake data to work with using the function `generate_data`. We will generate data of recovered and incidence data of infected. We will generate data at ``t=0``, ``t=1``, ..., and ``t=10``. The second input to `generate_data` is the seed of the random number generator in case if noise is added. The third input accepts a function whose input are the values of the data points and output the noisy data. For this tutorial, we will add noise from the normal distribution and Poisson distribution using the `Distributions.jl` package. We will use the anonymous functions `i -> truncated(Normal(i,30), lower = 0)` `i -> Poisson(i)` to add noise. Note that if we want incidence data from cumulative data, we need to set `incidence_obs_status=true`. Also, we can pass keyword arguments to the differential equation solver the same way we did when we solve the differential equation earlier.

times = LinRange{Float64}(0.0, 10.0, 11)
perfect_data_recovered, noisy_data_recovered = generate_data(3, 111, i -> truncated(Normal(i,30), lower=0), 
prob, Tsit5(), times; incidence_obs_status=false, abstol=1e-10, reltol=1e-5)
perfect_data_incidence_infected, noisy_data_incidence_infected = generate_data(4, 111, 
i -> Poisson(i), prob, Tsit5(), times; incidence_obs_status=true, abstol=1e-10, reltol=1e-5)
nothing # hide

# Let plot the perfect and noisy data of recovered!

plt_recovered = plot(times, perfect_data_recovered, dpi = 400, 
labels = "Perfect Recovered Data", xlabel = L"t", ylabel = "Recovered Data")
plot!(times, noisy_data_recovered, labels = "Noisy Recovered Data")

# Let also plot the perfect and noisy incidence data of infected!
plt_incidence_infected = plot(times, perfect_data_incidence_infected, dpi = 400, 
labels = "Perfect Incidence Data", xlabel = L"t", ylabel = "Incidence Data of Infected")
plot!(times, noisy_data_incidence_infected, labels = "Noisy Incidence Infected Data")

# ## Estimating parameters 

# Since there is only data from ``t=0`` to ``t=10``, we can change `tspan` and remake the ODE problem. This saves the computer from doing unnecessary computations.

tspan = (0.0, 10.0)
prob = remake(prob; tspan = tspan)
nothing # hide

# Let fit the ``SIR`` model to the noisy data that we have generated. The package `ProfileLikelihood.jl` already have objective functions derived from maxmimum likelihood estimation. The third input of `const_variance_error` is the standard deviation of the noise distribution which is ``σ = 30``.

obj_const_variance = (data, sol) -> const_variance_error(data, sol, 30.0)
obj_poisson = (data, sol) -> poisson_error(data, sol)
nothing # hide

# To estimate parameters, we use the function `estimate_params`. This accepts the keywords `solver_diff_opts`, `opti_prob_opts`, and `opti_solver_opts` for the differential equation solver, optimization problem, and optimization solver. See the documentation for more information.

solver_diff_opts = Dict(
    :reltol => 1e-5,
    :abstol => 1e-10,
)
nothing # hide

# The first input to `estimate_params` is the initial guess of parameters, the second input is any non-incidence data, and the third input is the indices of the state variables of the DEs that we are interested in. The eighth, ninth, and tenth inputs are the optimizaiton algorithm, lower bounds, and upper bounds. For more information, see the `ProfileLikelihood.jl` documentation and `Optimization.jl` documentation. For this tutorial, we will use the `NOMADOpt()` optimization algorithm from the package `NOMAD.jl` package.

initial_guess = [1.0, 1.0] 
loss, fitted_params = estimate_params(initial_guess, [noisy_data_recovered, noisy_data_incidence_infected], 
[3], prob, Tsit5(), times, [obj_const_variance, obj_poisson], NOMADOpt(), [0.0, 0.0], [3.0, 3.0]; 
incidence_obs = [4], solver_diff_opts=solver_diff_opts)

# This does not look quite right as we know the true parameters is `[0.0001, 0.1]`. We will try again with another optimization algorithm. We will use the `DE()` algorithm from the `Metaheuristics.jl` package.

loss, fitted_params = estimate_params(initial_guess, [noisy_data_recovered, noisy_data_incidence_infected], 
[3], prob, Tsit5(), times, [obj_const_variance, obj_poisson], OptimizationMetaheuristics.DE(N=50), 
[0.0, 0.0], [3.0, 3.0]; incidence_obs = [4], solver_diff_opts=solver_diff_opts, print_status = true)

# To make sure this is the global minimum, we can use another optimization algorithm. We will use the `generating_set_search` from `BlackBoxOptim.jl`.

opti_solver_opts = Dict(
    :maxtime => 120.0, # how long for it to run in seconds
    :f_calls_limit => 1000000, 
    :iterations => 2000000,
    :TraceInterval => 10.0 # print results every 10.0 seconds 
)

loss, fitted_params = estimate_params(initial_guess, [noisy_data_recovered, noisy_data_incidence_infected], 
[3], prob, Tsit5(), times, [obj_const_variance, obj_poisson], BBO_generating_set_search(), 
[0.0, 0.0], [3.0, 3.0]; incidence_obs = [4], solver_diff_opts=solver_diff_opts, 
opti_solver_opts = opti_solver_opts)

# That matches close with what we got with the `DE()` algorithm from the `Metaheuristics.jl` package!

# ## Threshold and constants 
# For profile likelihood, we can calculate likelihood-based confidence intervals using a threshold. Since there are two unknown parameters, we use ``df=2`` for simultaneous confidence intervals and ``df=1`` for pointwise confidence interval. For more information, read Raue et. al's "Exploiting Profile Likelihood".

threshold_simu = find_threshold(0.95, 2, loss)
threshold_poin = find_threshold(0.95, 1, loss)

# Furthermore, for the profile likelihood plots, we dropped the constants when finding the minimizer. We can add the constant back when making the profile likelihood plots. 

pl_const_const_variance = likelihood_const("const_variance_error"; times = times, sigma = 30)
pl_const_poisson = likelihood_const("poisson_error"; data = noisy_data_incidence_infected)
pl_const = pl_const_const_variance + pl_const_poisson

# ## Profile Likelihood 

# We can now plot the profile likelihood. The algorithm implemented explore the profile likelihood by taking steps of equal size in both right and left directions. The first and second argument is the step size and number of steps to take for one direction respectively. The third argument tells the function what parameter we want to fix when plotting the profile likelihood. The seventh argument is the value at which the algorithm will stop computing points if the loss of the computed points is greater than the minimum loss plus the value. We will use the threshold computed earlier plus a small number so that the algorithm explore the profile likelihood a bit more. 

# The plot of the profile likelihood of ``β`` is shown below. 
theta_beta, sol_beta = find_profile_likelihood(2e-8, 50, 1, fitted_params, 
[noisy_data_recovered, noisy_data_incidence_infected], [3], threshold_simu + 3, loss, prob, 
Tsit5(), times, [obj_const_variance, obj_poisson], NOMADOpt(), [0.0, 0.0], [3.0, 3.0]; 
incidence_obs = [4], solver_diff_opts=solver_diff_opts,
pl_const = pl_const, 
print_status = false)
PL_beta = plot(theta_beta, [sol_beta, (x) -> (threshold_simu + pl_const), (x) -> (threshold_poin + pl_const)], 
xlabel = L"\beta", ylabel = L"\chi^2_{\rm PL}", yformatter = :plain, legend=:topright, 
labels = [L"\chi^2_{\rm PL}" "Simultaneous Threshold" "Pointwise Threshold"], right_margin=5mm, 
dpi = 400)
scatter!([fitted_params[1]], [loss + pl_const], color = "orange", labels = "Fitted Parameter")

# The plot of the profile likelihood of ``γ`` is shown below. 
theta_gamma, sol_gamma = find_profile_likelihood(3e-4, 50, 2, fitted_params, 
[noisy_data_recovered, noisy_data_incidence_infected], [3], threshold_simu + 3, loss, prob, 
Tsit5(), times, [obj_const_variance, obj_poisson], NOMADOpt(), [0.0, 0.0], [3.0, 3.0];
incidence_obs = [4], solver_diff_opts=solver_diff_opts, pl_const = pl_const, 
print_status = false)
PL_gamma = plot(theta_gamma, [sol_gamma, (x) -> (threshold_simu + pl_const), (x) -> (threshold_poin + pl_const)], 
xlabel = L"\gamma", ylabel = L"\chi^2_{\rm PL}", yformatter = :plain, legend=:topright, 
labels = [L"\chi^2_{\rm PL}" "Simultaneous Threshold" "Pointwise Threshold"], right_margin=5mm, 
dpi = 400)
scatter!([fitted_params[2]], [loss + pl_const], color = "orange", labels = "Fitted Parameter")

# ## Identifiability analysis and confidence interval estimation 
# If the confidence intervals are finite and the global minimum is unique, then the parameter is practically identifiable. If the conifdence interval is not finite, then the parameter is not practically identifiable. From the profile likelihood plots, the parameters ``β`` and ``γ`` are practically identifiable. 

# The confidence intervals are determined by the intercepts of the threshold and the profile likelihood plot. For this tutorial, we will find simultaneous confidence intervals. We use the `Interpolations.jl` package to make a linear interpolant of the points found.  

beta_inter = linear_interpolation(theta_beta, sol_beta)
gamma_inter = linear_interpolation(theta_gamma, sol_gamma)
nothing # hide

# We can use `find_roots()` from the `ProfileLikelihood.jl` package. 

# The simultaneous confidence interval of ``β`` is ``[9.953 \cdot 10^{-5}, 0.000101]``.
ci_simu_beta_h = find_roots(beta_inter, theta_beta, threshold_simu + pl_const)

# The simultaneous confidence interval of ``γ`` is ``[0.0940,0.102]``.

ci_simu_beta_v = find_roots(gamma_inter, theta_gamma, threshold_simu + pl_const)

