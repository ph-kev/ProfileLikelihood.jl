using ProfileLikelihood, DifferentialEquations, Plots, Distributions

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

perfectData, noisyData = generate_data(5, 366, i -> truncated(Poisson(i), lower = -eps(Float64)), prob, Tsit5(), times; incidenceStatus = true, abstol = 1e-10, reltol = 1e-5)

plt = plot(times, perfectData)
plot!(times, noisyData)
display(plt)







