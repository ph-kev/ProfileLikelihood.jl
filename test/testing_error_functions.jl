using Pkg
Pkg.activate("ProfileLikelihood.jl")

using Revise, ProfileLikelihood
using DifferentialEquations, Plots, Random, Distributions, LaTeXStrings, BenchmarkTools, Measures

using Test

data = [1.0, 3.0, 6.0, 9.0]
sol = [1.0, 2.0, 9.0, 11.0]
times = [2.0, 5.0, 8.0, 13.0]

@testset "ProfileLikelihood.jl" begin
    @test likelihood_const("constVarianceError"; times=times, sigma = 3.0) ≈ 16.140406575
    @test const_variance_error(data, sol, 3.0) ≈ 1.55555555556
    @test likelihood_const("poissonError"; data = data) ≈ 42.3456763226
    @test poisson_error(data, sol) ≈ -27.6876929218
    @test likelihood_const("relativeError"; times = times, noise_level = 0.1) ≈ -11.0691724783
    @test relative_error(data, sol, 0.1) ≈ 49.9934302965
end

