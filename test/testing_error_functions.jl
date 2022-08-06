using Pkg
Pkg.activate("ProfileLikelihood.jl")

using Revise, ProfileLikelihood
using DifferentialEquations, Plots, Random, Distributions, LaTeXStrings, BenchmarkTools, Measures

using Test

data = [1.0, 3.0, 6.0, 9.0]
sol = [1.0, 2.0, 5.0, 10.0]
times = [2.0, 5.0, 8.0, 13.0]
println(relative_error(data, sol, 0.3))

@testset "ProfileLikelihood.jl" begin
    # Write your tests here.
end

