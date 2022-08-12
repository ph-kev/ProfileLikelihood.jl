# Packages 
using ProfileLikelihood
using Test

@testset "ProfileLikelihood.jl" begin
    @test find_threshold(0.95, 7, 2.0) ≈ 16.0671404493402
    @test find_threshold(0.9, 3, 0.0) ≈ 6.25138863117032
    @test find_threshold(0.68, 1, -5.0) ≈ -4.01105351852198
end
