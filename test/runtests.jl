using MimiFUND
using Mimi
using Test
using DataFrames
using CSVFiles

@testset "fund" begin

#------------------------------------------------------------------------------
#   1. Run the whole model
#------------------------------------------------------------------------------

@testset "fund-model" begin

#default model exported by fund module
default_nsteps = 1050
m = MimiFUND.getmodel()
run(m)
@test Mimi.time_labels(m) == collect(1950:1:1950+default_nsteps)

#default model created by MimiFUND.getmodel()
m1 = MimiFUND.getmodel()
run(m1)
@test Mimi.time_labels(m1) == collect(1950:1:1950+default_nsteps)

#use optional args for MimiFUND.getmodel()
new_nsteps = 10
@test_throws ErrorException m2 = MimiFUND.getmodel(nsteps = new_nsteps) #should error because parameter lenghts won't match time dim

end #fund-model testset

#------------------------------------------------------------------------------
#   2. Run tests to make sure integration version (Mimi v0.5.0)
#   values match Mimi 0.4.0 values
#------------------------------------------------------------------------------
@testset "test-integration" begin

Mimi.reset_compdefs()

m = MimiFUND.getmodel()
run(m)

missingvalue = -999.999
err_number = 1.0e-9
err_array = 0.0

for c in map(name, Mimi.compdefs(m)), v in Mimi.variable_names(m, c)

    #load data for comparison
    filename = joinpath(@__DIR__, "../contrib/validation_data_v040/$c-$v.csv")
    results = m[c, v]

    df = load(filename) |> DataFrame
    if typeof(results) <: Number
        validation_results = df[1,1]
        @test results ≈ validation_results atol = err_number #slight imprecision with these values due to rounding

    else
        validation_results = convert(Matrix, df)

        #replace missings with missingvalue so they can be compared
        results[ismissing.(results)] .= missingvalue
        validation_results[ismissing.(validation_results)] .= missingvalue

        #match dimensions
        if size(validation_results,1) == 1
            validation_results = validation_results'
        end

        @test results ≈ validation_results atol = err_array

    end
end

end #fund-integration testset

#------------------------------------------------------------------------------
# 3. Test marginal damages functions (test that each function does not error)
#------------------------------------------------------------------------------

@testset "test-marginaldamages" begin

# new_marginaldamages.jl

# Test the default SCC function 
scc = MimiFUND.computeSCC(emissionyear = 2020) 
@test scc isa Float64   # test that it's not missing or a NaN

# Test with a modified model and more keyword arguments
m = MimiFUND.getmodel()
update_param!(m, :climatesensitivity, 5)    
scc = MimiFUND.computeSCC(m, emissionyear=2020, eta=0.85, prtp=0.0001, yearstorun=350, useequityweights=true)
@test scc isa Float64   # test that it's not missing or a NaN

# Test getMarginalModel
mm = MimiFUND.getMarginalModel(emissionyear=2020, gas=:CH4)
scc = MimiFUND.computeSCC(mm; emissionyear=2020, gas=:CH4)
@test scc isa Float64   # test that it's not missing or a NaN

# Test old exported versions of the functions
scc = MimiFUND.get_social_cost()
md = MimiFUND.getmarginaldamages()

end #marginaldamages testset

#------------------------------------------------------------------------------
# 4. Run basic test of Marginal Damages and MCS functionality
#------------------------------------------------------------------------------

@testset "test-mcs" begin

# mcs
MimiFUND.run_fund_mcs(10)        # Run 10 trials of basic FUND MCS
MimiFUND.run_fund_scc_mcs(10)    # Run 10 trials of FUND MCS SCC calculations

end #test-mcs testset

end #fund testset

nothing
