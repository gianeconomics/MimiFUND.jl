function compute_sc_prtp_eta(m::Model=MimiFUND.get_model(); 
    gas::Symbol = :CO2, 
    year::Union{Int, Nothing} = nothing,  
    equity_weights::Bool = false, 
    equity_weights_normalization_region::Int = 0,
    last_year::Int = 2200, 
    pulse_size::Float64 = 1e7, 
    return_mm::Bool = false,
    n::Union{Int, Nothing} = nothing,
    trials_output_filename::Union{String, Nothing} = nothing,
    output_dir::Union{String, Nothing} = nothing,
    seed::Union{Int, Nothing} = nothing
    )

    year === nothing ? error("Must specify an emission year. Try `compute_sc(year=2020)`.") : nothing
    !(last_year in 1950:3000) ? error("Invlaid value for `last_year`: $last_year. `last_year` must be within the model's time index 1950:3000.") : nothing
    !(year in 1950:last_year) ? error("Invalid value for `year`: $year. `year` must be within the model's time index 1950:$last_year.") : nothing
    
    output_dir = output_dir === nothing ? joinpath(@__DIR__, "output_FUND/", "SCC $(Dates.format(now(), "yyyy-mm-dd HH-MM-SS")) MC$n") : output_dir
    mkpath("$output_dir/results")
    trials_output_filename = trials_output_filename === nothing ? joinpath(@__DIR__, "$output_dir/trials.csv") : trials_output_filename
    
    scc_file = joinpath(output_dir, "scc.csv")
    open(scc_file, "w") do f 
        write(f, "trial, SCC: $year\n")
    end
    
    rates_file = joinpath(output_dir, "rates.csv")
    open(rates_file, "w") do f
        write(f, "trial, eta, prtp\n")
    end
    
    mm = MimiFUND.get_marginal_model(m; year = year, gas = gas, pulse_size = pulse_size)

    ntimesteps = MimiFUND.getindexfromyear(last_year)
    
    prtp = zeros(n)
    Distributions.rand!(Uniform(0.01, 0.02), prtp)
    eta = zeros(n)
    Distributions.rand!(Uniform(1.385748,1.693692), eta)

    function _fund_sc_post_trial_prtp_eta(sim::SimulationInstance, trialnum::Int, ntimesteps::Int, tup::Union{Tuple, Nothing})
        mm = sim.models[1]  # get the already-run MarginalModel
        (sc_results, year, gas, ntimesteps, equity_weights, equity_weights_normalization_region, eta, prtp) = Mimi.payload(sim)  # unpack the payload information
        sc = sc_from_mm(mm, year = year, gas = gas, ntimesteps = ntimesteps, equity_weights = equity_weights, trialnum = trialnum, eta = eta[trialnum], prtp = prtp[trialnum], equity_weights_normalization_region=equity_weights_normalization_region)
    
        open(scc_file, "a") do f 
            write(f, "$trialnum, $sc\n")
        end
        
        eta1 = eta[trialnum]
        prtp1 = prtp[trialnum]
        
        open(rates_file, "a") do f
            write(f, "$trialnum, $eta1,  $prtp1\n")
        end
    
        sc_results[trialnum] = sc
    end

    function sc_from_mm(mm::MarginalModel; year::Int, gas::Symbol, ntimesteps::Int, equity_weights::Bool, trialnum::Int, equity_weights_normalization_region::Int, eta::Float64, prtp::Float64)

        # Calculate the marginal damage between run 1 and 2 for each year/region
        marginaldamage = mm[:impactaggregation, :loss]

        ypc = mm.base[:socioeconomic, :ypc]

        # Compute discount factor with or without equityweights
        df = zeros(ntimesteps, 16)
        if !equity_weights
            for r = 1:16
                x = 1.
                for t = MimiFUND.getindexfromyear(year):ntimesteps
                    df[t, r] = x
                    gr = (ypc[t, r] - ypc[t - 1, r]) / ypc[t - 1,r]
                    x = x / (1. + prtp + eta * gr)
                end
            end
        else
            normalization_ypc = equity_weights_normalization_region==0 ? mm.base[:socioeconomic, :globalypc][MimiFUND.getindexfromyear(year)] : ypc[MimiFUND.getindexfromyear(year), equity_weights_normalization_region]
            df = Float64[t >= MimiFUND.getindexfromyear(year) ? (normalization_ypc / ypc[t, r]) ^ eta / (1.0 + prtp) ^ (t - MimiFUND.getindexfromyear(year)) : 0.0 for t = 1:ntimesteps, r = 1:16]
        end 

    # Compute global social cost
        sc = sum(marginaldamage[2:ntimesteps, :] .* df[2:ntimesteps, :])   # need to start from second value because first value is missing
        return sc
    end
    
    if n === nothing
        # Run the "best guess" social cost calculation
        run(mm; ntimesteps = ntimesteps)
        sc = MimiFUND._compute_sc_from_mm(mm, year = year, gas = gas, ntimesteps = ntimesteps, equity_weights = equity_weights, eta = eta[1], prtp = prtp[1], equity_weights_normalization_region=equity_weights_normalization_region)
    elseif n < 1
        error("Invalid n = $n. Number of trials must be a positive integer.")
    else
        # Run a Monte Carlo simulation
        simdef = MimiFUND.getmcs()
        payload = (Array{Float64, 1}(undef, n), year, gas, ntimesteps, equity_weights, equity_weights_normalization_region, eta, prtp) # first item is an array to hold SC values calculated in each trial
        Mimi.set_payload!(simdef, payload) 
        seed !== nothing ? Random.seed!(seed) : nothing
        si = run(simdef, mm, n, ntimesteps = ntimesteps, post_trial_func = _fund_sc_post_trial_prtp_eta, trials_output_filename = trials_output_filename)
        sc = Mimi.payload(si)[1]
    end

    if return_mm
        return (sc = sc, mm = mm)
    else
        return sc
    end
end
