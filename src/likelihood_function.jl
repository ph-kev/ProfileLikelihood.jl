function likelihood(paramsCur, data::Vector{Vector{T}}, solObserved, prob, solver_opts, times, objArr; incidenceObserved = [], paramIndex=0, paramEval=0) where {T<:Real}
    solObservedCopy = vcat(solObserved, incidenceObserved)
    if paramIndex != 0
        paramsCurCopy = copy(paramsCur)
        insert!(paramsCurCopy, paramIndex, paramEval)
        probCur = remake(prob, p=paramsCurCopy)
    else
        probCur = remake(prob, p=paramsCur)
    end

    # solve odes
    sol = solve(
        probCur,
        solver_opts[:alg],
        reltol=solver_opts[:reltol],
        abstol=solver_opts[:abstol],
        saveat=times,
        save_idxs=solObservedCopy
    )
    # loss
    loss = 0.0
    index = 0 
    for ind in eachindex(solObserved)
        loss += objArr[ind](data[ind], sol[ind,:])
        index += 1
    end
    for ind in eachindex(incidenceObserved)
        incidenceSol = generateIncidenceData(sol[ind + index,:])
        loss += objArr[ind + index](data[ind + index][2:end], incidenceSol[2:end]) # remove the first data point for cumulative data since it is always 0 and taking the log of 0 is -Inf  
    end
    return loss
end