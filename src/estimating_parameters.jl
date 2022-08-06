function estimateParams(p0, fitter_opts, data, solObserved, prob, solver_opts, times, obj; incidenceObserved = [], paramIndex=0, paramEval=0, order = 4, maxRange = 1e-4, status = "local")
    if paramIndex == 0
        g = (paramsCur) -> likelihood(paramsCur, data, solObserved, prob, solver_opts, times, obj; incidenceObserved = incidenceObserved)
        searchRange = fitter_opts[:searchRange]
        resFirst = bboptimize(g, p0; Method=fitter_opts[:algFirst], SearchRange=searchRange, MaxTime=fitter_opts[:maxTimeFirst], TraceMode=fitter_opts[:traceMode], PopulationSize=1000)
        return best_fitness(resFirst), best_candidate(resFirst)
    else
        if status == "local"
        g = (paramsCur) -> likelihood(paramsCur, data, solObserved, prob, solver_opts, times, obj; incidenceObserved = incidenceObserved, paramIndex=paramIndex, paramEval=paramEval)
        function h(p, gradient)
            if length(gradient) > 0
                gradient[:] = grad(central_fdm(order, 1, max_range=maxRange), g, p)[1]
            end
            return g(p)
        end
        println("Gradient: ", grad(central_fdm(order, 1, max_range=maxRange), g, p0)[1])
        opt = Opt(fitter_opts[:algNLOpt], length(p0))
        opt.min_objective = (p, grad) -> h(p, grad)
        lowerBoundsCopy = copy(fitter_opts[:lowerBoundsNLOpt])
        upperBoundsCopy = copy(fitter_opts[:upperBoundsNLOpt])
        lowerBounds = deleteat!(lowerBoundsCopy, paramIndex)
        upperBounds = deleteat!(upperBoundsCopy, paramIndex)
        opt.lower_bounds = lowerBounds
        opt.upper_bounds = upperBounds
        opt.ftol_rel = 1e-7
        opt.ftol_abs = 1e-14
        (minf, minx, ret) = NLopt.optimize(opt, p0)
        println("Loss: ", minf)
        println("Fitted Parameters: ", minx)
        println("Return value: ", ret)
        # searchRangeCopy = copy(fitter_opts[:searchRange])
        # searchRange = deleteat!(searchRangeCopy, paramIndex)
        # resFirst = bboptimize(g, minx; Method=fitter_opts[:algFirst], SearchRange=searchRange, MaxTime=fitter_opts[:maxTimeFirst], TraceMode=fitter_opts[:traceMode], PopulationSize=500)
        # return best_fitness(resFirst), best_candidate(resFirst)
        return minf, minx
    elseif status == "global"
        g = (paramsCur) -> likelihood(paramsCur, data, solObserved, prob, solver_opts, times, obj; incidenceObserved = incidenceObserved, paramIndex=paramIndex, paramEval=paramEval)
        bounds = copy(fitter_opts[:boundsMH])
        bounds = bounds[:, 1:end .!= paramIndex]
        println(bounds)
        options = Options(time_limit = 240.0, f_calls_limit = 10000000000, iterations = 200000000)
        information = Information()
        algor = DE(N = 100, options = options, information = information)
        resOpt = Metaheuristics.optimize(g, bounds, algor)
        println(resOpt)
        return minimum(resOpt), minimizer(resOpt)
    elseif status == "globalBB"
        g = (paramsCur) -> likelihood(paramsCur, data, solObserved, prob, solver_opts, times, obj; incidenceObserved = incidenceObserved, paramIndex=paramIndex, paramEval=paramEval)
        searchRange = copy(fitter_opts[:searchRange])
        deleteat!(searchRange, paramIndex)
        resFirst = bboptimize(g, p0; Method=fitter_opts[:algFirst], SearchRange=searchRange, MaxTime=fitter_opts[:maxTimeFirst], TraceMode=fitter_opts[:traceMode], PopulationSize=1000)
        return best_fitness(resFirst), best_candidate(resFirst)
    end
    end
end