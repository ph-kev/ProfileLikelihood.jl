function minPoint(paramIndex, loss, parametersFitted)
    return [parametersFitted[paramIndex]], [loss]
end

function goRightPL(stepSize, maxSteps, paramIndex, parametersFitted, data, solObserved, prob, solver_opts, times, loss, upperBound, fitter_opts, obj, incidenceObserved, order, maxRange, status)
    thetaRight = Vector{Float64}()
    solRight = Vector{Float64}()
    currVal = loss
    paramsFittedCopy = copy(parametersFitted)
    paramToLookAt = paramsFittedCopy[paramIndex]
    paramGuess = deleteat!(paramsFittedCopy, paramIndex)
    iter = 0
    if paramToLookAt + stepSize > fitter_opts[:searchRange][paramIndex][2]
        return thetaRight, solRight
    end
    while currVal < upperBound && iter < maxSteps && paramToLookAt < fitter_opts[:searchRange][paramIndex][2]
        paramToLookAt = paramToLookAt + stepSize
        append!(thetaRight, paramToLookAt)
        loss, paramGuess = estimateParams(paramGuess, fitter_opts, data, solObserved, prob, solver_opts, times, obj; incidenceObserved = incidenceObserved, paramIndex=paramIndex, paramEval=paramToLookAt, order = order, maxRange = maxRange, status = status)
        append!(solRight, loss)
        iter = iter + 1
        currVal = loss
    end
    return thetaRight, solRight
end

function goLeftPL(stepSize, maxSteps, paramIndex, parametersFitted, data, solObserved, prob, solver_opts, times, loss, upperBound, fitter_opts, obj, incidenceObserved, order, maxRange, status)
    thetaLeft = Vector{Float64}()
    solLeft = Vector{Float64}()
    currVal = loss
    paramsFittedCopy = copy(parametersFitted)
    paramToLookAt = paramsFittedCopy[paramIndex]
    paramGuess = deleteat!(paramsFittedCopy, paramIndex)
    iter = 0
    if paramToLookAt - stepSize < fitter_opts[:searchRange][paramIndex][1]
        return reverse(thetaLeft), reverse(solLeft)
    end
    while currVal < upperBound && iter < maxSteps && paramToLookAt > fitter_opts[:searchRange][paramIndex][1]
        paramToLookAt = paramToLookAt - stepSize
        append!(thetaLeft, paramToLookAt)
        loss, paramGuess = estimateParams(paramGuess, fitter_opts, data, solObserved, prob, solver_opts, times, obj; incidenceObserved = incidenceObserved, paramIndex=paramIndex, paramEval=paramToLookAt, order = order, maxRange = maxRange, status = status)
        append!(solLeft, loss)
        iter = iter + 1
        currVal = loss
    end
    return reverse(thetaLeft), reverse(solLeft)
end

function PL(stepSize, maxSteps, paramIndex, parametersFitted, data, solObserved, upperBound, loss, prob, solver_opts, times, fitter_opts, obj; incidenceObserved = [], order = 4, maxRange = 1e-4, status = "local")
    thetaRight, solRight = goRightPL(stepSize, maxSteps, paramIndex, parametersFitted, data, solObserved, prob, solver_opts, times, loss, upperBound, fitter_opts, obj, incidenceObserved, order, maxRange, status)
    thetaLeft, solLeft = goLeftPL(stepSize, maxSteps, paramIndex, parametersFitted, data, solObserved, prob, solver_opts, times, loss, upperBound, fitter_opts, obj, incidenceObserved, order, maxRange, status)
    thetaPoint, solPoint = minPoint(paramIndex, loss, parametersFitted)
    theta = vcat(thetaLeft, thetaPoint, thetaRight)
    sol = vcat(solLeft, solPoint, solRight)
    return theta, sol
end