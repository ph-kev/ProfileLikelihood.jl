function findThresholdOptimization(confidence, numsParams, loss)
    threshold = loss + cquantile(Chisq(numsParams), 1 - confidence)
    println("The threshold to use for optimization is $threshold.")
    return threshold
end
