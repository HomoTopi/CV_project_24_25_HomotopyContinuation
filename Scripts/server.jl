using HomotopyContinuation
using LinearAlgebra

tooCloseThreshold = 1e-3

"""
This function processes a set of conic parameters to build the conic matrix

# Arguments
- `params`: A vector of parameters that define the conics.

# Returns
- A 3x3 matrix representing the conic defined by the parameters.

# Examples
- `params = [1, 0.0, 1.0, 0.0, 0.0, -1.0]`
- `buildMatrixFromParams(params)` returns a 3x3 matrix representing the conic.
"""
function buildMatrixFromParams(params)
    a, b, c, d, e, f = params
    return [a b/2 d/2; b/2 c e/2; d/2 e/2 f]
end

"""
This function processes a set of conic parameters to build a vector of matrices.
# Arguments
- `params`: A vector of parameters that define three conics, each defined by six parameters.
# Returns
- A vector containing the flattened matrices of the three conics.
# Examples
- `params = [1, 0.0, 1.0, 0.0, 0.0, -1.0, 1, 0.0, 1, -0.0, -1.0, -0.75, 1.0, 0.0, 1.0, 0.0, 0.0, -2.25]`
- `buildMatricesFromParams(params)` returns a vector containing the flattened matrices of the three conics.
"""
function buildMatricesFromParams(params)
    C_1 = buildMatrixFromParams(params[1:6])
    C_2 = buildMatrixFromParams(params[7:12])
    C_3 = buildMatrixFromParams(params[13:18])
    return vcat(vec(C_1), vec(C_2), vec(C_3))
end

#(Direct|Optimization).(MatrixSpace|EllipseSpace).(WholeComplex|ReIm)

#0.0.0
function findImCP_Direct_MatrixSpace_WholeComplex(params)
    Conics = buildMatricesFromParams(params)
    @info "Conics = " * string(Conics)

    # Define variables - using same variable names as rectify.jl (x, y, z instead of x, y, w)
    @var x[1:3], C[1:3, 1:3], Cs[1:3, 1:3, 1:3]

    belongToConic = x' * C * x
    @info "belongToConic = " * string(belongToConic)

    conicsConstraints = []
    for i in 1:3
        conicsConstraints = vcat(conicsConstraints, subs(belongToConic, C => Cs[:, :, i]))
    end
    @info "conicsConstraints = " * string(conicsConstraints)

    normConstraint = x' * x - 1
    @info "normConstraint = " * string(normConstraint)

    F = System(
        vcat(conicsConstraints, normConstraint),
        variables=x,
        parameters=vec(Cs)
    )
    @info "F = " * string(F)

    res = solve(F, [x], show_progress=true, start_system=:total_degree, target_parameters=Conics)
    @info "res = " * string(res)

    solutions_ = solutions(res, only_real=false, only_finite=false, only_nonsingular=false)
    @info "solutions = " * string(solutions_)

    equations = vcat(conicsConstraints, normConstraint)
    solutionsEvaluations = [
        norm([Complex{Float64}(subs(eq, Cs => reshape(Conics, size(Cs)), x => sol)) for eq in equations])
        for sol in solutions_
    ]
    @info "solutionsEvaluations = " * string(solutionsEvaluations)

    sorted_indices = sortperm(solutionsEvaluations)
    @info "sorted_indices = " * string(sorted_indices)

    smallestIndices = sorted_indices[1:2]
    smallestEvaluations = solutionsEvaluations[smallestIndices]
    @info "smallestEvaluations = " * string(smallestEvaluations)

    smallestSolutions = solutions_[smallestIndices]
    @info "smallest_solution = " * string(smallestSolutions)

    return smallestSolutions
end

#0.0.1
function findImCP_Direct_MatrixSpace_ReIm(params)
    Conics = buildMatricesFromParams(params)
    @info "Conics = " * string(Conics)

    # Define variables - using same variable names as rectify.jl (x, y, z instead of x, y, w)
    @var x[1:3], y[1:3], C[1:3, 1:3], Cs[1:3, 1:3, 1:3]

    belongToConicRe = x' * C * x - y' * C * y
    belongToConicIm = x' * C * y
    belongToConic = vcat(belongToConicRe, belongToConicIm)
    @info "belongToConic = " * string(belongToConic)

    conicsConstraints = []
    for i in 1:3
        conicsConstraints = vcat(conicsConstraints, subs(belongToConic, C => Cs[:, :, i]))
    end
    @info "conicsConstraints = " * string(conicsConstraints)

    normConstraint = x' * x + y' * y - 1
    @info "normConstraint = " * string(normConstraint)

    F = System(
        vcat(conicsConstraints, normConstraint),
        variables=vec(x) ∪ vec(y),
        parameters=vec(Cs)
    )
    @info "F = " * string(F)

    res = solve(F, [x, y], show_progress=true, target_parameters=Conics, start_system=:total_degree)
    @info "res = " * string(res)

    solutions_ = solutions(res, only_real=false, only_finite=false, only_nonsingular=false)
    # @info "solutions = " * string(solutions_)

    equations = vcat(conicsConstraints, normConstraint)
    solutionsEvaluations = [
        norm([Complex{Float64}(subs(eq, Cs => reshape(Conics, size(Cs)), x => sol[1:3], y => sol[4:6])) for eq in equations])
        for sol in solutions_
    ]
    # @info "solutionsEvaluations = " * string(solutionsEvaluations)

    sorted_indices = sortperm(solutionsEvaluations)
    # @info "sorted_indices = " * string(sorted_indices)

    smallestIndices = sorted_indices[1]
    currentIndex = sorted_indices[2]
    while norm(solutionsEvaluations[currentIndex] - solutionsEvaluations[smallestIndices[1]]) < tooCloseThreshold && currentIndex <= length(solutionsEvaluations)
        currentIndex += 1
    end
    smallestIndices = vcat(smallestIndices, currentIndex)
    smallestEvaluations = solutionsEvaluations[smallestIndices]
    @info "smallestEvaluations = " * string(smallestEvaluations)

    smallestSolutions = solutions_[smallestIndices]
    @info "smallest_solution = " * string(smallestSolutions)

    reconstructedColutions = [sol[1:3] + im * sol[4:6] for sol in smallestSolutions]
    @info "reconstructedSolutions = " * string(reconstructedColutions)

    normalizedReconstructedSolutions = [sol / sol[1] for sol in reconstructedColutions]
    @info "normalizedReconstructedSolutions = " * string(normalizedReconstructedSolutions)

    return normalizedReconstructedSolutions
end

#1.0.1
function findImCP_Optimization_MatrixSpace_ReIm(params)
    Conics = buildMatricesFromParams(params)
    @info "Conics = " * string(Conics)

    # Define variables - using same variable names as rectify.jl (x, y, z instead of x, y, w)
    @var x[1:3], y[1:3], C[1:3, 1:3], Cs[1:3, 1:3, 1:3], lambda

    belongToConicRe = (x' * C * x - y' * C * y)^2
    belongToConicIm = 4(x' * C * y)^2
    belongToConic = belongToConicRe + belongToConicIm
    @info "belongToConic = " * string(belongToConic)

    conicConstraint = sum([subs(belongToConic, C => Cs[:, :, i]) for i in 1:3])
    @info "conicConstraint = " * string(conicConstraint)

    normConstraint = x' * x + y' * y - 1
    @info "normConstraint = " * string(normConstraint)

    J = conicConstraint + lambda * normConstraint
    @info "J = " * string(J)

    dJ_dx = differentiate(J, x)
    dJ_dy = differentiate(J, y)
    @info "dJ_dx = " * string(dJ_dx)
    @info "dJ_dy = " * string(dJ_dy)

    F = System(
        vcat(dJ_dx, dJ_dy, normConstraint),
        variables=vec(x) ∪ vec(y) ∪ [lambda],
        parameters=vec(Cs)
    )
    @info "F = " * string(F)

    res = solve(F, [x, y, lambda], show_progress=true, target_parameters=Conics, start_system=:total_degree)
    @info "res = " * string(res)

    solutions_ = solutions(res, only_real=false, only_finite=false, only_nonsingular=false)
    # @info "solutions = " * string(solutions_)

    equations = conicConstraint
    solutionsEvaluations = [
        norm([Complex{Float64}(subs(eq, Cs => reshape(Conics, size(Cs)), x => sol[1:3], y => sol[4:6], lambda => sol[7])) for eq in equations])
        for sol in solutions_
    ]
    # @info "solutionsEvaluations = " * string(solutionsEvaluations)

    sorted_indices = sortperm(solutionsEvaluations)
    # @info "sorted_indices = " * string(sorted_indices)



    smallestIndices = sorted_indices[1]
    currentIndex = sorted_indices[2]
    while norm(solutionsEvaluations[currentIndex] - solutionsEvaluations[smallestIndices[1]]) < tooCloseThreshold
        currentIndex += 1
    end
    smallestIndices = vcat(smallestIndices, currentIndex)
    smallestEvaluations = solutionsEvaluations[smallestIndices]
    @info "smallestEvaluations = " * string(smallestEvaluations)

    smallestSolutions = solutions_[smallestIndices]
    @info "smallest_solution = " * string(smallestSolutions)

    reconstructedColutions = [sol[1:3] + im * sol[4:6] for sol in smallestSolutions]
    @info "reconstructedSolutions = " * string(reconstructedColutions)

    normalizedReconstructedSolutions = [sol / sol[1] for sol in reconstructedColutions]
    @info "normalizedReconstructedSolutions = " * string(normalizedReconstructedSolutions)

    return normalizedReconstructedSolutions
end







# Example usage
c_1 = [1, 0.0, 1.0, 0.0, 0.0, -1.0]
c_2 = [1, 0.0, 1, -0.0, -1.0, -0.75]
c_3 = [1.0, 0.0, 1.0, 0.0, 0.0, -2.25]

params = vcat(c_1, c_2, c_3);
result = findImCP_Optimization_MatrixSpace_ReIm(params);