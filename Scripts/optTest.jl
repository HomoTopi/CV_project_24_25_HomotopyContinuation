using HomotopyContinuation
using LinearAlgebra

@var x[1:3] C_1[1:3, 1:3] C_2[1:3, 1:3]

quadratic_1 = x' * C_1 * x
quadratic_2 = x' * C_2 * x
constraint = x' * x - 1
@info "constraint=" * string(constraint)

@info "quadratic_1=" * string(quadratic_1)

@info "quadratic_2=" * string(quadratic_2)

J_1 = differentiate(quadratic_1, x)
@info "J=" * string(J)

system = System([quadratic_1; quadratic_2; constraint], variables=x, parameters=vec(C_1) ∪ vec(C_2))
@info "system=" * string(system)

C_1_ = [1 0 0; 0 1 0; 0 0 1]
C_2_ = [1 0 0; 0 1 0; 0 0 2]

res = solve(system, target_parameters=vcat(vec(C_1_), vec(C_2_)), show_progress=true)
@info "res=" * string(res)

found_solutions = solutions(res, only_nonsingular=false)
@info "solutions=" * string(found_solutions)

#For each solution normalize by deviding by the first element of the solution vector
for i in 1:length(found_solutions)
    found_solutions[i] = found_solutions[i] / found_solutions[i][1]
end
@info "normalized solutions=" * string(found_solutions)