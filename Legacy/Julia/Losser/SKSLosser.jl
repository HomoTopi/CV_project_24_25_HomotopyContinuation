module SKSLosserModule
include("./Losser.jl")

import .LosserModule: Homography
import LinearAlgebra

function normalize_points(points)
    # Center points and scale isotropically
    O = sum(points[:, 1:2], dims=2) / 2
    dir = points[:, 1] - O
    H_s_inv = [
        dir[1] -dir[2] O[1];
        dir[2] dir[1] O[2];
        0 0 1
    ]
    H_s = inv(H_s_inv)
    points_normalized = H_s * points

    @assert isapprox(points_normalized[:, 1], [1, 0, 1])
    @assert isapprox(points_normalized[:, 2], [-1, 0, 1])

    return H_s, points_normalized
end

function solve_kernel(src_points, tgt_points)
    H_e = [
        1 0 0;
        0 0 1;
        0 1 0
    ]

    #The first two points are moved into the thyperbolic points [+-1, 1, 0]
    src_points_hp = H_e * src_points
    tgt_points_hp = H_e * tgt_points

    src_points_hp[:, 3:4] = src_points_hp[:, 3:4] ./ transpose(src_points_hp[3, 3:4])
    tgt_points_hp[:, 3:4] = tgt_points_hp[:, 3:4] ./ transpose(tgt_points_hp[3, 3:4])

    @assert isapprox(src_points_hp[:, 1], [1, 1, 0])
    @assert isapprox(src_points_hp[:, 2], [-1, 1, 0])
    @assert isapprox(tgt_points_hp[:, 1], [1, 1, 0])
    @assert isapprox(tgt_points_hp[:, 2], [-1, 1, 0])

    println("src_points_hp: ", src_points_hp)
    println("tgt_points_hp: ", tgt_points_hp)

    H_T_1 = [
        1 0 -src_points_hp[1, 3];
        0 1 -src_points_hp[2, 3];
        0 0 1
    ]

    H_T_2 = [
        1 0 -tgt_points_hp[1, 3];
        0 1 -tgt_points_hp[2, 3];
        0 0 1
    ]

    src_points_hp_t = H_T_1 * src_points_hp
    tgt_points_hp_t = H_T_2 * tgt_points_hp


    println("src_points_hp_t: ", src_points_hp_t)
    println("tgt_points_hp_t: ", tgt_points_hp_t)

    @assert isapprox(src_points_hp_t[:, 1], [1, 1, 0])
    @assert isapprox(src_points_hp_t[:, 2], [-1, 1, 0])
    @assert isapprox(src_points_hp_t[:, 3], [0, 0, 1])
    @assert isapprox(tgt_points_hp_t[:, 1], [1, 1, 0])
    @assert isapprox(tgt_points_hp_t[:, 2], [-1, 1, 0])
    @assert isapprox(tgt_points_hp_t[:, 3], [0, 0, 1])

    src_points_hp_t[:, 4] = src_points_hp_t[:, 4] ./ transpose(src_points_hp_t[3, 4])
    tgt_points_hp_t[:, 4] = tgt_points_hp_t[:, 4] ./ transpose(tgt_points_hp_t[3, 4])

    A = [
        src_points_hp_t[1, 4] src_points_hp_t[2, 4];
        src_points_hp_t[2, 4] src_points_hp_t[1, 4]
    ]

    b = [
        tgt_points_hp_t[1, 4];
        tgt_points_hp_t[2, 4]
    ]

    coeffs = A \ b

    H_g = [
        coeffs[1] coeffs[2] 0;
        coeffs[2] coeffs[1] 0;
        0 0 1
    ]

    @assert isapprox(H_g * src_points_hp_t[:, 4], tgt_points_hp_t[:, 4])

    return inv(H_e) * inv(H_T_2) * H_g * H_T_1 * H_e

end

#Extends the computeLoss function from LosserModule
function computeLoss(H_true::Homography, H_computed::Homography, width::Int, height::Int)::Float64
    H_computed_inv = LinearAlgebra.inv(H_computed.H)
    H_error = H_computed_inv * H_true.H
    H_error = H_error ./ H_error[3, 3]

    src_points = [
        0 width width 0;
        0 0 height height;
        1 1 1 1
    ]

    tgt_points = H_error * src_points
    tgt_points = tgt_points ./ transpose(tgt_points[3, :])


    T_src, src_normalized = normalize_points(src_points)
    T_tgt, tgt_normalized = normalize_points(tgt_points)
    println(src_normalized)
    println(tgt_normalized)

    K = solve_kernel(src_normalized, tgt_normalized)
    S_source = inv(T_src)
    S_target = T_tgt

    println("S_source: ", S_source)
    println("K: ", K)
    println("S_target: ", S_target)

    H_error_reconstructed = inv(S_target) * K * S_source
    H_error_reconstructed = H_error_reconstructed ./ H_error_reconstructed[3, 3]
    println("H_error_reconstructed: ", H_error_reconstructed)

    @assert isapprox(H_error_reconstructed, H_error)
    return LinearAlgebra.norm(K)
end

export computeLoss
end