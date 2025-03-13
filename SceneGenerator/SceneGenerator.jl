module SceneGenerator

    include("../DataStructures/Datastructures.jl")
    using .Datastructures
    using CairoMakie # for plotting

    ##################################
    ### ---- exposed function ----###
    ##################################
    function generate_scene(sd::SceneDescription)::Img
        # Generate the true conics
        conics_true = Conics(circlealg2par(sd.circle1), circlealg2par(sd.circle2))
        # Compute the homography matrix
        H = homography_matrix(sd.f, sd.theta)
        H_inv = inv(H)
        # Apply homography to the true conics
        conic1 = Conic(transpose(H_inv) * conics_true.C1.M * H_inv)
        conic2 = Conic(transpose(H_inv) * conics_true.C2.M * H_inv)
        conics = Conics(conic1,conic2)
        # Plot the conics
        plot_conics(conics_true,conics)
        return Img(Homography(H),conics)
    end

    #Plot true conics and conics after homography
    function plot_conics(CT::Conics,C::Conics)
        #TODO: Fix range of the plot and improve the visualization
        # true conic parameters
        a1, b1, c1, d1, e1, f1 = conic_par2alg(CT.C1)
        a2, b2, c2, d2, e2, f2 = conic_par2alg(CT.C2)
        # conic parameters after the homography
        a3, b3, c3, d3, e3, f3 = conic_par2alg(C.C1)
        a4, b4, c4, d4, e4, f4 = conic_par2alg(C.C2)
        # Create the figure and axis
        fig = Figure()
        ax1 = Axis(fig[1,1], title="TrueConics",aspect = DataAspect())
        ax2 = Axis(fig[1,2], title="ConicsAfterHomography",aspect = DataAspect())
        # range of plot
        x_range = range(-500, 500, length=1000)
        y_range = range(-500, 500, length=1000)
        # plot the conics
        CT1 = [a1*x^2 + b1*x*y + c1*y^2 + d1*x + e1*y + f1 for x in x_range, y in y_range]
        CT2 = [a2*x^2 + b2*x*y + c2*y^2 + d2*x + e2*y + f2 for x in x_range, y in y_range]
        C1 = [a3*x^2 + b3*x*y + c3*y^2 + d3*x + e3*y + f3 for x in x_range, y in y_range]
        C2 = [a4*x^2 + b4*x*y + c4*y^2 + d4*x + e4*y + f4 for x in x_range, y in y_range]
        contour!(ax1, x_range, y_range, CT1, levels=[0])
        contour!(ax1, x_range, y_range, CT2, levels=[0])
        contour!(ax2, x_range, y_range, C1, levels=[0])   
        contour!(ax2, x_range, y_range, C2, levels=[0])
        save("SceneGenerator/conic_plot.png", fig)
        display(fig)
    
        # Plot the conic
        #x_range = range(xlims[1], xlims[2], length=n)
        #y_range = range(ylims[1], ylims[2], length=n)
        #plot = contour(x_range, y_range, (x, y) -> a*x^2 + b*x*y + c*y^2 + d*x + e*y + f, levels=[0], linewidth=2)
        #display(plot)
    end

    
    function conic_par2alg(C::Conic)
        ## Conversion from the matrix form to the parameters form
        #a = C[1,1]
        #b = C[1,2]*2
        #c = C[2,2]
        #d = C[1,3]*2
        #e = C[2,3]*2
        #f = C[3,3]
        #return a, b, c, d, e, f
        return C.M[1,1], C.M[1,2]*2, C.M[2,2], C.M[1,3]*2, C.M[2,3]*2, C.M[3,3]
    end

    function circlealg2par(circle::Vector)::Conic
        # Draw a circle with center (centerX, centerY) and radius
        # (x - centerX)^2 + (y - centerY)^2 = radius^2
        # => x^2 + y^2 - 2*centerX*x - 2*centerY*y + centerX^2 + centerY^2 - radius^2 = 0
        # => x^2 + y^2 + 0*x + 0*y + 0*x^2 + 0*y^2 - 2*centerX*x - 2*centerY*y + centerX^2 + centerY^2 - radius^2 = 0
        return Conic([1.0  0.0  -circle[1]; 0.0  1.0  -circle[2]; -circle[1]  -circle[2]  (circle[1]^2 + circle[2]^2 - circle[3]^2)])
    end

    #Generate the homography matrix
    function homography_matrix(f, theta)::Matrix{Float64}
        # Convert degrees to radians
        theta_rad = deg2rad(theta)

        # Define intrinsic matrix (assuming principal point is [0,0])
        K = [f 0.0 0.0;
            0.0 f 0.0;
            0.0 0.0 1.0]
        
        # Rotation matrix
        R = [1.0 0.0 0.0;
            0.0 cos(theta_rad) -sin(theta_rad);
            0.0 sin(theta_rad) cos(theta_rad)]
        
        # Homography is K * R * K^{-1}
        H = K * R * inv(K)
        return H
    end

    export generate_scene
end