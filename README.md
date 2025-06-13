# Computer Vision Project 2024/2025 - Homotopy Continuation

This repository contains a comprehensive framework to explore the application of homotopy continuation techniques for image rectification and to investigate whether this method is more robust with respect to standard numerical and symbolic solvers.

## Documentation

A detailed description of the project is available in the [report](deliverables/report.pdf).

## Authors - HomoTopi
* Filippo Balzarini
* Paolo Ginefra
* Martina Missana




## Overview

### Homotopy Continuation

The project leverages the `HomotopyContinuation.jl` package for solving systems of polynomial equations using the Homotopy Continuation approach.


### Simulator

To test the application of homotopy continuation techniques to image rectification it is used a simulator
with the following pipeline:

- **[Scene Generator](src/HomoTopiContinuation/SceneGenerator/scene_generator.py)**:
  
  Given a scene definition it generates synthetic scenes warping three circles and creating an homography matrix.

- **[Rectifier](src/HomoTopiContinuation/Rectifier/rectifier.py)**:
  
  Abstract class to reconstruct the scene using different approaches:
    * [Homotopy Rectifier](src/HomoTopiContinuation/Rectifier/homotopyc_rectifier.py)
    * [Numeric Rectifier](src/HomoTopiContinuation/Rectifier/numeric_rectifier.py)

  It reconstruct the image computing an homography matrix.

- **[Conic Warper](src/HomoTopiContinuation/ConicWarper/ConicWarper.py)**:
  
  Warps the conics using the computed homography.

- **[Losser](src/HomoTopiContinuation/Losser/Losser.py)**:
  
  Computes reconstruction errors and distortions using various metrics:
    * [Circle Loss](src/HomoTopiContinuation/Losser/CircleLosser.py): 
      
      uses the eccentricity of the reconstructed conics.
    * [Angle Distortion Loss](src/HomoTopiContinuation/Losser/AngleDistortionLosser.py): 
    
      uses the cosine of the angle between two perpendicular lines.
    * [Circular Points Loss](src/HomoTopiContinuation/Losser/CPLosser.py): 
    
      uses the distance of the reconstructed  circular points from the true ones.
    * [Frobenious Norm Loss](src/HomoTopiContinuation/Losser/FrobNormLosser.py): 
    
      uses Frobenius norm of the difference between two homography matrices.
    * [Lines at the infinity Loss](src/HomoTopiContinuation/Losser/LinfLosser.py): 
    
      uses the angle between the lines at infinity.
    * [Reconstruction Error Loss](src/HomoTopiContinuation/Losser/ReconstructionErrorLosser.py): 
    
      uses the L2 norm of the difference between the warped points using the true and the computed homographies.


## Installation

### Python Dependencies

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:

    ```pip install -r requirements.txt```

3. Install the package in development mode:

    ```pip install -e .```

### Julia Dependencies

1. Install Julia from [JuliaLang.org](https://julialang.org/)

2. Install the `HomotopyContinuation` package:
   ```julia
   using Pkg
   Pkg.add("HomotopyContinuation")
   ```

### MATLAB Dependencies
1. Install MATLAB from [MathWorks](https://it.mathworks.com/products/matlab.html)

### Docker Setup and usage

 1. Build and start the services:
    ```docker-compose up -d```

 2. Example Julia API call:

    ```bash
    curl -X POST http://localhost:8081/rectify \
      -H "Content-Type: application/json" \
      -d '{"conics": [
            [1.0, 0.0, 1.0, 0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, -1.0]
          ]}'
          ```

 3. Example Conics Intersection API call:

    ```bash
    curl -X POST http://localhost:8082/intersect \
      -H "Content-Type: application/json" \
      -d '{"conics": [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
            [[3.0, 0.0, 0.0], [0.0, 0.0, -0.5], [0.0, -0.5, -2.0]]
          ]}'
        ```

 4. Check services health:

    ```bash
    curl http://localhost:8081/health
    curl http://localhost:8082/health 
    ```

### Optuna

To test different experiment parameters [Optuna](https://optuna.org/) is used.
To visualize the Optuna dashboard

```bash
optuna-dashboard sqlite:///db.sqlite3
```

## Testing

To run the tests for the Python package, use the following command:

```bash
pytest
```