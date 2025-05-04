# Julia Rectifier HTTP Service

This service provides a REST API for the Julia-based rectification algorithm. It closely mirrors the functionality of `rectify.jl` but exposes it via HTTP endpoints.

## What it does

The service accepts three conic sections and computes their intersection points using the `HomotopyContinuation.jl` package. The service returns the complex solutions after normalization and numerical cleaning.

## API Endpoints

### Health Check
- **URL:** `/health`
- **Method:** `GET`
- **Response:** Returns "OK" with status 200 if the service is running

### Rectification
- **URL:** `/rectify`
- **Method:** `POST`
- **Content-Type:** `application/json`
- **Request Body:**
  ```json
  {
    "conics": [
      [a1, b1, c1, d1, e1, f1],
      [a2, b2, c2, d2, e2, f2],
      [a3, b3, c3, d3, e3, f3]
    ]
  }
  ```
  Where each array represents the parameters of a conic section: ax² + bxy + cy² + dxz + eyz + fz²

- **Success Response:**
  - **Code:** 200
  - **Content:** 
    ```json
    {
      "complex_sols": [
        [x1, y1, z1],
        [x2, y2, z2],
        ...
      ]
    }
    ```
    Where each solution is an array of complex values representing a point of intersection in homogeneous coordinates.

- **Error Response:**
  - **Code:** 400 Bad Request
  - **Content:** `{"error": "Invalid input format"}`
  - **Code:** 500 Internal Server Error
  - **Content:** `{"error": "Internal server error", "details": "..."}`

## Running with Docker

### Using docker-compose

1. Build and start the service:
   ```bash
   docker-compose up -d
   ```

2. Check if the service is running:
   ```bash
   curl http://localhost:8081/health
   ```

3. Example API request:
   ```bash
   curl -X POST http://localhost:8081/rectify \
     -H "Content-Type: application/json" \
     -d '{"conics": [
           [1.0, 0.0, 1.0, 0.0, 0.0, -1.0],
           [1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
           [0.0, 0.0, 1.0, 0.0, 0.0, -1.0]
         ]}'
   ```

## Integration with Python

You can easily call this API from Python using the `requests` library:

```python
import requests
import json

def rectify_conics(conics):
    """
    Call the Julia rectifier API with conics data.
    
    Args:
        conics: List of 3 lists, each containing 6 conic parameters [a,b,c,d,e,f]
        
    Returns:
        List of intersection points (complex solutions)
    """
    url = "http://localhost:8081/rectify"
    payload = {"conics": conics}
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        return response.json()["complex_sols"]
    else:
        raise Exception(f"API call failed: {response.json()}")

# Example usage
conics = [
    [1.0, 0.0, 1.0, 0.0, 0.0, -1.0],  # Circle x² + y² - 1 = 0
    [1.0, 0.0, 0.0, 0.0, 0.0, -1.0],  # Horizontal line y = 0
    [0.0, 0.0, 1.0, 0.0, 0.0, -1.0]   # Vertical line x = 0
]

solutions = rectify_conics(conics)
print(solutions)
``` 

# Homotopy Continuation application

The algorithm used in this service is based on the `HomotopyContinuation.jl` package, which provides a robust framework for solving systems of polynomial equations.

## Background & goal
A generic circle whose center is at $(x_0, y_0)$ and whose radius is $r$, is described by the equation $(x-x_0)^2 + (y-y_0)^2 - r^2 = 0$ that can be homogenized to $x^2 + y^2 -2x_0xw - 2y_0yw + (x_0^2 + y_0^2 - r^2)w^2 = 0$. By intersecting a circle with the line at infinity $w=0$ we get:

$$
\begin{cases}
x^2 + y^2 - 2x_0xw - 2y_0yw + (x_0^2 + y_0^2 - r^2)w^2 = 0 \\
w = 0
\end{cases}
$$

That yields $x^2 + y^2 = 0$ solved by $y = \pm i x$. The solutions are thus $[x, ix, 0]^T$ and $[x, -ix, 0]^T$ since we are using homogenous coordinates both vectors can be divided by $x$ to get the solutions $I = [1, i, 0]^T$ and $J = [1, -i, 0]^T$. These are called **Circular Points** and since we have not made any assumptions about the circles we can conclude that they are shared by all circles.

When an Homography is applied to a plane, knowing the images of the circular points is enough to compute a rectifing homography. That's because it is easy to prove that the circular points are invariant under similarities and thus finding an homography that maps them back to their original position is enough to perform shape reconstruction.

Since the circular points are shared by all circles, their images under an homography are shared by all the images of circles under the same homography.

The images of the circular points can thus be found as the intersection of three images of three different circles, and that's the goal of this service. 

## Standard intersection
Given a generic Homography described by the invertible matrix $H \in \mathbb{R}^{3 \times 3}$, and a circle described by the symmetric matrix $C^* \in \mathbb{R}^{3 \times 3}$, the image of the circle under the homography is given by the matrix $C = H^{-T} C^* H^{-1}$ and it is usually an ellipse. 

A point $z \in \mathbb{C}^3$ belongs to a conic if $z^T C z = 0$. Thus, given the images of three distinct circles $C_1, C_2, C_3$ we can find the images of the circular points as the solutions of the following system of equations:
$$
\begin{cases}
z^T C_1 z = 0 \\
z^T C_2 z = 0 \\
z^T C_3 z = 0
\end{cases}
$$

This system is a polynomial system of degree 2 in the variables $z = [x, y, w]^T$ and can be solved using the `HomotopyContinuation.jl` package. 

This method works well when the images of the circles are noiseless but the extraction of the images of the circles is usually noisy and thus the systems quickly becomes solutionless.

## Optimization problem
In order to deal with the noise we can aim to find the point $z$ that best fits the system of equations without necessarily being a solution. Let's start by considering a single conic rappresented by the symmetric matrix $C \in \mathbb{R}^{3 \times 3}$. We are looking for a point $z^* \in \mathbb{C}^3$ such that:
$$
z^* = \arg\min_{z \in \mathbb{C}^3} ||z^T C z||^2_2$$
The $L_2$ norm squared of a generic complex vector $t=[t_1, t_2, t_3]^T, t_{1,2,3} \in \mathbb{C}$ where $t_i$ can be written as $t_i = a_i + i b_i, a_i, b_i \in \mathbb{R}$ is defined as:
$$
||t||^2_2 = \sum_{i=1}^3 |t_i|^2 = \sum_{i=1}^3 (a_i^2 + b_i^2)
$$

Let's remember that we are working in homogenous coordinates and thus a constraint must be added to deal with the scale of the vector $z$. 

The chosen constraint if $g(z) = ||z||^2_2 - 1 = 0$. The problem thus becomes:
$$
z^* = \arg\min_{z \in \mathbb{C}^3} ||z^T C z||^2_2 \quad \\ \text{s.t.} \quad g(z) = 0
$$

Unfortunatley the HomotopyContinuation.jl package does not allow to access the real and imaginary parts of the complex variables and thus we need to convert the problem into a real optimization problem. 

This can be done by expressing the complex vector $z$ as $z = x+ iy$ where $x,y \in \mathbb{R}^3$. The objective function can be rewritten as:
$$
||z^T C z||^2_2 = ||(x+iy)^T C (x+iy)||^2_2 = \\
||x^T C x - y^T C y + i y^T C x + i x^T C y ||^2_2 = \\
||x^T C x - y^T C y||^2_2 + ||2 x^t C y||^2_2
$$

While the constraint becomes:
$$
g(z) = ||z||^2_2 - 1 =\\
||x+iy||^2_2 - 1 = \\
||x||^2_2 + ||y||^2_2 - 1 = 0
$$

The real optimization problem is thus:
$$
z^* = \arg\min_{x,y \in \mathbb{R}^3} ||x^T C x - y^T C y||^2_2 + ||2 x^t C y||^2_2 \quad \\ \text{s.t.} \quad g(z) = 0
$$

Adding baack the three conics $C_1, C_2, C_3$ we get the following optimization problem:
$$
z^* = \arg\min_{x,y \in \mathbb{R}^3} \sum_{i=1}^3 ||x^T C_i x - y^T C_i y||^2_2 + ||2 x^t C_i y||^2_2 \quad \\ \text{s.t.} \quad g(z) = 0
$$

To solve this constrained optimization problem we can use the Lagrange multipliers method. The Lagrangian function is defined as:
$$
\mathcal{L}(x, y, \lambda) = \sum_{i=1}^3 ||x^T C_i x - y^T C_i y||^2_2 + ||2 x^t C_i y||^2_2 + \lambda g(z)
$$

Where $\lambda$ is the Lagrange multiplier. The Lagrangian function combines the objective function and the constraint into a single function. The term $\lambda g(z)$ penalizes any violation of the constraint $g(z) = 0$.

The optimal solution can be found by taking the partial derivatives of the Lagrangian with respect to $x$, $y$, and $\lambda$, and setting them to zero. This gives us a system of equations that we can solve for the optimal values of $x$, $y$, and $\lambda$.

$$
\begin{cases}
\frac{\partial \mathcal{L}}{\partial x} = 0 \\
\frac{\partial \mathcal{L}}{\partial y} = 0 \\
\frac{\partial \mathcal{L}}{\partial \lambda} = 0
\end{cases}
$$

This system of equations is polynomial and it is thus solvable using the `HomotopyContinuation.jl` package.

The solutions found are just stationary points of the Lagrangian function and thus they need to be evaluated to find the global minima.
