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