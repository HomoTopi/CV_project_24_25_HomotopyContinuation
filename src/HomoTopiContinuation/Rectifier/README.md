# Homography Rectification Library


## Components

The library consists of several components:

- `datastructures.py`: Classes for representing scene descriptions, conics, homographies, and images
- `rectifier.py`: Abstract base class defining the interface for rectification algorithms
- `standard_rectifier.py`: Implementation of the rectifier using SymPy for symbolic computation
- `example.py`: Example usage of the library

## Requirements

- Python 3.6+
- NumPy
- SymPy

## Installation

```bash
pip install numpy sympy
```

## Usage

Here's a simple example of using the library:

```python
import numpy as np
from datastructures import Conic, Conics
from standard_rectifier import StandardRectifier

# Create two conic matrices
M1 = np.array([
    [1.0, 0.2, 0.3],
    [0.2, 1.0, 0.1],
    [0.3, 0.1, -1.0]
])

M2 = np.array([
    [1.2, -0.1, 0.4],
    [-0.1, 0.8, 0.2],
    [0.4, 0.2, -1.5]
])

# Create Conic objects
C1 = Conic(M1)
C2 = Conic(M2)

# Create a Conics object containing both conics
conics = Conics(C1, C2)

# Create the rectifier
rectifier = StandardRectifier()

# Perform rectification
homography = rectifier.rectify(conics)
print(homography.H)
```

## Extending the Library

To implement a new rectification algorithm, create a new class that inherits from `Rectifier` and implements the `rectify` method:

```python
from rectifier import Rectifier
from datastructures import Conics, Homography

class MyRectifier(Rectifier):
    def rectify(self, C_img: Conics) -> Homography:
        # Implement your rectification algorithm here
        pass
```

## License

This code is available under the MIT License. 