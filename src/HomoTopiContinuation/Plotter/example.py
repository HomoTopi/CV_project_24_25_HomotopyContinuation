import numpy as np
import matplotlib.pyplot as plt
from HomoTopiContinuation.DataStructures.datastructures import Conics, Conic
from HomoTopiContinuation.Plotter.plotter import Plotter

def main():
    """
    Example demonstrating the usage of the plotter
    """
    # Create two conics
    # These are just example conics - replace with real data for actual use
    
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
    # Create a plotter object
    plotter = Plotter() 
    # Plot the conics
    plotter.plot_conics(conics)
    print("Conics plotted successfully!")

if __name__ == "__main__":
    main()