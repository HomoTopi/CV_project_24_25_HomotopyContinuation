import numpy as np
import logging
from HomoTopiContinuation.Rectifier.rectifier import Rectifier
from HomoTopiContinuation.DataStructures.datastructures import Conics, Homography, Conic
import os
class NumericRectifier(Rectifier):
    """
    A numeric implementation of the Rectifier using numpy for numerical computation.
    The algorithm is based on the intersection of conics method described in 
        "Jürgen Richter-Gebert Perspectives on Projective Geometry A Guided Tour Through Real and Complex Geometry".
        Chapter 11.1.  
        
    The algorithm interface with the matlab Conic Intersection by Pierlugi Taddei.
    https://it.mathworks.com/matlabcentral/fileexchange/28318-conics-intersection
    
    """

    def rectify(self, C_img: Conics) -> Homography:
        """
        Rectify a pair of conics using numerical intersection conic method.

        Args:
            C_img: A Conics object representing the image conics.
            
        Returns:
            Homography: The rectifying homography
        """
        # Import matlab engine for Python
        try:
            import matlab.engine
        except ImportError:
            raise ImportError("MATLAB Engine for Python is required. Install with 'pip install matlabengine'")
        
        
        
        # Extract the three conics from the Conics object
        C1, C2, C3 = C_img
        
        assert type(C1) == Conic
        assert type(C2) == Conic
        assert type(C3) == Conic
        
        
        
        # Start MATLAB engine
        eng = matlab.engine.start_matlab()
        
        # add the script to the matlab path
        eng.addpath(os.path.join(os.path.dirname(__file__), 'Matlab/conicsIntersection1.04'))
        
        # convert the conics to matlab arrays
        C1_matlab = matlab.double(C1.M.tolist())
        C2_matlab = matlab.double(C2.M.tolist())
        C3_matlab = matlab.double(C3.M.tolist())
        
        try:
            # Find the intersection points between the conics
            # Using Pierlugi Taddei's Conic Intersection implementation
            
            #NB in this case just 2 conics are sufficient to get the image of the circular points
            result = eng.intersectConics(C1_matlab, C2_matlab)            
            result2 = eng.intersectConics(C1_matlab, C3_matlab)
            result3 = eng.intersectConics(C2_matlab, C3_matlab)
            
            intersection_points = np.concatenate((result, result2), axis=0)
            intersection_points = np.concatenate((intersection_points, result3), axis=0)
            
            # TODO: get only the complex intersection points
            
            # Convert back to a list or array if needed
            #intersection_points = np.intersect1d(result, result2)
            #intersection_points = np.intersect(intersection_points, result3)
            
            # TODO: set to 0 values under a treshold
            #distances = np.linalg.norm(intersection_points, axis=1)  # Calculate distances from the origin
            #intersection_points[distances < self.treshold] = 0.0 
                
            #print("intersection_points: \n", intersection_points)
            
            # Compute the rectifying homography from the intersection points
            # This will depend on the specific algorithm implementation
            # Compute the rectifying homography from the intersection points
            # This will depend on the specific algorithm implementation
            # For a standard approach:
            # 1. Arrange the intersection points in a specific order
            # 2. Map these points to a canonical position (e.g., unit circle)
            
            # Create a homography matrix based on the points
            H = self._compute_homography_from_points(intersection_points)
            
            # Return as Homography object
            return Homography(H)
            
        except Exception as e:
            logging.error(f"Error in MATLAB conic intersection: {e}")
            raise
        finally:
            # Close MATLAB engine
            eng.quit()
    
    def _compute_homography_from_points(self, points):
        """
        Compute homography matrix from intersection points.
        
        Args:
            points: Array of intersection points between conics
            
        Returns:
            numpy.ndarray: 3x3 homography matrix
        """
        # Implementation depends on the specific algorithm
        # This is a placeholder for the actual computation
        
        # Example implementation:
        # 1. Arrange points in a specific order based on algorithm requirements
        # 2. Compute the homography that maps these points to canonical positions
        
        # Simple placeholder
        H = np.eye(3)
        
        # TODO: Implement actual homography computation based on
        # the intersection of conics algorithm described in the reference
        
        return H
    