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
            # transpose the result to be a 2x3 matrix
            result = np.array(result).T
            result2 = eng.intersectConics(C1_matlab, C3_matlab)
            result2 = np.array(result2).T
            result3 = eng.intersectConics(C2_matlab, C3_matlab)
            result3 = np.array(result3).T

            intersection_points = np.concatenate((result, result2, result3), axis=0)
            # set to 0 elements inside the array under a treshold
            intersection_points = self._clear_found_intersection_points(intersection_points)
            filtered_points = self._get_common_intersection_points(intersection_points)
            
            
            print("filtered_points: \n", filtered_points)
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
            
    
    def _get_common_intersection_points(self, intersection_points):
        """
        Filter unique intersection points.
        The result is the intersection points of all the conics.
        Args:
            intersection_points: Array of intersection points
        """       
        # First identify which points appear more than once
        _, inverse_indices, counts = np.unique(intersection_points, axis=0, return_inverse=True, return_counts=True)
        # Create a mask for points that appear more than once
        mask = counts[inverse_indices] > 1
        # Filter the original array
        filtered_points = np.unique(intersection_points[mask], axis=0)
        return filtered_points
        
    def _clear_found_intersection_points(self, intersection_points):
        """
        Sets to 0 the elements under a treshold and normalize the vector wrt to the third element or the first element (if w = 0)
        
        Args:
            intersection_points: Array of intersection points"""
        
        intersection_points[np.abs(intersection_points) < self.treshold] = 0.0
        
        #remove imaginary parts if under the treshold
        intersection_points = np.real_if_close(intersection_points, tol=self.treshold)
          
            
        # for each row, normalize the vector wrt to the first element (if w = 0)
        for i in range(len(intersection_points)):
            if intersection_points[i,2] != 0:
                intersection_points[i,:] = intersection_points[i,:] / intersection_points[i,2]
            elif intersection_points[i,0] != 0:
                intersection_points[i,:] = intersection_points[i,:] / intersection_points[i,0]
        
    
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
    