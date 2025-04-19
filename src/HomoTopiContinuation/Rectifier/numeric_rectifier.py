import numpy as np
import logging
import requests
import json
from HomoTopiContinuation.Rectifier.rectifier import Rectifier
from HomoTopiContinuation.DataStructures.datastructures import Conics, Homography, Conic

class NumericRectifier(Rectifier):
    """
    A numeric implementation of the Rectifier using numpy for numerical computation.
    The algorithm is based on the intersection of conics method described in 
        "Jürgen Richter-Gebert Perspectives on Projective Geometry A Guided Tour Through Real and Complex Geometry".
        Chapter 11.1.  
        
    The algorithm interfaces with a REST API for conic intersection calculations.
    """

    def rectify(self, C_img: Conics) -> Homography:
        """
        Rectify a pair of conics using numerical intersection conic method.

        Args:
            C_img: A Conics object representing the image conics.
            
        Returns:
            Homography: The rectifying homography
        """
        # Extract the three conics from the Conics object
        C1, C2, C3 = C_img
        
        assert type(C1) == Conic
        assert type(C2) == Conic
        assert type(C3) == Conic
        
        try:
            # Find the intersection points between the conics using the REST API
            result = self._call_intersect_conics_api(C1.M, C2.M)
            
            result2 = self._call_intersect_conics_api(C1.M, C3.M)
            
            result3 = self._call_intersect_conics_api(C2.M, C3.M)

            intersection_points = np.concatenate((result, result2, result3), axis=0)
            self.logger.info("intersection_points: \n", intersection_points)
            
            # set to 0 elements inside the array under a treshold
            intersection_points = self._clear_found_intersection_points(intersection_points)
            filtered_points = self._get_common_intersection_points(intersection_points)
            self.logger.info("filtered_points: \n", filtered_points)

            # Create a homography matrix based on the points
            imDCCP = self.compute_imDCCP_from_solutions(filtered_points)
            H = self._compute_h_from_svd(imDCCP)
            self.logger.info("H: \n", H.H)
            return H
            
        except Exception as e:
            logging.error(f"Error in conic intersection API call: {e}")
            raise
    
    def _call_intersect_conics_api(self, conic1, conic2):
        """
        Call the REST API for conic intersection.
        
        Args:
            conic1: First conic matrix (3x3)
            conic2: Second conic matrix (3x3)
            
        Returns:
            Array of intersection points
        """
        url = "http://localhost:9910/conicIntersector/intersectConics"
        headers = {"Content-Type": "application/json"}
        
        # Prepare the request payload
        payload = {
            "rhs": [conic1.tolist(), conic2.tolist()],
            "nargout": 1
        }
        
        # Make the API call
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        
        if response.status_code != 200:
            raise Exception(f"API call failed with status code {response.status_code}: {response.text}")
        
        # Parse the response
        result_data = response.json()
        # Extract the data array from the response
        mw_data = result_data["lhs"][0]["mwdata"]
        mw_size = result_data["lhs"][0]["mwsize"]
        
        # Check if the data is complex (contains real and imaginary parts)
        # If complex, every two consecutive values represent a complex number
        if "mwcomplex" in result_data["lhs"][0] and result_data["lhs"][0]["mwcomplex"]:
            # Create complex numbers by pairing real and imaginary parts
            complex_data = []
            for i in range(0, len(mw_data), 2):
                if i+1 < len(mw_data):  # Ensure we have both real and imaginary parts
                    complex_data.append(complex(mw_data[i], mw_data[i+1]))
            
            # Reshape the complex data according to the actual size
            # The mw_size should be half the length in the dimension that contains complex values
            reshaped_data = np.array(complex_data).reshape(mw_size[1], mw_size[0])
        else:
            # Regular real data - reshape as before
            reshaped_data = np.array(mw_data).reshape(mw_size[1], mw_size[0])
        
        return reshaped_data
        
    def _get_common_intersection_points(self, intersection_points):
        """
        Filter unique intersection points.
        The result is the intersection points of all the conics.
        Args:
            intersection_points: Array of intersection points
        """
        # TODO: would be nice having a treshold for considering a point as same as another one       
        # First identify which points appear more than once
        _, inverse_indices, counts = np.unique(intersection_points, axis=0, return_inverse=True, return_counts=True)
        # Create a mask for points that appear more than once
        mask = counts[inverse_indices] > 1
        # if mask is empty, return the intersection points
        if not mask.any():
            return intersection_points
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

        return intersection_points
