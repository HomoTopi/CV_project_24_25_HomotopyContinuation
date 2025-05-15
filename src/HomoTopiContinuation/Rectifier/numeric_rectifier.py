import numpy as np
import logging
import requests
import json
from HomoTopiContinuation.Rectifier.rectifier import Rectifier
from HomoTopiContinuation.DataStructures.datastructures import Conics, Homography, Conic
from sklearn.cluster import KMeans


class NumericRectifier(Rectifier):
    """
    A numeric implementation of the Rectifier using numpy for numerical computation.
    The algorithm is based on the intersection of conics method described in 
        "Jürgen Richter-Gebert Perspectives on Projective Geometry A Guided Tour Through Real and Complex Geometry".
        Chapter 11.1.  

    The algorithm interfaces with a REST API for conic intersection calculations.
    """

    def computeImagesOfCircularPoints(self, C_img):
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

            intersection_points = np.concatenate(
                (result, result2, result3), axis=0)
            # self.logger.info(f"intersection_points: \n{intersection_points}")

            # compute KNN searching for 2 clusters
            # filtered_points = self._get_KMeans_clusters(intersection_points)

            filtered_points = self._normalize_points(intersection_points)

            filtered_points = self._get_common_intersection_points(
                filtered_points)
            filtered_points = self._clear_found_intersection_points(
                filtered_points)
            return filtered_points
        except Exception as e:
            logging.error(f"Error in conic intersection API call: {e}")
            raise

    def rectify(self, C_img: Conics, returnCP: bool = False) -> Homography:
        """
        Rectify a pair of conics using numerical intersection conic method.

        Args:
            C_img: A Conics object representing the image conics.

        Returns:
            Homography: The rectifying homography
        """

        filtered_points = self.computeImagesOfCircularPoints(C_img)

        # Create a homography matrix based on the points
        imDCCP = self.compute_imDCCP_from_solutions(filtered_points)
        H = self._compute_h_from_svd(imDCCP)
        # self.logger.info(f"H: \n{H.H}")
        if returnCP:
            return H, filtered_points[:2]
        return H

    def _get_KMeans_clusters(self, intersection_points) -> np.ndarray:
        """
        Get the KMeans clusters from the intersection points
        """

        N_CLUSTERS = 2
        # remove real solutions
        filtered_points = intersection_points[~np.all(
            np.isreal(intersection_points), axis=1)]
        processed_points = np.column_stack(
            (filtered_points.real, filtered_points.imag))

        kmeans = KMeans(n_clusters=N_CLUSTERS,
                        random_state=0).fit(processed_points)

        # convert back to complex numbers
        filtered_points = np.zeros((N_CLUSTERS, 3), dtype=np.complex128)
        reshaped_centers = kmeans.cluster_centers_.reshape(N_CLUSTERS, 3, 2)

        # for i, e in enumerate(kmeans.cluster_centers_):
        #    assert len(e) == 6, "the cluster center should have 6 elements"
        #    filtered_points[i,:] = np.array([e[0] + e[1] * 1j, e[2] + e[3] * 1j, e[4] + e[5] * 1j])

        filtered_points = reshaped_centers[...,
                                           0] + 1j * reshaped_centers[..., 1]
        assert len(
            filtered_points) == N_CLUSTERS, f"the filtered points should have {N_CLUSTERS} elements"
        # self.logger.info(f"filtered_points: \n{filtered_points}")

        return filtered_points

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
        response = requests.post(url, headers=headers,
                                 data=json.dumps(payload))

        if response.status_code != 200:
            raise Exception(
                f"API call failed with status code {response.status_code}: {response.text}")

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
            reshaped_data = np.array(complex_data).reshape(
                mw_size[1], mw_size[0])
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
        _, inverse_indices, counts = np.unique(
            intersection_points, axis=0, return_inverse=True, return_counts=True)
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

        intersection_points[np.abs(intersection_points) < self.threshold] = 0.0

        # remove imaginary parts if under the treshold
        intersection_points = np.real_if_close(
            intersection_points, tol=self.threshold)

        return intersection_points
