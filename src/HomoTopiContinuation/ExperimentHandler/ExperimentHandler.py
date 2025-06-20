from time import sleep
from tqdm import tqdm
from enum import Enum
import json
import pandas as pd
from itertools import product
import numpy as np
from HomoTopiContinuation.Rectifier import standard_rectifier as sr, homotopyc_rectifier as hr
from HomoTopiContinuation.SceneGenerator.scene_generator import SceneGenerator
from HomoTopiContinuation.ConicWarper.ConicWarper import ConicWarper
from HomoTopiContinuation.Losser import AngleDistortionLosser, CircleLosser, FrobNormLosser, LinfLosser
from HomoTopiContinuation.DataStructures.datastructures import Circle, SceneDescription, Img, Homography, Conics


class ExperimentHandler:
    """
    Class to handle the experiment pipeline.
    """
    def __init__(self):
        """
        Initialize the ExperimentHandler class.
        Initialize the scene generator, conic warper, and rectifiers.
        """
        self.scene_generator = SceneGenerator()
        self.conic_warper = ConicWarper()   
        self.standard_rectifier = sr.StandardRectifier()
        self.homotopyc_rectifier = hr.HomotopyContinuationRectifier()
        self.rectifiers = {
            "standard": self.standard_rectifier,
            "homotopy": self.homotopyc_rectifier
        }
        return
    
    def runExperiment(self, path, path_scene):
        """
        Run the experiment pipeline.
        """
        #TODO: resilience to errors in the pipeline
        scene = self.sceneDefinition(path, path_scene)
        self.generateSceneDescriptionJson(scene, path)
        print("sceneDescription.json generated")
        self.sceneGeneration(path)
        self.rectify(path, scene[-1])
        self.warpConics(path)
        self.computeLosses(path)
        self.json_to_df(path)
        return
    
    def computeLosses(self, path):
        """
        Compute the losses for the given scenes and homographies.
        """
        with open(path + '/sceneDescription.json', 'r') as file:
            scene_data = json.load(file)
        with open(path + '/sceneImage.json', 'r') as file:
            image_data = json.load(file)
        with open(path + '/rectifiedHomography.json', 'r') as file:
            homography_data = json.load(file)
        with open(path + '/rectifiedConics.json', 'r') as file:
            conics_data = json.load(file)

        losses = []
        for scene_json, img_json, H_json, conic_json in tqdm(
        zip(scene_data, image_data, homography_data, conics_data),
        total=len(scene_data), desc="Computing losses"):
            scene = SceneDescription.from_json(scene_json)
            img = Img.from_json(img_json)
            H = Homography.from_json(H_json)
            conics = Conics.from_json(conic_json)

            # Compute the loss for each type
            angle_loss = AngleDistortionLosser.AngleDistortionLosser.computeLoss(img.h_true, H)
            circle_loss = CircleLosser.CircleLosser.computeCircleLoss(scene, conics)
            frob_loss = FrobNormLosser.FrobNormLosser.computeLoss(img.h_true, H)
            linf_loss = LinfLosser.LinfLosser.computeLoss(img.h_true, H)

            losses.append({
                "angle_loss": angle_loss,
                "circle_loss": circle_loss,
                "frob_loss": frob_loss,
                "linf_loss": linf_loss
            })
        with open(path + '/losses.json', 'w') as file:
            losses_serializable = json.loads(json.dumps(losses, default=str))
            json.dump(losses_serializable, file, indent=4)
    
    def extractFields(self, scene, image, homography, conics, losses):
        """
        Method to extract signicative fields from the JSON files in order to create the output dataframe.
        """
        return {
            "circle1Center:": scene["circle1"]["center"],
            "circle2Center:": scene["circle2"]["center"],
            "circle3Center:": scene["circle3"]["center"],
            "circle1Radius:": scene["circle1"]["radius"],
            "circle2Radius:": scene["circle2"]["radius"],
            "circle3Radius:": scene["circle3"]["radius"],
            "f": scene["f"],
            "y_rotation": scene["y_rotation"],
            "offset": scene["offset"],
            "True homography": image["h_true"],
            "Warped conic 1": image["C_img"]["C1"],
            "Warped conic 2": image["C_img"]["C2"],
            "Warped conic 3": image["C_img"]["C3"],
            "rectified homography": homography["H"],
            "rectified conic 1": conics["C1"],
            "rectified conic 2": conics["C2"],
            "rectified conic 3": conics["C3"],
            "angle_loss": losses["angle_loss"],
            "circle_loss": losses["circle_loss"],
            "frob_loss": losses["frob_loss"],
            "linf_loss": losses["linf_loss"]
        }
    
    def json_to_df(self, path):
        """
        Create a pandas dataframe from the JSON files with our results and save it to a CSV file.
        """
        json_sources = [
        ("scene", path + "/sceneDescription.json"),
        ("image", path + "/sceneImage.json"),
        ("homography", path + "/rectifiedHomography.json"),
        ("conics", path + "/rectifiedConics.json"),
        ("losses", path + "/losses.json")
        ]
        loaded_data = {}
        for col_name, file_path in json_sources:
            with open(file_path, "r") as f:
                loaded_data[col_name] = json.load(f)
        
        rows = []
        for scene_json, image_json, homography_json, conic_json, losses_json in zip(loaded_data['scene'], loaded_data['image'], loaded_data['homography'], loaded_data['conics'], loaded_data['losses']):
            row = self.extractFields(scene_json, image_json, homography_json, conic_json, losses_json)
            rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(rows)
        print(df.head())  # Display the first few rows of the DataFrame

        # Save the DataFrame to a CSV file if needed
        df.to_csv(path + '/dataframe.csv', index=False)

    def sceneGeneration(self, path):
        """
        Generate the scene images from the scene descriptions and save them to a JSON file.
        """
        # Load all scene descriptions from file

        with open(path + '/sceneDescription.json', 'r') as file:
            data = json.load(file)  # list of sceneDescription JSON objects

        all_images = []

        for scene_json in tqdm(data, desc="Generating scenes"):
            scene_description = SceneDescription.from_json(scene_json)
            all_images.append(self.scene_generator.generate_scene(scene_description).to_json())

        # Write all images to a single JSON file
        with open(path + '/sceneImage.json', 'w') as file:
            json.dump(all_images, file, indent=4)

    def warpConics(self, path):
        """
        Warp the conics using the reconstructed homography matrix for all the scenes.
        """
        with open(path + "/sceneImage.json", "r") as file:
            data = json.load(file)

        with open(path + "/rectifiedHomography.json", "r") as file:
            rectified_homographies = json.load(file)
        rectified_conics = []

        for img_json, H_json in tqdm(zip(data, rectified_homographies), total=len(data), desc="Warping conics"):
            img = Img.from_json(img_json)
            C_img = img.C_img
            H = Homography.from_json(H_json)
            rectified_conics.append(self.conic_warper.warpConics(C_img, H).to_json())

        with open(path + '/rectifiedConics.json', 'w') as file:
            json.dump([conic for conic in rectified_conics], file, indent=4)
    
    def rectify(self, path, rectifier: str):
        """
        Rectify the images using the specified rectifier.
        """
        rectifier = self.rectifiers.get(rectifier)
        with open(path + "/sceneImage.json", "r") as file:
            data = json.load(file)  # This is now a list of Img JSONs

        rectified_homographies = []

        for img_json in tqdm(data, desc="Rectifying scenes"):
            img = Img.from_json(img_json)
            C_img = img.C_img
            H = rectifier.rectify(C_img)
            rectified_homographies.append(H.to_json())
        
        with open(path + '/rectifiedHomography.json', 'w') as file:
            json.dump(rectified_homographies, file, indent=4)


    def generateSceneDescriptionJson(self, sceneDescription, path):
        """"
        Generate a JSON file with all combinations of the parameters.
        """
        (
            f_range,
            y_rotation_range,
            offset_range,
            circle1_centers, circle1_radii,
            circle2_centers, circle2_radii,
            circle3_centers, circle3_radii,
            rectifier
        ) = sceneDescription

        combinations = product(
            f_range,
            y_rotation_range,
            offset_range,
            circle1_centers, circle1_radii,
            circle2_centers, circle2_radii,
            circle3_centers, circle3_radii
        )

        scenes = []
        for (
            f, y_rot, offset,
            c1_center, c1_radius,
            c2_center, c2_radius,
            c3_center, c3_radius
        ) in combinations:

            circle1 = Circle(np.array(c1_center), c1_radius)
            circle2 = Circle(np.array(c2_center), c2_radius)
            circle3 = Circle(np.array(c3_center), c3_radius)

            scene = SceneDescription(
                f, y_rot, np.array(offset),
                circle1, circle2, circle3
            )
            scenes.append(scene.to_json())
        output_path = f"{path}/sceneDescription.json"
        with open(output_path, 'w') as file:
            json.dump(scenes, file, indent=4)


    def parse_range(self, range_list):
        """
        Parse a range list from the JSON file.

        Raises:
            ValueError: If step is 0.
        """
        start, stop, step = range_list
        if step == 0 and start != stop:
            raise ValueError(f"Invalid range: {range_list} (step cannot be 0)")
        else:
            step = 1
        return np.arange(start, stop + step, step).tolist()


    def sceneDefinition(self, path, path_scene) -> tuple:
        """
        Define the scene parameters for the experiment.
        """
        with open(path + path_scene, 'r') as file:
            data = json.load(file)
        # Scalar ranges
        f_range = self.parse_range(data["f_range"])
        y_rotation_range = self.parse_range(data["y_rotation_range"])

        # Offset 3D range
        offset_range_x = self.parse_range(data["offset_range_x"])
        offset_range_y = self.parse_range(data["offset_range_y"])
        offset_range_z = self.parse_range(data["offset_range_z"])
        offset_range = [tuple(map(float, p)) for p in product(offset_range_x, offset_range_y, offset_range_z)]

        # Circle 1
        c1_centre_x_range = self.parse_range(data["c1_centre_x_range"])
        c1_centre_y_range = self.parse_range(data["c1_centre_y_range"])
        c1_radius_range = self.parse_range(data["c1_radius_range"])
        circle1_centers = [tuple(map(float, p)) for p in product(c1_centre_x_range, c1_centre_y_range)]

        # Circle 2
        c2_centre_x_range = self.parse_range(data["c2_centre_x_range"])
        c2_centre_y_range = self.parse_range(data["c2_centre_y_range"])
        c2_radius_range = self.parse_range(data["c2_radius_range"])
        circle2_centers = [tuple(map(float, p)) for p in product(c2_centre_x_range, c2_centre_y_range)]


        # Circle 3
        c3_centre_x_range = self.parse_range(data["c3_centre_x_range"])
        c3_centre_y_range = self.parse_range(data["c3_centre_y_range"])
        c3_radius_range = self.parse_range(data["c3_radius_range"])
        circle3_centers = [tuple(map(float, p)) for p in product(c3_centre_x_range, c3_centre_y_range)]

        rectifier = "homotopy"  # or "standard"

        return (
            f_range,
            y_rotation_range,
            offset_range,
            circle1_centers, c1_radius_range,
            circle2_centers, c2_radius_range,
            circle3_centers, c3_radius_range,
            rectifier
            )

    

if __name__ == "__main__":
    experiment_handler = ExperimentHandler()
    path = "src/HomoTopiContinuation/Data/"  # Replace with your actual path
    experiment_handler.runExperiment(path, "/scene.json")