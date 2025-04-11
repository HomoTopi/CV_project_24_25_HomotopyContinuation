import pandas as pd
import json

def extractFields(scene, image, homography, conics, losses):
    """
    Extract interesting fields from the JSON data.
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

def json_to_df() -> None:
    """
    Create a pandas dataframe from the JSON files with our results.
    """
    json_sources = [
    ("scene", "src\\HomoTopiContinuation\\Data\\sceneDescription.json"),
    ("image", "src\\HomoTopiContinuation\\Data\\sceneImage.json"),
    ("homography", "src\\HomoTopiContinuation\\Data\\rectifiedHomography.json"),
    ("conics", "src\\HomoTopiContinuation\\Data\\rectifiedConics.json"),
    ("losses", "src\\HomoTopiContinuation\\Data\\losses.json")
    ]
    loaded_data = {}
    for col_name, file_path in json_sources:
        with open(file_path, "r") as f:
            loaded_data[col_name] = json.load(f)
    
    rows = []
    for scene_json, image_json, homography_json, conic_json, losses_json in zip(loaded_data['scene'], loaded_data['image'], loaded_data['homography'], loaded_data['conics'], loaded_data['losses']):
        row = extractFields(scene_json, image_json, homography_json, conic_json, losses_json)
        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)
    print(df.head())  # Display the first few rows of the DataFrame

    # Save the DataFrame to a CSV file if needed
    df.to_csv('src\\HomoTopiContinuation\\Data\\dataframe.csv', index=False)
   

if __name__ == "__main__":
    json_to_df()