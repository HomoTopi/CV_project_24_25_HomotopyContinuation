import json 
import numpy as np
from HomoTopiContinuation.DataStructures.datastructures  import Circle, SceneDescription

def sceneDefinition():
    f= 1
    y_rotation  = np.array([0,50,1]) #start, end, step
    offset = np.array([0,0,2])
    circle1 = Circle(np.array([0,0]),1)
    circle2 = Circle(np.array([0.5,0]),1)
    circle3 = Circle(np.array([0,0]),1.5)
    return f, y_rotation, offset, circle1, circle2, circle3

def main():
    f, y_rotation, offset, circle1, circle2, circle3 = sceneDefinition()

    scenes = []

    for i in range(y_rotation[0], y_rotation[1], y_rotation[2]):
        scene = SceneDescription(f, i, offset, circle1, circle2, circle3)
        scenes.append(scene.to_json())

    with open('src/HomoTopiContinuation/Data/sceneDescription.json', 'w') as file:
        json.dump(scenes, file, indent=4)
        
    return

if __name__ == "__main__":
    main()