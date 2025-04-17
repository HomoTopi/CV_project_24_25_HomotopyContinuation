import numpy as np
from itertools import product
from HomoTopiContinuation.ExperimentHandler.ExperimentHandler import ExperimentHandler


def main():
    path = "src/HomoTopiContinuation/Data/"
    experimentHandler = ExperimentHandler()
    experimentHandler.runExperiment(path, "/scene.json")

if __name__ == "__main__":
    main()







