import os
import numpy as np
from scipy.spatial import distance


def calculate_distance(source, target, distance_calculator="hamming"):
    if distance_calculator is "hamming":
        return distance.hamming(source, target)
    elif distance_calculator is "euclidean":
        # return numpy.linalg.norm(a - b)
        return distance.euclidean(source, target)
    elif distance_calculator is "cosine":
        return distance.cosine(source, target)
    else:
        return 0


if __name__ == '__main__':
    features_root = os.path.join(os.getcwd(), "../features")

    feature_dict = {}
    for file in os.listdir(features_root + "/20190112"):
        with open(features_root + "/20190112/" + file) as f:
            feature_dict[file] = np.array(f.readline().split(","), "float64")

    example_key = "C-MPX0013-170123.jpg_2_2.jpg.txt"
    example_val = feature_dict[example_key]

    result_dict = {}
    for key, val in feature_dict.items():
        # if key == example_key:
        #     continue
        result_dict[calculate_distance(example_val, val, "cosine")] = key

    for key in sorted(result_dict.keys()):
        print(result_dict[key], ":", key)
