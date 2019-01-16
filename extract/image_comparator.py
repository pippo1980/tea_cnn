import os
import numpy as np

import extract.comparator as comparator


def load_feature_dict():
    features_root = os.path.join(os.getcwd(), "../features")

    feature_dict = {}
    for file in os.listdir(features_root + "/20190112"):
        with open(features_root + "/20190112/" + file) as f:
            feature_dict[file] = np.array(f.readline().split(","), "float64")

    return feature_dict


if __name__ == '__main__':
    # example_img = "C-MPX0001-153214.jpg"
    example_img = "C-MPX0021-170950.jpg"
    example_features = list()

    # 一个茶饼的36个特征
    i = 0
    while i < 5:
        i = i + 1
        j = 0
        while j < 5:
            j = j + 1
            example_features.append(example_img + "_" + str(i) + "_" + str(j) + ".jpg.txt")

    f_dict = load_feature_dict()

    result = {}
    for name in example_features:
        example_f = f_dict[name]

        one_example_f_result = {}
        for key, val in f_dict.items():
            one_example_f_result[comparator.calculate_distance(example_f, val, "euclidean")] = key

        # for key in sorted(one_example_f_result.keys()):
        #     print(one_example_f_result[key], ":", key)

        result[name] = one_example_f_result

    for key, val in result.items():
        i = 0
        for ff in sorted(val.keys()):
            i = i + 1
            if i > 10:
                break

            print(key, val[ff], ff)
