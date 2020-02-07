from hopenet_estimator.HopenetEstimatorImages import HopenetEstimatorImages
from Utils.yaml_utils.ConfigParser import ConfigParser

import pandas as pd
import os
from compare_output_to_ground.compare_output_to_ground import CompareOutputToGround

from scipy.spatial.transform import Rotation as R
import numpy as np

import csv
from Utils.path_utils import path_leaf


def compare(ground_df, results_df):
    pd.testing.assert_frame_equal(ground_df[['file name']], results_df[['file name']])
    expected = ground_df[['rx', 'ry', 'rz']]
    actual = results_df[['rx', 'ry', 'rz']]

    angles = []
    for (i, row_actual), (_, row_expected) in zip(actual.iterrows(), expected.iterrows()):
        a = R.from_rotvec(row_actual)
        e = R.from_rotvec(row_expected)

        a_mat = a.as_matrix()
        e_mat = e.as_matrix()

        angle_rad = np.arccos((np.trace(a_mat.T @ e_mat) - 1) / 2)
        angle_deg = np.rad2deg(angle_rad)
        angle_deg = np.min([angle_deg, 180 - angle_deg])
        angles.append(angle_deg)

    print(f"max: {np.max(angles)}")
    print(f"argmax: {np.argmax(angles)}")
    print(f"mean: {np.mean(angles)}")
    return np.array(angles)


def write_results_to_csv(results, output_dir, output_file_name):
    assert os.path.isdir(output_dir)
    output_full_path = os.path.join(output_dir, output_file_name)

    with open(output_full_path, "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(",file name,rx,ry,rz,tx,ty,tz".split(','))
        for i, (file_name, line) in enumerate(results):
            writer.writerow([str(i)] + [path_leaf(file_name)] + list(line))


if __name__ == "__main__":
    def main():
        config = ConfigParser("config.yaml").parse()
        images_folder_path1 = config.test_images_folder1_path
        images_folder_path2 = config.test_images_folder2_path

        valid_set_1_csv_path = config.valid_set1_csv_path

        valid_truth1 = pd.read_csv(valid_set_1_csv_path,
                                  sep=r'\s*,\s*',
                                  header=0,
                                  encoding='ascii',
                                  engine='python')[['file name', 'rx', 'ry', 'rz']]

        abs_paths1 = [os.path.join(images_folder_path1, path) for path in valid_truth1['file name']]

        for p in abs_paths1:
            assert os.path.isfile(p)

        hopenet_estimator = HopenetEstimatorImages(config,
                                                   config,
                                                   abs_paths1,
                                                   )

        results1 = hopenet_estimator.calculate_results()
        write_results_to_csv(results1, config.output_dir, config.output_file_name1)
        results1_path = os.path.join(config.output_dir, config.output_file_name1)
        results1_df = pd.read_csv(results1_path,
                                  sep=r'\s*,\s*',
                                  header=0,
                                  encoding='ascii',
                                  engine='python')[['file name', 'rx', 'ry', 'rz']]

        compare(valid_truth1, results1_df)


        ######################################################################################################

        valid_set_2_csv_path = config.valid_set2_csv_path
        valid_truth2 = pd.read_csv(valid_set_2_csv_path,
                                   sep=r'\s*,\s*',
                                   header=0,
                                   encoding='ascii',
                                   engine='python')[['file name', 'rx', 'ry', 'rz']]
        abs_paths2 = [os.path.join(images_folder_path2, path) for path in valid_truth2['file name']]
        for p in abs_paths2:
            assert os.path.isfile(p)

        hopenet_estimator = HopenetEstimatorImages(config,
                                                   config,
                                                   abs_paths2,
                                                   )

        results2 = hopenet_estimator.calculate_results()
        write_results_to_csv(results2, config.output_dir, config.output_file_name2)
        results2_path = os.path.join(config.output_dir, config.output_file_name2)
        results2_df = pd.read_csv(results2_path,
                                  sep=r'\s*,\s*',
                                  header=0,
                                  encoding='ascii',
                                  engine='python')[['file name', 'rx', 'ry', 'rz']]

        compare(valid_truth2, results2_df)


    main()
