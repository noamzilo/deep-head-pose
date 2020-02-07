from hopenet_estimator.HopenetEstimatorImages import HopenetEstimatorImages
from Utils.yaml_utils.ConfigParser import ConfigParser

import pandas as pd
import os
from compare_output_to_ground.compare_output_to_ground import CompareOutputToGround

from scipy.spatial.transform import Rotation as R
import numpy as np

import csv
from Utils.path_utils import path_leaf


def compare(self):
    pd.testing.assert_frame_equal(self._ground_files_df[['file name']], self._results_files_df[['file name']])
    expected = self._ground_truth
    actual = self._results

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

        valid_truth1 = pd.read_csv(config.valid_set1_csv_path,
                                  sep=r'\s*,\s*',
                                  header=0,
                                  encoding='ascii',
                                  engine='python')[['file name', 'rx', 'ry', 'rz']]
        valid_truth2 = pd.read_csv(config.valid_set2_csv_path,
                                  sep=r'\s*,\s*',
                                  header=0,
                                  encoding='ascii',
                                  engine='python')[['file name', 'rx', 'ry', 'rz']]

        abs_paths1 = [os.path.join(images_folder_path1, path) for path in valid_truth1['file name']]
        abs_paths2 = [os.path.join(images_folder_path2, path) for path in valid_truth2['file name']]

        images_full_paths = abs_paths1 + abs_paths2
        for p in images_full_paths:
            assert os.path.isfile(p)

        hopenet_estimator = HopenetEstimatorImages(config,
                                                   config,
                                                   abs_paths1,
                                                   )

        results = hopenet_estimator.calculate_results()
        write_results_to_csv(results, config.output_dir, config.output_file_name1)

        comparer = CompareOutputToGround(config, config, config)
        comparer.compare()

    main()
