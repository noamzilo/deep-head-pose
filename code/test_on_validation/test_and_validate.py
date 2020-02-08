from hopenet_estimator.HopenetEstimatorImages import HopenetEstimatorImages
from Utils.yaml_utils.ConfigParser import ConfigParser

import pandas as pd
import os
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

    angles_max = np.max(angles)
    angles_argmax = np.argmax(angles)
    angles_mean = np.mean(angles)

    print(f"max: {angles_max}")
    print(f"argmax: {angles_argmax}")
    print(f"mean: {angles_mean}")
    return np.array(angles), angles_max, angles_argmax, angles_mean


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
        snapshot_folder = config.snapshots_folder
        snapshot_name = config.snapshot_name
        # snapshot_path = config.snapshot_path
        means = []
        maxes = []
        # for snapshot_num in range(1, 30):
        for snapshot_num in range(1, 26):
            snapshot_path = os.path.join(snapshot_folder, snapshot_name + f"{snapshot_num}.pkl")

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
                                                       snapshot_path)

            results1 = hopenet_estimator.calculate_results()
            write_results_to_csv(results1, config.output_dir, config.output_file_name1)
            results1_path = os.path.join(config.output_dir, config.output_file_name1)
            results1_df = pd.read_csv(results1_path,
                                      sep=r'\s*,\s*',
                                      header=0,
                                      encoding='ascii',
                                      engine='python')[['file name', 'rx', 'ry', 'rz']]

            angles1, angles_max1, angles_argmax1, angles_mean1 = compare(valid_truth1, results1_df)


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
                                                       snapshot_path)

            results2 = hopenet_estimator.calculate_results()
            write_results_to_csv(results2, config.output_dir, config.output_file_name2)
            results2_path = os.path.join(config.output_dir, config.output_file_name2)
            results2_df = pd.read_csv(results2_path,
                                      sep=r'\s*,\s*',
                                      header=0,
                                      encoding='ascii',
                                      engine='python')[['file name', 'rx', 'ry', 'rz']]

            angles2, angles_max2, angles_argmax2, angles_mean2 = compare(valid_truth2, results2_df)


            #########################################################################################
            total_max = max(angles_max1, angles_max2)
            total_mean = (len(angles1) * angles_mean1 + len(angles2) * angles_mean2) / (len(angles1) + len(angles2))

            print(f"total_max snapshot #{snapshot_num}: {total_max}")
            print(f"total_mean snapshot #{snapshot_num}: {total_mean}")
            maxes.append(total_max)
            means.append(total_mean)

        best_mean_epoch = int(np.argmin(means))
        best_max_epoch = int(np.argmin(maxes))

        print(f"best max epoch: {best_max_epoch} with max: {maxes[best_max_epoch]}")
        print(f"best max epoch: {best_mean_epoch} with mean: {means[best_mean_epoch]}")

        validation_error_per_epoch_file_name = "validation_error_per_epoch_cloud.csv"
        with open(validation_error_per_epoch_file_name, "w", newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for i, (mean, mx) in enumerate(zip(means, maxes)):
                writer.writerow([i + 1] + [mean] + [mx])


    main()
