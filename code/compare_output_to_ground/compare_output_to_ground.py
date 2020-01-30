from test_on_validation.validation_requirements.ValidationSetLoader import ValidationSetLoader
from Utils.yaml_utils.ConfigParser import ConfigParser
import os
import pandas as pd
from scipy.spatial.transform import Rotation as R
import numpy as np


class CompareOutputToGround(object):
    def __init__(self, validation_config, test_config, hopenet_config):
        self._validation_config = validation_config
        self._test_config = test_config
        self._hopenet_config = hopenet_config

        self._extract_rxyz_from_ground()
        self._extract_yaw_pitch_roll_from_results()

    def compare(self):
        pd.testing.assert_frame_equal(self._ground_files_df[['file name']], self._results_files_df[['file name']])
        expected = self._ground_truth
        actual = self._results

        # doing the conversion in the original
        def convert(row):
            roll, pitch, yaw = row[0], row[1], row[2]
            r = R.from_euler('zxy', (roll, pitch, yaw), degrees=True)
            # r = R.from_euler('xyz', (roll, pitch, yaw), degrees=True)
            # print(r.as_rotvec())
            return r.as_rotvec()

        for i, row in actual.iterrows():
            rotvec = convert(row)
            rx, ry, rz = rotvec
            actual['rx'][i] = rx
            actual['ry'][i] = ry
            actual['rz'][i] = rz

        # for i, row in expected.iterrows():
        #     rx, ry, rz = row
        #     actual['rx'][i] = rx
        #     actual['ry'][i] = -ry
        #     actual['rz'][i] = rz

        # calculate angles between actual and expected
        diffs = []
        angles = []
        for (i, row_actual), (_, row_expected) in zip(actual.iterrows(), expected.iterrows()):
            a = R.from_rotvec(row_actual)
            e = R.from_rotvec(row_expected)
            # e = R.from_rotvec((row_expected[0], -row_expected[1], row_expected[2]))

            a_mat = a.as_matrix()
            e_mat = e.as_matrix()

            # angle_rad = np.arccos((np.trace(a_mat.T @ e_mat) - 1) / 2)
            angle_rad = np.arccos((np.trace(a_mat.T @ e_mat) - 1) / 2)
            angle_deg = np.rad2deg(angle_rad)
            angle_deg = np.min([angle_deg, 180 - angle_deg])
            angles.append(angle_deg)

            # diff = a * e.inv()
            # diffs.append(diff.as_matrix())
            # angles = [np.rad2deg(np.arccos((np.trace(diff) - 1) / 2)) for diff in diffs]

        print(f"max: {np.max(angles)}")
        print(f"argmax: {np.argmax(angles)}")
        print(f"mean: {np.mean(angles)}")
        return np.array(angles)

    def _extract_rxyz_from_ground(self):
        self._ground_truth_path = self._validation_config.ground_truth_file_path
        assert os.path.isfile(self._ground_truth_path)

        self._ground_truth_df = pd.read_csv(self._ground_truth_path,
                                            sep=r'\s*,\s*',
                                            header=0,
                                            encoding='ascii',
                                            engine='python')
        self._ground_truth_df.sort_values(by="file name", inplace=True)
        self._ground_truth_df.reset_index(drop=True, inplace=True)

        self._ground_truth = self._ground_truth_df[['rx', 'ry', 'rz']]
        # self._ground_truth.loc[:, 'ry'] = -self._ground_truth_df[['ry']]
        self._ground_files_df = self._ground_truth_df[['file name']]

    def _extract_yaw_pitch_roll_from_results(self):
        self.results_file_path = os.path.join(self._hopenet_config.output_dir, self._test_config.output_file_name)
        assert os.path.isfile(self.results_file_path)

        self._results_df = pd.read_csv(self.results_file_path,
                                       sep=r'\s*,\s*',
                                       header=0,
                                       encoding='ascii',
                                       engine='python')
        self._results_df.sort_values(by="file name", inplace=True)
        self._results_df.reset_index(drop=True, inplace=True)

        self._results = self._results_df[['rx', 'ry', 'rz']]
        self._results_files_df = self._results_df[['file name']]


if __name__ == "__main__":
    def main():
        validation_config_path = r"C:\Noam\Code\vision_course\hopenet\deep-head-pose\code\config\validation_config.yaml"
        validation_config = ConfigParser(validation_config_path).parse()
        test_config_path = r"C:\Noam\Code\vision_course\hopenet\deep-head-pose\code\config\test_config.yaml"
        test_config = ConfigParser(test_config_path).parse()
        hopenet_config_path = r"C:\Noam\Code\vision_course\hopenet\deep-head-pose\code\config\hopenet_config.yaml"
        hopenet_config = ConfigParser(hopenet_config_path).parse()
        comparer = CompareOutputToGround(validation_config, test_config, hopenet_config)
        comparer.compare()

    main()
