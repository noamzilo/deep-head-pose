from test_on_validation.validation_requirements.ValidationSetLoader import ValidationSetLoader
from Utils.yaml_utils.ConfigParser import ConfigParser
import os
import pandas as pd


class CompareOutputToGround(object):
    def __init__(self, validation_config, test_config, hopenet_config):
        self._validation_config = validation_config
        self._test_config = test_config
        self._hopenet_config = hopenet_config

        self._extract_rxyz_from_ground()
        self._extract_yaw_pitch_roll_from_results()

    def compare(self):
        assert self._gound_files_df == self._results_files_df

    def _extract_rxyz_from_ground(self):
        self._ground_truth_path = self._validation_config.ground_truth_file_path
        assert os.path.isfile(self._ground_truth_path)

        self._ground_truth_df = pd.read_csv(self._ground_truth_path,
                                            sep=r'\s*,\s*',
                                            header=0,
                                            encoding='ascii',
                                            engine='python')

        self._ground_truth = self._ground_truth_df[['rx', 'ry', 'rz']].to_numpy()
        self._gound_files_df = self._ground_truth_df[['file name']]

    def _extract_yaw_pitch_roll_from_results(self):
        self.results_file_path = os.path.join(self._hopenet_config.output_dir, self._test_config.output_file_name)
        assert os.path.isfile(self.results_file_path)

        self._results_df = pd.read_csv(self.results_file_path,
                                       sep=r'\s*,\s*',
                                       header=0,
                                       encoding='ascii',
                                       engine='python')

        self._results = self._results_df[['rx', 'ry', 'rz']].to_numpy()
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
