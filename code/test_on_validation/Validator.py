import numpy as np
from test_on_validation.validation_requirements import ThreeD_Model
from test_on_validation.validation_requirements.ValidationSetLoader import ValidationSetLoader
from Utils.yaml_utils.ConfigParser import ConfigParser
import os


class Validator(object):
    def __init__(self, validation_config, results_calculator, validation_set_loader):
        self._validation_config = validation_config
        self._model_full_path = self._validation_config.model_full_path
        self._ground_truth_file_path = self._validation_config.ground_truth_file_path
        self._validation_images_folder_path = self._validation_config.validation_images_folder_path
        self._predictor_path = self._validation_config.predictor_path

        self._validation_set_loader = validation_set_loader
        self._ground_truth = self._validation_set_loader.ground_truth

        # self._load_validation_set()
        self._results_calculator = results_calculator

    # def _load_validation_set(self):
    #
    #     self._validation_set_loader.load_validation_set()
    #     self._ground_truth = self._validation_set_loader.ground_truth
    #     self._validation_image_paths = self._validation_set_loader.validation_image_paths
    #     self._validation_points = self._validation_set_loader.validation_points
    #
    # def _create_image_paths_file(self):
    #     paths = self._validation_image_paths
    #     paths = sorted(paths)
    #
    #     create_file_at = os.path.join(self._validation_config.create_rel_paths_file_at, self._validation_config.rel_paths_file_name)
    #     if os.path.isfile(create_file_at):
    #         os.remove(create_file_at)
    #
    #     with open(create_file_at, 'w') as f:
    #         for path in paths:
    #             f.write(f"{path}\n")
    #
    #     return create_file_at, paths

    def compare_result_to_ground(self, results_list):
        validation_df = self._ground_truth
        ground_rx_ry_rz = np.array(validation_df[['rx', 'ry', 'rz']])
        results_rx_ry_rz = [r[0:3] for _, r in results_list]
        image_paths = [path for path, _ in results_list]
        results_rx_ry_rz = np.array(results_rx_ry_rz)
        pic_names = validation_df['file name'].values

        # np.testing.assert_almost_equal(ground_rx_ry_rz, results_rx_ry_rz)
        # The above assert was supposed to work, but the validation tags seem incorrect, while the model seems fine.
        for i, (image_path, (x, y, z)) in enumerate(zip(image_paths, results_rx_ry_rz)):
            print(f"comparing {i}, at {image_path}")
            np.testing.assert_almost_equal(ground_rx_ry_rz[i, :], np.array([x, y, z]))

    def validate(self):
        # results_list = self._calculate_original_results()
        results_list = self._results_calculator.calculate_results()

        self.compare_result_to_ground(results_list)


