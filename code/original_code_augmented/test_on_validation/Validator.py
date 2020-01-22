import numpy as np
import os
import dlib
from original_code_augmented.test_on_validation.validation_requirements import ThreeD_Model
import cv2
from original_code_augmented.test_on_validation.validation_requirements.ValidationSetLoader import ValidationSetLoader
from Utils.yaml_utils.ConfigParser import ConfigParser


class Validator(object):
    def __init__(self, config_path, results_calculator):
        self._paths = ConfigParser(config_path).parse()
        self._model_full_path = self._paths.model_full_path
        self._ground_truth_file_path = self._paths.ground_truth_file_path
        self._validation_images_folder_path = self._paths.validation_images_folder_path
        self._predictor_path = self._paths.predictor_path

        self._load_validation_set()
        self._results_calculator = results_calculator

    def _load_validation_set(self):
        self._validation_set_loader = ValidationSetLoader(self._ground_truth_file_path,
                                                          self._validation_images_folder_path)
        self._validation_set_loader.load_validation_set()
        self._ground_truth = self._validation_set_loader.ground_truth
        self._validation_image_paths = self._validation_set_loader.validation_image_paths
        self._validation_points = self._validation_set_loader.validation_points

    def _preload_3d_model(self):
        model3d = ThreeD_Model.FaceModel(self._model_full_path, 'model3D', True)
        return model3d

    @staticmethod
    def shape_to_np(shape, dtype="int"):
        coords = np.zeros((68, 2), dtype=dtype)

        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        return coords

    def compare_result_to_ground(self, results_list):
        validation_df = self._ground_truth
        ground_rx_ry_rz = np.array(validation_df[['rx', 'ry', 'rz']])
        results_rx_ry_rz = [r[0:3] for r in results_list]
        results_rx_ry_rz = np.array(results_rx_ry_rz)
        pic_names = validation_df['file name'].values

        np.testing.assert_almost_equal(ground_rx_ry_rz, results_rx_ry_rz)
        # The above assert was supposed to work, but the validation tags seem incorrect, while the model seems fine.

    def validate(self):
        # results_list = self._calculate_original_results()
        results_list = self._results_calculator.calculate_results()

        self.compare_result_to_ground(results_list)


