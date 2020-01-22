import numpy as np
import os
import dlib
from original_code_augmented.test_on_validation.validation_requirements import ThreeD_Model
import cv2
from original_code_augmented.test_on_validation.validation_requirements.ValidationSetLoader import ValidationSetLoader
from Utils.yaml_utils.ConfigParser import ConfigParser
from original_code_augmented.test_on_validation.Validator import Validator


class PnpEstimatorManualTags(object):
    def __init__(self, config_path):
        self._paths = ConfigParser(config_path).parse()
        self._model_full_path = self._paths.model_full_path
        self._ground_truth_file_path = self._paths.ground_truth_file_path
        self._validation_images_folder_path = self._paths.validation_images_folder_path

        self._load_validation_set()

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

    def calculate_results(self):
        # initialize dlib's face detector and create facial landmark predictor
        results_list = []
        model3d = self._preload_3d_model()
        for i, image_path in enumerate(self._validation_image_paths):
            success, rotation_vec, translation_vec = cv2.solvePnP(model3d.model_TD,
                                                                  self._validation_points[i],
                                                                  model3d.out_A,
                                                                  None,
                                                                  None,
                                                                  None,
                                                                  False)

            results_list.append(np.append(rotation_vec, translation_vec))
        return results_list


if __name__ == "__main__":
    def main():
        config_path = r"C:\Noam\Code\vision_course\hopenet\deep-head-pose\code\original_code_augmented\config\paths.yaml"
        pnp_estimator = PnpEstimatorManualTags(config_path)
        original_validator = Validator(config_path, pnp_estimator)
        original_validator.validate()

    main()
