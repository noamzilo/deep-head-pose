import numpy as np
from test_on_validation.validation_requirements import ThreeD_Model
import cv2
from test_on_validation.validation_requirements.ValidationSetLoader import ValidationSetLoader
from Utils.yaml_utils.ConfigParser import ConfigParser
from test_on_validation.Validator import Validator


class PnpEstimatorManualTags(object):
    def __init__(self, validation_config):
        self._validation_config = validation_config
        self._model_full_path = self._validation_config.model_full_path
        self._ground_truth_file_path = self._validation_config.ground_truth_file_path
        self._validation_images_folder_path = self._validation_config.validation_images_folder_path

        self._load_validation_set()

    def _load_validation_set(self):
        self.validation_set_loader = ValidationSetLoader(self._validation_config)
        self._ground_truth = self.validation_set_loader.ground_truth
        self._validation_image_paths = self.validation_set_loader.validation_image_paths
        self._validation_points = self.validation_set_loader.validation_points

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

            results_list.append((image_path, np.append(rotation_vec, translation_vec)))
        return results_list


if __name__ == "__main__":
    def main():
        validation_config_path = r"C:\Noam\Code\vision_course\hopenet\deep-head-pose\code\config\validation_config.yaml"
        validation_config = ConfigParser(validation_config_path).parse()
        pnp_estimator = PnpEstimatorManualTags(validation_config)
        original_validator = Validator(validation_config, pnp_estimator, pnp_estimator.validation_set_loader)
        original_validator.validate()

    main()
