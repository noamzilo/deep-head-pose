import numpy as np
import os
import dlib
import cv2
from test_on_validation import ValidationSetLoader
from Utils.yaml_utils.ConfigParser import ConfigParser


class HopenetValidator(object):
    def __init__(self, config_path):
        self._paths = ConfigParser(config_path).parse()
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

    def compare_result_to_ground(self, results_list):
        validation_df = self._ground_truth
        ground_rx_ry_rz = np.array(validation_df[['rx', 'ry', 'rz']])
        results_rx_ry_rz = [r[0:3] for r in results_list]
        results_rx_ry_rz = np.array(results_rx_ry_rz)
        pic_names = validation_df['file name'].values

        np.testing.assert_almost_equal(ground_rx_ry_rz, results_rx_ry_rz)
        # The above assert was supposed to work, but the validation tags seem incorrect, while the model seems fine.

    def validate(self):
        # TODO make this validate using "test hopenet"
        # TODO output images with annotation
        # TODO look at images manually, and get some numeric result per angle axis.

        # initialize dlib's face detector and create facial landmark predictor
        detector = dlib.get_frontal_face_detector()
        predictor_path = self._predictor_path
        assert os.path.isfile(predictor_path)
        predictor = dlib.shape_predictor(predictor_path)

        results_list = []

        model3d = self._preload_3d_model()

        for i, image_path in enumerate(self._validation_image_paths):
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            rects = detector(gray, 1)

            # for i, rect in enumerate(rects):
            #     # determine thefacial landmarks for the face region then convert the landmarks to x,y np array
            #     shape = predictor(gray, rect)
            #     shape = shape_to_np(shape)
            #
            #     success, rotation_vec, translation_vec = cv2.solvePnP(model3d.model_TD,
            #                                                           validation_points[i],
            #                                                           model3d.out_A,
            #                                                           None,
            #                                                           None,
            #                                                           None,
            #                                                           False)
            #
            #     results_list.append(np.append(rotation_vec, translation_vec))

            success, rotation_vec, translation_vec = cv2.solvePnP(model3d.model_TD,
                                                                  self._validation_points[i],
                                                                  model3d.out_A,
                                                                  None,
                                                                  None,
                                                                  None,
                                                                  False)

            results_list.append(np.append(rotation_vec, translation_vec))

        self.compare_result_to_ground(results_list)


if __name__ == "__main__":
    def main():
        config_path = r"C:\Noam\Code\vision_course\hopenet\deep-head-pose\code\original_code_augmented\config\paths.yaml"
        original_validator = OriginalValidator(config_path)
        original_validator.validate()

    main()
