import numpy as np
import os
import dlib
from original_code_augmented.test_on_validation.validation_requirements import ThreeD_Model
import cv2
from original_code_augmented.test_on_validation.validation_requirements.ValidationSetLoader import ValidationSetLoader


class OriginalValidator(object):
    def __init__(self):
        self._model_full_path = \
            r'C:\\Noam\Code\vision_course\face_specific_augm/model_lecturer/model3D_aug_-00_00_01.mat'
        self._ground_truth_file_path = r"C:\Noam\Code\vision_course\face_pose_estimation\images\valid_set\validation_set.csv"
        self._validation_images_folder_path = r"C:\Noam\Code\vision_course\face_pose_estimation\images\valid_set\images"
        self._predictor_path = r"C:\Noam\Code\vision_course\shape_predictor\shape_predictor_68_face_landmarks.dat"
        # self._read_ground_truth_validation()

        self._validation_set_loader = ValidationSetLoader(self._ground_truth_file_path,
                                                          self._validation_images_folder_path)
        self._validation_set_loader.load_validation_set()

    def _preload(self):
        model3d = ThreeD_Model.FaceModel(self._model_full_path, 'model3D', True)
        return model3d

    @staticmethod
    def shape_to_np(shape, dtype="int"):
        coords = np.zeros((68, 2), dtype=dtype)

        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        return coords

    def compare_result_to_ground(self, results_list):
        validation_df = self._validation_set_loader.ground_truth
        ground_rx_ry_rz = np.array(validation_df[['rx', 'ry', 'rz']])
        results_rx_ry_rz = [r[0:3] for r in results_list]
        results_rx_ry_rz = np.array(results_rx_ry_rz)
        pic_names = validation_df['file name'].values

        np.testing.assert_almost_equal(ground_rx_ry_rz, results_rx_ry_rz)
        # The above assert was supposed to work, but the validation tags seem incorrect, while the model seems fine.

    def validate(self):
        # initialize dlib's face detector and create facial landmark predictor
        detector = dlib.get_frontal_face_detector()
        predictor_path = self._predictor_path
        assert os.path.isfile(predictor_path)
        predictor = dlib.shape_predictor(predictor_path)

        results_list = []

        model3d = self._preload()

        for i, image_path in enumerate(self._validation_set_loader.validation_image_paths):
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
                                                                  self._validation_set_loader.validation_points[i],
                                                                  model3d.out_A,
                                                                  None,
                                                                  None,
                                                                  None,
                                                                  False)

            results_list.append(np.append(rotation_vec, translation_vec))

        self.compare_result_to_ground(results_list)


if __name__ == "__main__":
    def main():
        original_validator = OriginalValidator()
        original_validator.validate()

    main()
