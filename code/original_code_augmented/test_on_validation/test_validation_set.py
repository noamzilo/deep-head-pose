import pandas as pd
import numpy as np
import os
import dlib
from original_code_augmented.test_on_validation.validation_requirements import ThreeD_Model
import cv2


class OriginalValidation(object):
    def __init__(self):
        self._read_ground_truth_validation()

    def preload(self, this_path, pose_models_folder, pose_models, nSub):
        print('> Preloading all the models for efficiency')
        allModels = dict()
        for posee in pose_models:
            for subj in range(1, nSub + 1):
                pose = posee + '_' + str(subj).zfill(2) + '.mat'
                # load detections performed by dlib library on 3D model and Reference Image
                print("> Loading pose model in " + pose)
                # model3D = ThreeD_Model.FaceModel(this_path + "/models3d_new/" + pose, 'model3D')
                if '-00' in posee:
                    model3d = ThreeD_Model.FaceModel(this_path + pose_models_folder + pose, 'model3D', True)
                else:
                    model3d = ThreeD_Model.FaceModel(this_path + pose_models_folder + pose, 'model3D', False)

                allModels[pose] = model3d
        return allModels

    def _read_ground_truth_validation(self, ):
        ground_truth_file_path = r"C:\Noam\Code\vision_course\face_pose_estimation\images\valid_set\validation_set.csv"
        ground_truth = pd.read_csv(ground_truth_file_path,
                                   sep=r'\s*,\s*',
                                   header=0,
                                   encoding='ascii',
                                   engine='python')
        self._ground_truth = ground_truth

        validation_images_filenames = [filename for filename in ground_truth['file name'].values]
        validation_pts_filenames = [filename.replace('.png', '.pts') for filename in validation_images_filenames]
        self._validation_images_filenames = validation_images_filenames
        self._validation_pts_filenames = validation_pts_filenames

        validation_images_folder_path = r"C:\Noam\Code\vision_course\face_pose_estimation\images\valid_set\images"
        validation_image_paths = [os.path.join(validation_images_folder_path, image_file_name) for image_file_name in
                                  validation_images_filenames]
        validation_points = [self._load_landmarks_from_path(validation_images_folder_path, file_name) for
                             file_name in validation_pts_filenames]

        self._validation_image_paths = validation_image_paths
        self._validation_points = validation_points

    def _load_landmarks_from_path(self, dir_path, file_name):
        assert os.path.isdir(dir_path)
        file_path = os.path.abspath(os.path.join(dir_path, file_name))
        assert os.path.isfile(file_path)

        points = np.loadtxt(file_path, comments=("version:", "n_points:", "{", "}"))
        return points


    def shape_to_np(self, shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype=dtype)

        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        # return the list of (x, y)-coordinates
        return coords

    def compare_result_to_ground(self, validation_df, results_list):
        ground_rx_ry_rz = np.array(validation_df[['rx', 'ry', 'rz']])
        results_rx_ry_rz = [r[0:3] for r in results_list]
        results_rx_ry_rz = np.array(results_rx_ry_rz)
        pic_names = validation_df['file name'].values

        np.testing.assert_almost_equal(ground_rx_ry_rz, results_rx_ry_rz)
        # The above assert was supposed to work, but the validation tags seem incorrect, while the model seems fine.

    def validate(self):
        # initialize dlib's face detector and create facial landmark predictor
        detector = dlib.get_frontal_face_detector()
        predictor_path = r"C:\Noam\Code\vision_course\shape_predictor\shape_predictor_68_face_landmarks.dat"
        assert os.path.isfile(predictor_path)
        predictor = dlib.shape_predictor(predictor_path)

        results_list = []
        this_path = r"C:\\Noam\\Code\\vision_course\\face_specific_augm"
        pose_models_folder = r'/model_lecturer/'
        pose_models = [r'model3D_aug_-00_00']
        n_sub = 1
        pose = r'model3D_aug_-00_00_01.mat'
        model3d = self.preload(this_path, pose_models_folder, pose_models, n_sub)[pose]

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

        self.compare_result_to_ground(self._ground_truth, results_list)


if __name__ == "__main__":
    def main():
        original_validator = OriginalValidation()
        # validation_df = original_validator.read_ground_truth_validation()
        # validation_images_filenames = [filename for filename in validation_df['file name'].values]
        # validation_pts_filenames = [filename.replace('.png', '.pts') for filename in validation_images_filenames]

        # validation_images_folder_path = r"C:\Noam\Code\vision_course\face_pose_estimation\images\valid_set\images"
        # validation_image_paths = [os.path.join(validation_images_folder_path, image_file_name) for image_file_name in validation_images_filenames]
        # validation_points = [original_validator.load_landmarks_from_path(validation_images_folder_path, file_name) for file_name in validation_pts_filenames]


        # # initialize dlib's face detector and create facial landmark predictor
        # detector = dlib.get_frontal_face_detector()
        # predictor_path = r"C:\Noam\Code\vision_course\shape_predictor\shape_predictor_68_face_landmarks.dat"
        # assert os.path.isfile(predictor_path)
        # predictor = dlib.shape_predictor(predictor_path)
        #
        # results_list = []
        # this_path = r"C:\\Noam\\Code\\vision_course\\face_specific_augm"
        # pose_models_folder = r'/model_lecturer/'
        # pose_models = [r'model3D_aug_-00_00']
        # n_sub = 1
        # pose = r'model3D_aug_-00_00_01.mat'
        # model3d = original_validator.preload(this_path, pose_models_folder, pose_models, n_sub)[pose]

        # for i, image_path in enumerate(validation_image_paths):
        #     image = cv2.imread(image_path)
        #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #
        #     rects = detector(gray, 1)
        #
        #     # for i, rect in enumerate(rects):
        #     #     # determine thefacial landmarks for the face region then convert the landmarks to x,y np array
        #     #     shape = predictor(gray, rect)
        #     #     shape = shape_to_np(shape)
        #     #
        #     #     success, rotation_vec, translation_vec = cv2.solvePnP(model3d.model_TD,
        #     #                                                           validation_points[i],
        #     #                                                           model3d.out_A,
        #     #                                                           None,
        #     #                                                           None,
        #     #                                                           None,
        #     #                                                           False)
        #     #
        #     #     results_list.append(np.append(rotation_vec, translation_vec))
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
        #
        # original_validator.compare_result_to_ground(validation_df, results_list)

        hi = 5
        original_validator.validate()

    main()