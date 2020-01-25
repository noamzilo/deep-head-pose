import os
import pandas as pd
import numpy as np
from validation_adapters.create_validation_rel_paths_file import create_image_paths_file


class ValidationSetLoader(object):
    def __init__(self, validation_config):
        self._validation_config = validation_config
        self._ground_truth_file_path = self._validation_config.ground_truth_file_path
        self._validation_images_folder_path = self._validation_config.validation_images_folder_path

        self._load_validation_set()

    def _load_validation_points(self):
        validation_points = []
        for file_path in self._validation_pts_paths:
            points = np.loadtxt(file_path, comments=("version:", "n_points:", "{", "}"))
            validation_points.append(points)
        self.validation_points = validation_points

        # create paths file for validation
        path_validation_images_paths_file, _ = create_image_paths_file(  # TODO this belongs in ValidationSetLoader
            self.validation_image_paths,
            self._validation_config.create_rel_paths_file_at,
            self._validation_config.rel_paths_file_name)

    def _load_validation_set(self):
        ground_truth = pd.read_csv(self._ground_truth_file_path,
                                   sep=r'\s*,\s*',
                                   header=0,
                                   encoding='ascii',
                                   engine='python')
        self.ground_truth = ground_truth

        self._validation_images_filenames = [filename for filename in ground_truth['file name'].values]
        self._validation_pts_filenames = [filename.replace('.png', '.pts')
                                          for filename in self._validation_images_filenames]

        self.validation_image_paths = [os.path.join(self._validation_images_folder_path, image_file_name)
                                        for image_file_name in self._validation_images_filenames]
        self._validation_pts_paths = [os.path.join(self._validation_images_folder_path, pts_file_name)
                                      for pts_file_name in self._validation_pts_filenames]

        # self._load_validation_points()
