import os
import pandas as pd
import numpy as np
from validation_adapters.create_validation_rel_paths_file import create_image_paths_file
import glob
from os.path import relpath


class TestSetLoader(object):  # no ground truth file!
    def __init__(self, test_config):
        self._test_config = test_config
        self._test_images_folder_path = self._test_config.test_images_folder_path
        self._create_image_paths_file()

    def _create_image_paths_file(self):
        paths = []
        treeroot = self._test_config.test_images_folder_path
        assert os.path.isdir(treeroot)
        self._out_file_full_path = os.path.join(treeroot, self._test_config.rel_paths_file_name)
        if os.path.isfile(self._out_file_full_path):
            os.remove(self._out_file_full_path)

        for filename in glob.iglob(treeroot + '**/**', recursive=True):
            if os.path.isfile(os.path.join(treeroot, filename)):
                ext = filename.split('.')[1]
                if ext == 'jpg' or ext == 'png' or ext == 'png':
                    paths.append(relpath(filename, treeroot))

        paths = sorted(paths)
        self.test_images_rel_paths = paths
        self.test_images_full_paths = [os.path.join(treeroot, path) for path in self.test_images_rel_paths]

        out_file_full_path = os.path.join(self._test_config.create_rel_paths_file_at, self._test_config.rel_paths_file_name)
        with open(out_file_full_path, 'w') as f:
            f.writelines(self.test_images_rel_paths)
