from hopenet_estimator.HopenetEstimatorImages import HopenetEstimatorImages
from Utils.yaml_utils.ConfigParser import ConfigParser
from test_on_validation.validation_requirements.TestSetLoader import TestSetLoader
from Utils.create_filename_list import file_names_in_tree_root
from Utils.write_results_to_csv import write_results_to_csv
from test_on_validation.Validator import Validator
import pandas as pd
import os


if __name__ == "__main__":
    def main():
        hopenet_config, validation_config, test_config = _parse_config()

        test_set_loader = TestSetLoader(test_config)

        # hopenet_estimator = HopenetEstimatorImages(hopenet_config,
        #                                            validation_config,
        #                                            test_set_loader.test_images_full_paths)

        # full_validation_paths = r"C:\Noam\Code\vision_course\hopenet\input\validation\valid_set\rel_paths_filtered.txt"
        # with open(full_validation_paths, 'r') as f:
        #     validation_images_only_annotated = [path.strip() for path in f.readlines()]

        # hopenet_estimator = HopenetEstimatorImages(hopenet_config,
        #                                            validation_config,
        #                                            validation_images_only_annotated)

        _, rel_paths = file_names_in_tree_root(test_config.test_images_folder_path, test_config.test_images_folder_path,
                                "rel_paths.txt")

        abs_paths = [os.path.join(test_config.test_images_folder_path, path) + ".png" for path in rel_paths]

        hopenet_estimator = HopenetEstimatorImages(hopenet_config,
                                                   validation_config,
                                                   abs_paths)

        # no ground truth, can't compare to anything.
        results = hopenet_estimator.calculate_results()
        write_results_to_csv(test_config, results, hopenet_config.output_dir)

    def _parse_config():
        hopenet_config_path = r"C:\Noam\Code\vision_course\hopenet\deep-head-pose\code\config\hopenet_config.yaml"
        validation_config_path = r"C:\Noam\Code\vision_course\hopenet\deep-head-pose\code\config\validation_config.yaml"
        test_config_path = r"C:\Noam\Code\vision_course\hopenet\deep-head-pose\code\config\test_config.yaml"
        hopenet_config = ConfigParser(hopenet_config_path).parse()
        validation_config = ConfigParser(validation_config_path).parse()
        test_config = ConfigParser(test_config_path).parse()
        return hopenet_config, validation_config, test_config


    main()
