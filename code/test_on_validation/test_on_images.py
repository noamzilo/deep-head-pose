from hopenet_estimator.HopenetEstimatorImages import HopenetEstimatorImages
from Utils.yaml_utils.ConfigParser import ConfigParser
from test_on_validation.validation_requirements.ValidationSetLoader import ValidationSetLoader
from Utils.create_filename_list import file_names_in_tree_root
from test_on_validation.Validator import Validator


if __name__ == "__main__":
    def main():
        hopenet_config, validation_config = _parse_config()

        validation_set_loader = ValidationSetLoader(validation_config)

        hopenet_estimator = HopenetEstimatorImages(hopenet_config,
                                             validation_config,
                                             input_images_folder=validation_config.validation_images_folder_path,
                                             input_images_rel_paths_file_name=validation_config.rel_paths_file_name)

        validator = Validator(validation_config, hopenet_estimator, validation_set_loader)
        validator.validate()


    def _parse_config():
        hopenet_config_path = r"C:\Noam\Code\vision_course\hopenet\deep-head-pose\code\config\hopenet_config.yaml"
        validation_config_path = r"C:\Noam\Code\vision_course\hopenet\deep-head-pose\code\config\validation_config.yaml"
        hopenet_config = ConfigParser(hopenet_config_path).parse()
        validation_config = ConfigParser(validation_config_path).parse()
        return hopenet_config, validation_config


    main()
