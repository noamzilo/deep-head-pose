from Utils.yaml_utils.ConfigParser import ConfigParser
from Utils.utils import draw_axis_rotvec
import pandas as pd
from test_on_validation.validation_requirements.ValidationSetLoader import ValidationSetLoader
import os
import cv2
from Utils.path_utils import path_leaf
import numpy as np
import csv


if __name__ == "__main__":
    def main():
        validation_config_path = r"C:\Noam\Code\vision_course\hopenet\deep-head-pose\code\config\validation_config.yaml"
        validation_config = ConfigParser(validation_config_path).parse()
        test_config_path = r"C:\Noam\Code\vision_course\hopenet\deep-head-pose\code\config\test_config.yaml"
        test_config = ConfigParser(test_config_path).parse()
        hopenet_config_path = r"C:\Noam\Code\vision_course\hopenet\deep-head-pose\code\config\hopenet_config.yaml"
        hopenet_config = ConfigParser(hopenet_config_path).parse()

        validation_loader = ValidationSetLoader(validation_config)
        ground_truth_df = validation_loader.ground_truth[['file name', 'rx', 'ry', 'rz']]

        input_folder_path = validation_config.validation_images_folder_path
        # images_full_paths = [os.path.join(input_folder_path, image_name) for image_name in ground_truth_df['file name']]

        if not os.path.isdir(validation_config.test_validation_output_dir):
            os.mkdir(validation_config.test_validation_output_dir)

        results = []
        for i, row in enumerate(ground_truth_df.iterrows()):
            print(f"handling validation image #{i}")
            image_name, rx, ry, rz = row[1]
            image_full_path = os.path.join(input_folder_path, image_name)
            frame = cv2.imread(image_full_path)
            draw_axis_rotvec(frame, rx, ry, rz)
            results.append((image_full_path, np.array([rx, ry, rz, 0., 0., 0.])))

            out_file_full_path = os.path.join(validation_config.test_validation_output_dir, path_leaf(image_full_path))

            cv2.imwrite(filename=out_file_full_path, img=frame)

        with open(os.path.join(validation_config.test_validation_output_dir, "validation_output.csv"), "w", newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(",file name,rx,ry,rz,tx,ty,tz".split(','))
            for i, (file_name, line) in enumerate(results):
                writer.writerow([str(i)] + [path_leaf(file_name)] + list(line))
        hi=5

    main()
