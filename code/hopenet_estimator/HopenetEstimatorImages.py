import numpy as np
from test_on_validation.validation_requirements import ThreeD_Model
import cv2
from test_on_validation.validation_requirements.ValidationSetLoader import ValidationSetLoader
from Utils.yaml_utils.ConfigParser import ConfigParser
from test_on_validation.Validator import Validator
import sys
import os
import argparse

import cv2

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from Utils import utils
from Utils.create_filename_list import file_names_in_tree_root
from original_code_augmented import hopenet
from original_code_augmented import datasets

import torch.nn.functional as F
from PIL import Image
import dlib


class HopenetEstimatorImages(object):
    def __init__(self, hopenet_config, validation_config, image_path_list):
        self._hopenet_config = hopenet_config
        self._validation_config = validation_config
        self._image_path_list = image_path_list

    def _setup(self):
        args = self._hopenet_config
        self._cnn_face_detector = dlib.cnn_face_detection_model_v1(args.face_detector_path)

        cudnn.enabled = True
        self._gpu_id = args.gpu_id

        snapshot_path = args.snapshot_path
        out_dir = args.output_dir
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print(f"will output to {out_dir}")
        self._out_dir = out_dir

        self._hopenet = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)


        print('Loading snapshot.')
        # Load snapshot
        saved_state_dict = torch.load(snapshot_path)
        self._hopenet.load_state_dict(saved_state_dict)
        print('Loading data.')
        self._transformations = transforms.Compose([transforms.Resize(224),
                                                    transforms.CenterCrop(224), transforms.ToTensor(),
                                                    transforms.Normalize(
                                                        mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])
        self._hopenet.cuda(self._gpu_id)
        print('Ready to test network.')
        # Test the Model
        self._hopenet.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

        idx_tensor = list(range(66))
        idx_tensor = torch.FloatTensor(idx_tensor).cuda(self._gpu_id)
        self._idx_tensor = idx_tensor


    def calculate_results(self):
        self._setup()
        args = self._hopenet_config

        results = []

        input_images_paths = self._image_path_list

        frame_num = 1
        for image_full_path in input_images_paths:
            print(f"frame #{frame_num}")
            frame_num += 1
            
            frame = cv2.imread(image_full_path)

            width = int(frame.shape[1] * args.scale_percent // 100)
            height = int(frame.shape[0] * args.scale_percent // 100)
            dsize = (width, height)
            frame = cv2.resize(frame, dsize)
            # cv2.imshow("current frame", frame)
            # cv2.waitKey(0)

            cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Dlib detect
            dets = self._cnn_face_detector(cv2_frame, 1)

            for idx, det in enumerate(dets):
                # Get x_min, y_min, x_max, y_max, conf
                x_min = det.rect.left()
                y_min = det.rect.top()
                x_max = det.rect.right()
                y_max = det.rect.bottom()
                conf = det.confidence

                if conf > 1.0:
                    bbox_width = abs(x_max - x_min)
                    bbox_height = abs(y_max - y_min)
                    x_min -= 2 * bbox_width // 4
                    x_max += 2 * bbox_width // 4
                    y_min -= 3 * bbox_height // 4
                    y_max += bbox_height // 4
                    x_min = max(x_min, 0)
                    y_min = max(y_min, 0)
                    x_max = min(frame.shape[1], x_max)
                    y_max = min(frame.shape[0], y_max)
                    # Crop image
                    img = cv2_frame[y_min:y_max, x_min:x_max]
                    img = Image.fromarray(img)

                    # Transform
                    img = self._transformations(img)
                    img_shape = img.size()
                    img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
                    img = Variable(img).cuda(self._gpu_id)

                    yaw, pitch, roll = self._hopenet(img)

                    yaw_predicted = F.softmax(yaw, dim=1)
                    pitch_predicted = F.softmax(pitch, dim=1)
                    roll_predicted = F.softmax(roll, dim=1)
                    # Get continuous predictions in degrees.
                    idx_tensor = self._idx_tensor
                    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
                    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
                    roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

                    results.append(np.array([roll_predicted, pitch_predicted, yaw_predicted, 0., 0., 0.]))

                    results.append(np.array([roll_predicted, pitch_predicted, yaw_predicted]))
                    # utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (x_min + x_max) / 2, (y_min + y_max) / 2, size = bbox_width)
                    utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx=(x_min + x_max) / 2,
                                    tdy=(y_min + y_max) / 2, size=bbox_height / 2)
                    # Plot expanded bounding box
                    # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)

        return results

