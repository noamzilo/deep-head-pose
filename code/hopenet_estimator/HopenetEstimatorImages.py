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
    def __init__(self, hopenet_config, validation_config, input_images_folder, input_images_rel_paths_file_name):
        self._hopenet_config = hopenet_config
        self._validation_config = validation_config
        self._input_images_folder = input_images_folder
        self._input_images_rel_paths_file_name = input_images_rel_paths_file_name
        self._input_images_file_abs_path = os.path.join(self._input_images_folder, self._input_images_rel_paths_file_name)
        self._calculate_absolute_image_paths()

    def _calculate_absolute_image_paths(self):
        # file_names_in_tree_root(self._input_images_folder ,self._input_images_folder, self._input_images_rel_paths_file_name)
        validation_set_loader = ValidationSetLoader(self._validation_config)
        paths = validation_set_loader.validation_image_paths
        with open(self._input_images_file_abs_path) as f:
            paths = f.readlines()
        self._abs_images_paths = [os.path.join(self._input_images_folder, path) for path in paths]
        hi=5

    def calculate_results(self):
        def _setup(args):
            args.face_model = r"C:\Noam\Code\vision_course\hopenet\models\mmod_human_face_detector.dat"
            args.video_path = r"C:\Noam\Code\vision_course\hopenet\videos\video_resize.mp4"
            args.n_frames = 3
            args.fps = 24.  # was 30.
            args.scale_percent = 100

            cudnn.enabled = True
            batch_size = 1
            gpu = args.gpu_id
            snapshot_path = args.snapshot_path
            out_dir = os.path.abspath('../output/video')
            print(f"will output to {out_dir}")

            video_path = args.video_path
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            if not os.path.exists(args.video_path):
                sys.exit('Video does not exist')
            # ResNet50 structure
            model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
            # Dlib face detection model
            cnn_face_detector = dlib.cnn_face_detection_model_v1(args.face_model)
            print('Loading snapshot.')
            # Load snapshot
            saved_state_dict = torch.load(snapshot_path)
            model.load_state_dict(saved_state_dict)
            print('Loading data.')
            transformations = transforms.Compose([transforms.Resize(224),
                                                  transforms.CenterCrop(224), transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225])])
            model.cuda(gpu)
            print('Ready to test network.')
            # Test the Model
            model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
            total = 0
            idx_tensor = list(range(66))
            idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
            video = cv2.VideoCapture(video_path)
            # New cv2
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter('../output/video/output-%s.avi' % args.output_string, fourcc, args.fps, (width, height))
            return args, cnn_face_detector, gpu, idx_tensor, model, out, transformations, video

        args, cnn_face_detector, gpu, idx_tensor, model, out, transformations, video = _setup(self._hopenet_config)

        results = []

        out_file_path = '../output/video/output-%s.txt' % args.output_string
        assert os.path.isfile(out_file_path)
        with open(out_file_path, 'w') as txt_out:
            frame_num = 1

            while frame_num <= args.n_frames:
                print(frame_num)

                ret, frame = video.read()
                if not ret:
                    break

                width = int(frame.shape[1] * args.scale_percent // 100)
                height = int(frame.shape[0] * args.scale_percent // 100)
                dsize = (width, height)
                frame = cv2.resize(frame, dsize)
                # cv2.imshow("current frame", frame)
                # cv2.waitKey(0)

                cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Dlib detect
                dets = cnn_face_detector(cv2_frame, 1)

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
                        img = transformations(img)
                        img_shape = img.size()
                        img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
                        img = Variable(img).cuda(gpu)

                        yaw, pitch, roll = model(img)

                        yaw_predicted = F.softmax(yaw, dim=1)
                        pitch_predicted = F.softmax(pitch, dim=1)
                        roll_predicted = F.softmax(roll, dim=1)
                        # Get continuous predictions in degrees.
                        yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
                        pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
                        roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

                        results.append(np.array([roll_predicted, pitch_predicted, yaw_predicted, 0., 0., 0.]))

                        # Print new frame with cube and axis
                        txt_out.write(f"{str(frame_num)} {yaw_predicted} {pitch_predicted} {roll_predicted}\n")
                        # utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (x_min + x_max) / 2, (y_min + y_max) / 2, size = bbox_width)
                        utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx=(x_min + x_max) / 2,
                                        tdy=(y_min + y_max) / 2, size=bbox_height / 2)
                        # Plot expanded bounding box
                        # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)

                out.write(frame)
                frame_num += 1

            out.release()
            video.release()

            return results

