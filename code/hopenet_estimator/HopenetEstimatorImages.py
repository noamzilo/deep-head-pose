import numpy as np
import os

import cv2

import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from Utils import utils
from original_code_augmented import hopenet

import torch.nn.functional as F
from PIL import Image
import dlib
from Utils.path_utils import path_leaf
from scipy.spatial.transform import Rotation as R


class HopenetEstimatorImages(object):
    def __init__(self, hopenet_config, validation_config, image_full_path_list, snapshot_path=None):
        self._hopenet_config = hopenet_config
        self._validation_config = validation_config
        self._image_path_list = image_full_path_list
        self._is_using_opencv_face_detector = self._hopenet_config.is_using_opencv_face_detector
        if snapshot_path is None:
            self._snapshot_path = self._hopenet_config.snapshot_path
        else:
            self._snapshot_path = snapshot_path

    def _setup(self):
        args = self._hopenet_config
        self._cnn_face_detector = dlib.cnn_face_detection_model_v1(args.face_detector_path)

        cudnn.enabled = True
        self._gpu_id = args.gpu_id

        snapshot_path = self._snapshot_path
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
        for path in input_images_paths:
            assert os.path.isfile(path)

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
            # CAN DISABLE FACE DETECTION FOR SPEED
            if self._is_using_opencv_face_detector:
                detections = self._cnn_face_detector(cv2_frame, 1)
            else:
                detections = []

            detections = sorted(detections, key=lambda x: x.confidence)
            if len(detections) > 0:
                detection = detections[0] # TODO if no detection, return the entire frame.
                x_min = detection.rect.left()
                y_min = detection.rect.top()
                x_max = detection.rect.right()
                y_max = detection.rect.bottom()
            else:  # we are certain there is one detection, and only one, we just didn't find it, so use everything
                x_min = 0
                y_min = 0
                x_max = cv2_frame.shape[1] - 1  # width
                y_max = cv2_frame.shape[0] - 1  # height

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

            def rpy2xyz(r, p, y):
                r = R.from_euler('zxy', (r, -p, y), degrees=True)
                return r.as_rotvec()

            x, y, z = rpy2xyz(roll_predicted.item(), pitch_predicted.item(), yaw_predicted.item())

            results.append((image_full_path, np.array([x, y, z, 0., 0., 0.])))
            is_plot_rvec = True
            if is_plot_rvec:
                utils.draw_axis_rotvec(frame, x, y, z, tdx=(x_min + x_max) / 2,
                                       tdy=(y_min + y_max) / 2, size=bbox_height / 2)
            else:
                utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx=(x_min + x_max) / 2,
                                tdy=(y_min + y_max) / 2, size=bbox_height / 2)


            # utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (x_min + x_max) / 2, (y_min + y_max) / 2, size = bbox_width)

            # Plot expanded bounding box
            # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)



            out_file_full_path = os.path.join(self._hopenet_config.output_dir, path_leaf(image_full_path))
            cv2.imwrite(filename=out_file_full_path, img=frame)

        return results

