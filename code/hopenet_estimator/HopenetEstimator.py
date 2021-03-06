import numpy as np
from test_on_validation.validation_requirements import ThreeD_Model
import cv2

from Utils.yaml_utils.ConfigParser import ConfigParser

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

from original_code_augmented import hopenet
from original_code_augmented import datasets

import torch.nn.functional as F
from PIL import Image
import dlib
from hopenet_estimator.HopenetEstimatorImages import HopenetEstimatorImages


class HopenetEstimator(object):
    def __init__(self, hopenet_config, validation_config, input_images_folder, input_images_rel_paths_file_name):
        self._hopenet_config = hopenet_config
        self._validation_config = validation_config
        self._input_images_folder = input_images_folder
        self._input_images_rel_paths_file_name = input_images_rel_paths_file_name

    def calculate_results_on_train_set(self):
        args = self._hopenet_config
        # data_dir_path = args.test_data_dir_path
        data_dir_path = self._input_images_folder

        file_name_list = os.path.join(self._input_images_folder, self._input_images_rel_paths_file_name)
        # file_name_list, _ = file_names_in_tree_root(data_dir_path, create_paths_file_at)

        results = []

        cudnn.enabled = True
        gpu = args.gpu_id
        snapshot_path = args.snapshot_path

        # ResNet50 structure
        model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

        print('Loading snapshot.')
        # Load snapshot
        saved_state_dict = torch.load(snapshot_path)
        model.load_state_dict(saved_state_dict)

        print('Loading data.')

        transformations = transforms.Compose([transforms.Resize(224),
                                              transforms.CenterCrop(224), transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])

        if args.dataset == 'Pose_300W_LP':
            pose_dataset = datasets.Pose_300W_LP(data_dir_path, file_name_list, transformations) #TODO either data loader for validation or use video loader.
        elif args.dataset == 'Pose_300W_LP_random_ds':
            pose_dataset = datasets.Pose_300W_LP_random_ds(data_dir_path, file_name_list, transformations)
        elif args.dataset == 'AFLW2000':
            pose_dataset = datasets.AFLW2000(data_dir_path, file_name_list, transformations)
        elif args.dataset == 'AFLW2000_ds':
            pose_dataset = datasets.AFLW2000_ds(data_dir_path, file_name_list, transformations)
        elif args.dataset == 'BIWI':
            pose_dataset = datasets.BIWI(data_dir_path, file_name_list, transformations)
        elif args.dataset == 'AFLW':
            pose_dataset = datasets.AFLW(data_dir_path, file_name_list, transformations)
        elif args.dataset == 'AFLW_aug':
            pose_dataset = datasets.AFLW_aug(data_dir_path, file_name_list, transformations)
        elif args.dataset == 'AFW':
            pose_dataset = datasets.AFW(data_dir_path, file_name_list, transformations)
        else:
            print('Error: not a valid dataset name')
            sys.exit()
        test_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                                  batch_size=args.batch_size_test,
                                                  num_workers=2)

        model.cuda(gpu)

        print('Ready to test network.')

        # Test the Model
        model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        total = 0

        idx_tensor = list(range(66))
        idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

        yaw_error = .0
        pitch_error = .0
        roll_error = .0

        l1loss = torch.nn.L1Loss(reduction='sum')

        for i, (images, labels, cont_labels, name) in enumerate(test_loader):
            images = Variable(images).cuda(gpu)
            total += cont_labels.size(0)

            label_yaw = cont_labels[:, 0].float()
            label_pitch = cont_labels[:, 1].float()
            label_roll = cont_labels[:, 2].float()

            yaw, pitch, roll = model(images)

            # Binned predictions
            _, yaw_bpred = torch.max(yaw.data, 1)
            _, pitch_bpred = torch.max(pitch.data, 1)
            _, roll_bpred = torch.max(roll.data, 1)

            # Continuous predictions
            yaw_predicted = utils.softmax_temperature(yaw.data, 1)
            pitch_predicted = utils.softmax_temperature(pitch.data, 1)
            roll_predicted = utils.softmax_temperature(roll.data, 1)

            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 99
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 99

            results.append(np.array([yaw_predicted, pitch_predicted, roll_predicted, 0., 0., 0.]))

            # Mean absolute error
            yaw_error += torch.sum(torch.abs(yaw_predicted - label_yaw))
            pitch_error += torch.sum(torch.abs(pitch_predicted - label_pitch))
            roll_error += torch.sum(torch.abs(roll_predicted - label_roll))

            # Save first image in batch with pose cube or axis.
            if args.save_viz:
                name = name[0]
                if args.dataset == 'BIWI':
                    cv2_img = cv2.imread(os.path.join(data_dir_path, name + '_rgb.png'))
                else:
                    cv2_img = cv2.imread(os.path.join(data_dir_path, name + '.jpg'))
                if args.batch_size_test == 1:
                    error_string = 'y %.2f, p %.2f, r %.2f' % (torch.sum(torch.abs(yaw_predicted - label_yaw)),
                                                               torch.sum(torch.abs(pitch_predicted - label_pitch)),
                                                               torch.sum(torch.abs(roll_predicted - label_roll)))
                    cv2.putText(cv2_img, error_string, (30, cv2_img.shape[0] - 30), fontFace=1, fontScale=1,
                                color=(0, 0, 255), thickness=2)
                # utils.plot_pose_cube(cv2_img, yaw_predicted[0], pitch_predicted[0], roll_predicted[0], size=100)
                utils.draw_axis(cv2_img, yaw_predicted[0], pitch_predicted[0], roll_predicted[0], tdx=200, tdy=200,
                                size=100)
                cv2.imwrite(os.path.join('output/images', name + '.jpg'), cv2_img)

        print(('Test error in degrees of the model on the ' + str(total) +
               ' test images. Yaw: %.4f, Pitch: %.4f, Roll: %.4f' % (yaw_error / total,
                                                                     pitch_error / total, roll_error / total)))

        return results

    def calculate_results_on_video(self):
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


