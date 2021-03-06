import sys, os, argparse

import cv2

import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image

from original_code_augmented import hopenet
from Utils import utils

import dlib
from Utils.validate_gpu import validate


def parse_args():
    # default_dlib_model_path = r"C:\Noam\Code\vision_course\hopenet\models\shape_predictor_68_face_landmarks.dat"
    default_dlib_model_path = r"C:\Noam\Code\vision_course\hopenet\models\mmod_human_face_detector.dat"
    default_snapshot_path = r"C:\Noam\Code\vision_course\hopenet\models\hopenet_robust_alpha1.pkl"

    default_video_path = r"C:\Noam\Code\vision_course\hopenet\videos\video_resize.mp4"
    default_n_frames = 24
    default_fps = 24.  # was 30.
    scale_percent = 100

    default_video_path = r"C:\Noam\Code\vision_course\hopenet\videos\video.mp4"
    default_n_frames = 24
    default_fps = 24.  # was 30.
    scale_percent = 12.5


    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
                        default=default_snapshot_path, type=str)
    parser.add_argument('--face_model', dest='face_model', help='Path of DLIB face detection model.',
                        default=default_dlib_model_path, type=str)
    parser.add_argument('--video', dest='video_path', help='Path of video',
                        default=default_video_path)
    parser.add_argument('--output_string', dest='output_string', help='String appended to output file')
    parser.add_argument('--n_frames', dest='n_frames', help='Number of frames', type=int,
                        default=default_n_frames)
    parser.add_argument('--fps', dest='fps', help='Frames per second of source video', type=float,
                        default=default_fps)
    parser.add_argument('--scale_percent', dest='scale_percent', help='scale_percent', type=float,
                        default=scale_percent)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    def main():
        args, cnn_face_detector, gpu, idx_tensor, model, out, transformations, video = _setup()

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

                        # Print new frame with cube and axis
                        txt_out.write(f"{str(frame_num)} {yaw_predicted} {pitch_predicted} {roll_predicted}\n")
                        # utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (x_min + x_max) / 2, (y_min + y_max) / 2, size = bbox_width)
                        utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx =(x_min + x_max) / 2, tdy=(y_min + y_max) / 2, size =bbox_height / 2)
                        # Plot expanded bounding box
                        # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)

                out.write(frame)
                frame_num += 1

            out.release()
            video.release()


    def _setup():
        validate()
        args = parse_args()
        cudnn.enabled = True
        batch_size = 1
        gpu = args.gpu_id
        snapshot_path = args.snapshot
        out_dir = '../output/video'
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


    main()
