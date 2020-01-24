import sys, os, argparse

import cv2

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from original_code_augmented import hopenet
from original_code_augmented import datasets
from Utils import utils
from Utils.create_filename_list import file_names_in_tree_root
from Utils.yaml_utils.ConfigParser import ConfigParser


if __name__ == '__main__':
    hopenet_config_path = r"C:\Noam\Code\vision_course\hopenet\deep-head-pose\code\config\hopenet_config.yaml"
    args = ConfigParser(hopenet_config_path).parse()
    create_paths_file_at = args.create_paths_file_at
    data_dir_path = args.test_data_dir_path
    file_name_list = file_names_in_tree_root(data_dir_path, create_paths_file_at)

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
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if args.dataset == 'Pose_300W_LP':
        pose_dataset = datasets.Pose_300W_LP(data_dir_path, args.file_name_list, transformations)
    elif args.dataset == 'Pose_300W_LP_random_ds':
        pose_dataset = datasets.Pose_300W_LP_random_ds(data_dir_path, args.file_name_list, transformations)
    elif args.dataset == 'AFLW2000':
        pose_dataset = datasets.AFLW2000(data_dir_path, args.file_name_list, transformations)
    elif args.dataset == 'AFLW2000_ds':
        pose_dataset = datasets.AFLW2000_ds(data_dir_path, args.file_name_list, transformations)
    elif args.dataset == 'BIWI':
        pose_dataset = datasets.BIWI(data_dir_path, args.file_name_list, transformations)
    elif args.dataset == 'AFLW':
        pose_dataset = datasets.AFLW(data_dir_path, args.file_name_list, transformations)
    elif args.dataset == 'AFLW_aug':
        pose_dataset = datasets.AFLW_aug(data_dir_path, args.file_name_list, transformations)
    elif args.dataset == 'AFW':
        pose_dataset = datasets.AFW(data_dir_path, args.file_name_list, transformations)
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

        label_yaw = cont_labels[:,0].float()
        label_pitch = cont_labels[:,1].float()
        label_roll = cont_labels[:,2].float()

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
                error_string = 'y %.2f, p %.2f, r %.2f' % (torch.sum(torch.abs(yaw_predicted - label_yaw)), torch.sum(torch.abs(pitch_predicted - label_pitch)), torch.sum(torch.abs(roll_predicted - label_roll)))
                cv2.putText(cv2_img, error_string, (30, cv2_img.shape[0]- 30), fontFace=1, fontScale=1, color=(0,0,255), thickness=2)
            # utils.plot_pose_cube(cv2_img, yaw_predicted[0], pitch_predicted[0], roll_predicted[0], size=100)
            utils.draw_axis(cv2_img, yaw_predicted[0], pitch_predicted[0], roll_predicted[0], tdx = 200, tdy= 200, size=100)
            cv2.imwrite(os.path.join('output/images', name + '.jpg'), cv2_img)

    print(('Test error in degrees of the model on the ' + str(total) +
        ' test images. Yaw: %.4f, Pitch: %.4f, Roll: %.4f' % (yaw_error / total,
        pitch_error / total, roll_error / total)))
