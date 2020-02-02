import sys, os, argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch.backends.cudnn as cudnn

from original_code_augmented import hopenet, datasets
import torch.utils.model_zoo as model_zoo
from Utils.create_filename_list import file_names_in_tree_root

from Utils.yaml_utils.ConfigParser import ConfigParser
import cv2
from scipy.spatial.transform import Rotation as R
from Utils import utils
import numpy as np


def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_yaw, model.fc_pitch, model.fc_roll]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param

def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


if __name__ == '__main__':
    hopenet_config_path = r"C:\Noam\Code\vision_course\hopenet\deep-head-pose\code\config\train_config_holdout.yaml"
    hopenet_config = ConfigParser(hopenet_config_path).parse()
    args = hopenet_config
    data_dir = args.train_data_dir_path
    filename_list = args.file_name_list

    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size_train
    gpu = args.gpu_id
    snapshot_path = args.snapshot_path
    train_percentage = args.train_percentage
    seed = args.seed

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots')

    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    if snapshot_path == '':
        load_filtered_state_dict(model, model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    else:
        saved_state_dict = torch.load(snapshot_path)
        model.load_state_dict(saved_state_dict)

    print('Loading data.')

    transformations = transforms.Compose([transforms.Resize(240),
    transforms.RandomCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # train_percentage = 100
    if args.dataset == 'Pose_300W_LP':
        pose_dataset = datasets.Pose_300W_LP(data_dir,
                                             filename_list,
                                             transformations,
                                             train_percent=train_percentage,
                                             use_train=True,
                                             seed=seed
                                             )
        pose_dataset_validation = datasets.Pose_300W_LP(data_dir,
                                             filename_list,
                                             transformations,
                                             train_percent=train_percentage,
                                             use_train=False,
                                             seed=seed
                                             )
    elif args.dataset == 'Pose_300W_LP_random_ds':
        pose_dataset = datasets.Pose_300W_LP_random_ds(data_dir, filename_list, transformations)
    elif args.dataset == 'Synhead':
        pose_dataset = datasets.Synhead(data_dir, filename_list, transformations)
    elif args.dataset == 'AFLW2000':
        pose_dataset = datasets.AFLW2000(data_dir, filename_list, transformations)
    elif args.dataset == 'BIWI':
        pose_dataset = datasets.BIWI(data_dir, filename_list, transformations)
    elif args.dataset == 'AFLW':
        pose_dataset = datasets.AFLW(data_dir, filename_list, transformations)
    elif args.dataset == 'AFLW_aug':
        pose_dataset = datasets.AFLW_aug(data_dir, filename_list, transformations)
    elif args.dataset == 'AFW':
        pose_dataset = datasets.AFW(data_dir, filename_list, transformations)
    else:
        print('Error: not a valid dataset name')
        sys.exit()

    train_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)
    validation_loader = torch.utils.data.DataLoader(dataset=pose_dataset_validation,
                                               batch_size=1,
                                               shuffle=True,
                                               num_workers=4)

    model.cuda(gpu)
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    reg_criterion = nn.MSELoss().cuda(gpu)
    # Regression loss coefficient
    alpha = args.alpha
    learning_rate = args.learning_rate

    softmax = nn.Softmax().cuda(gpu)
    idx_tensor = list(range(66))
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).cuda(gpu)

    idx_tensor_ = list(range(66))
    idx_tensor_ = Variable(torch.FloatTensor(idx_tensor_)).cuda(gpu)

    optimizer = torch.optim.Adam([{'params': get_ignored_params(model), 'lr': 0},
                                  {'params': get_non_ignored_params(model), 'lr': learning_rate},
                                  {'params': get_fc_params(model), 'lr': learning_rate * 5}],
                                   lr = learning_rate)

    print('Ready to train network.')
    for epoch in range(num_epochs):
        # model.train()
        for i, (images, labels, cont_labels, name) in enumerate(train_loader):
            if i % 20 == 0:
                print(f"image #{i}")
            images = Variable(images).cuda(gpu)

            # Binned labels
            label_yaw = Variable(labels[:,0]).cuda(gpu)
            label_pitch = Variable(labels[:,1]).cuda(gpu)
            label_roll = Variable(labels[:,2]).cuda(gpu)

            # Continuous labels
            label_yaw_cont = Variable(cont_labels[:,0]).cuda(gpu)
            label_pitch_cont = Variable(cont_labels[:,1]).cuda(gpu)
            label_roll_cont = Variable(cont_labels[:,2]).cuda(gpu)

            # Forward pass
            yaw, pitch, roll = model(images)

            # Cross entropy loss
            loss_yaw = criterion(yaw, label_yaw)
            loss_pitch = criterion(pitch, label_pitch)
            loss_roll = criterion(roll, label_roll)

            # MSE loss
            yaw_predicted = softmax(yaw)
            pitch_predicted = softmax(pitch)
            roll_predicted = softmax(roll)

            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * 3 - 99

            loss_reg_yaw = reg_criterion(yaw_predicted, label_yaw_cont)
            loss_reg_pitch = reg_criterion(pitch_predicted, label_pitch_cont)
            loss_reg_roll = reg_criterion(roll_predicted, label_roll_cont)

            # Total loss
            loss_yaw += alpha * loss_reg_yaw
            loss_pitch += alpha * loss_reg_pitch
            loss_roll += alpha * loss_reg_roll

            loss_seq = [loss_yaw, loss_pitch, loss_roll]
            grad_seq = [torch.tensor(1.0, dtype=torch.float).cuda(gpu) for _ in range(len(loss_seq))]
            optimizer.zero_grad()
            torch.autograd.backward(loss_seq, grad_seq)
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d/%d] Losses: Yaw %.4f, Pitch %.4f, Roll %.4f'
                       %(epoch+1,
                         num_epochs,
                         i+1,
                         len(pose_dataset)//batch_size,
                         loss_yaw.item(),
                         loss_pitch.item(),
                         loss_roll.item()))

        # Save models at numbered epochs.
        if epoch % 1 == 0 and epoch < num_epochs:
            print('Taking snapshot...')
            torch.save(model.state_dict(),
            'output/snapshots/' + args.output_string + '_epoch_'+ str(epoch+1) + '.pkl')

        # calculate validation error
        # print("about to validate")
        # model.eval()
        # with torch.no_grad():
        #     for i, (images_, labels_, cont_labels_, name_) in enumerate(validation_loader):
        #         original = images_[0].cpu().numpy().swapaxes(0, 1).swapaxes(1, 2)
        #         # cv2.imshow("image", original )
        #         # cv2.waitKey(0)
        #
        #         images_ = Variable(images_).cuda(gpu)
        #
        #         # Binned labels
        #         label_yaw_ = Variable(labels_[:,0]).cuda(gpu)
        #         label_pitch_ = Variable(labels_[:,1]).cuda(gpu)
        #         label_roll_ = Variable(labels_[:,2]).cuda(gpu)
        #
        #         # Continuous labels
        #         label_yaw_cont_ = Variable(cont_labels_[:,0]).cuda(gpu)
        #         label_pitch_cont_ = Variable(cont_labels_[:,1]).cuda(gpu)
        #         label_roll_cont_ = Variable(cont_labels_[:,2]).cuda(gpu)
        #
        #         # Forward pass
        #         yaw_, pitch_, roll_ = model(images_)
        #
        #         # Cross entropy loss
        #         loss_yaw_ = criterion(yaw_, label_yaw_)
        #         loss_pitch_ = criterion(pitch_, label_pitch_)
        #         loss_roll_ = criterion(roll_, label_roll_)
        #
        #         # MSE loss
        #         yaw_predicted_ = softmax(yaw_)
        #         pitch_predicted_ = softmax(pitch_)
        #         roll_predicted_ = softmax(roll_)
        #
        #         yaw_predicted_ = torch.sum(yaw_predicted_ * idx_tensor_, 1) * 3 - 99
        #         pitch_predicted_ = torch.sum(pitch_predicted_ * idx_tensor_, 1) * 3 - 99
        #         roll_predicted_ = torch.sum(roll_predicted_ * idx_tensor_, 1) * 3 - 99
        #
        #         loss_reg_yaw_ = reg_criterion(yaw_predicted_, label_yaw_cont_)
        #         loss_reg_pitch_ = reg_criterion(pitch_predicted_, label_pitch_cont_)
        #         loss_reg_roll_ = reg_criterion(roll_predicted_, label_roll_cont_)
        #
        #         # Total loss
        #         loss_yaw_ += alpha * loss_reg_yaw_
        #         loss_pitch_ += alpha * loss_reg_pitch_
        #         loss_roll_ += alpha * loss_reg_roll_
        #
        #         if (i+1) % 100 == 0:
        #             print ('Epoch Validation Loss: [%d/%d], Iter [%d/%d] Losses: Yaw %.4f, Pitch %.4f, Roll %.4f'
        #                    %(epoch+1,
        #                      num_epochs,
        #                      i+1,
        #                      len(pose_dataset_validation)//1,
        #                      loss_yaw_.item(),
        #                      loss_pitch_.item(),
        #                      loss_roll_.item()))
        #
        #             if not os.path.isdir(hopenet_config.output_dir):
        #                 os.mkdir(hopenet_config.output_dir)
        #             frame_ = images_[0]
        #
        #             def rpy2xyz(r, p, y):
        #                 r = R.from_euler('zxy', (r, -p, y), degrees=True)
        #                 return r.as_rotvec()
        #
        #             x_, y_, z_ = rpy2xyz(roll_predicted_.item(), pitch_predicted_.item(), yaw_predicted_.item())
        #
        #             is_plot_rvec = True
        #             # frame_cpu = frame_.cpu().numpy().swapaxes(0, 1).swapaxes(1, 2)
        #             # frame_cpu_normed = (frame_cpu - frame_cpu.min()) / (frame_cpu.max() - frame_cpu.min())
        #             frame_cpu_normed = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        #
        #             frame_cpu_normed = 255 * (frame_cpu_normed - frame_cpu_normed .min()) / (frame_cpu_normed .max() - frame_cpu_normed .min())
        #             frame_cpu_normed = np.array(frame_cpu_normed, np.int)
        #
        #             frame_cpu_tagged = utils.draw_axis_rotvec(frame_cpu_normed, x_, y_, z_)
        #
        #
        #
        #             file_name = os.path.join(hopenet_config.output_dir, "_".join(name_[0].split('\\'))) + f"epoch_{epoch}" + '.jpg'
        #             # cv2.imshow("process", frame_cpu_normed)
        #             # cv2.waitKey(0)
        #             cv2.imwrite(filename=file_name, img=frame_cpu_tagged)
        #             hi = 5

