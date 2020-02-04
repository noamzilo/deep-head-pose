import os
import scipy.io as sio
import numpy as np


def get_pose_params_from_mat(mat_path):
    # This functions gets the pose parameters from the .mat
    # Annotations that come with the Pose_300W_LP dataset.
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll, tdx, tdy]
    pose_params = pre_pose_params[:5]
    return pose_params

def filter_images_by_angle(images_dir, rel_paths_file_name, out_file_name):
    print(f"trying to read from {images_dir}")

    file_path = os.path.join(images_dir, rel_paths_file_name)
    with open(file_path) as f:
        image_names = [line.rstrip() for line in f]

    poses = []
    filtered = []
    last_image_name = None
    for i, image_name in enumerate(image_names):
        if image_name == last_image_name:
            continue
        if i % 200 == 0:
            print(f"Processing .mat #{i}/{len(image_names)}")
        image_full_path = os.path.join(images_dir, image_name)
        mat_path = image_full_path + ".mat"
        pose = np.rad2deg(get_pose_params_from_mat(mat_path)[0:3])
        if np.max(np.abs(pose[0:3])) < 99:
            poses.append(pose[0:3])
            filtered.append(image_name)
        last_image_name = image_name

    print("done parsing")

    assert 1000 < len(filtered) < len(image_names)
    out_path = os.path.join(images_dir, out_file_name)
    with open(out_path, 'w') as out:
        for path in filtered:
            out.write(f"{path}\n")


if __name__ == "__main__":
    def main():
        rel_paths_dir_windows = r"C:\Noam\Code\vision_course\downloads\datasets\300W-LP\big_set\300W_LP"
        rel_paths_dir_linux = r"/home/noams/hopenet/deep-head-pose/code/Data/Training/300W_LP"
        file_name = r"rel_paths.txt"
        out_file_name = r"rel_paths_filtered.txt"

        filter_images_by_angle(rel_paths_dir_linux, file_name, out_file_name)
        # filter_images_by_angle(rel_paths_dir_windows, file_name, out_file_name)

    main()
