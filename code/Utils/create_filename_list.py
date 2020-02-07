import os
import fnmatch
import glob
from os.path import relpath


def file_names_in_tree_root(treeroot, create_file_dir, file_name):
    file_path = os.path.join(create_file_dir, file_name)
    if os.path.isfile(file_path):
        os.remove(file_path)

    paths = []
    last_path = None
    for filename in glob.iglob(treeroot + '**/**', recursive=True):
        if os.path.isfile(os.path.join(treeroot, filename)):
            if not (filename.endswith(".jpg") or filename.endswith(".mat") or filename.endswith(".png")):
                continue
            new_path = relpath(filename.split('.')[0], treeroot)
            # print(f"new: {new_path} \t\t olf: {last_path} \n")
            if last_path == new_path:
                continue
                # print("ssss")
            paths.append(new_path)
            last_path = new_path

    paths = sorted(paths)
    print(f"number of files found: {len(paths)}")

    with open(file_path, 'w') as f:
        for path in paths:
            f.write(f"{path}\n")

    return file_path, paths


if __name__ == "__main__":
    def main():
        test_path_windows = r"C:\Noam\Code\vision_course\downloads\datasets\300W-LP\big_set\300W_LP"
        test_path_linux = r"/home/noams/hopenet/deep-head-pose/code/Data/Training/300W_LP"
        create_at = test_path_linux
        file_name = "rel_paths.txt"
        paths = file_names_in_tree_root(test_path_linux, create_at, file_name)
        print(paths)

    main()
