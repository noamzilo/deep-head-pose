import os
import fnmatch
import glob
from os.path import relpath


def file_names_in_tree_root(treeroot, create_file_dir, file_name):
    paths = []
    for filename in glob.iglob(treeroot + '**/**', recursive=True):
        if os.path.isfile(os.path.join(treeroot, filename)):
            paths.append(relpath(filename.split('.')[0], treeroot))

    paths = sorted(paths)

    file_path = os.path.join(create_file_dir, file_name)
    if os.path.isfile(file_path):
        os.remove(file_path)

    with open(file_path, 'w') as f:
        for path in paths:
            f.write(f"{path}\n")

    return file_path, paths


if __name__ == "__main__":
    def main():
        test_path = r"C:\Noam\Code\vision_course\downloads\datasets\300W-LP\300W-3D"
        paths = file_names_in_tree_root(test_path)
        print(paths)

    main()
