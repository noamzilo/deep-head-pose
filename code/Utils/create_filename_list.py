import os
import fnmatch
import glob
from os.path import relpath


def file_names_in_tree_root(treeroot):
    paths = []
    for filename in glob.iglob(treeroot + '**/**', recursive=True):
        if os.path.isfile(os.path.join(treeroot, filename)):
            paths.append(relpath(filename.split('.')[0], treeroot))

    paths = sorted(paths)

    create_paths_file_at = r"C:\Noam\Code\vision_course\downloads\datasets\300W-LP\rel_paths.txt"
    with open(create_paths_file_at, 'w') as f:
        for path in paths:
            f.write(f"{path}\n")

    return paths


if __name__ == "__main__":
    def main():
        test_path = r"C:\Noam\Code\vision_course\downloads\datasets\300W-LP\300W-3D"
        paths = file_names_in_tree_root(test_path)
        print(paths)

    main()
