import os


def create_image_paths_file(validation_image_paths, create_file_at_dir, file_name):
    paths = validation_image_paths
    paths = sorted(paths)

    create_file_at = os.path.join(create_file_at_dir, file_name)
    if os.path.isfile(create_file_at):
        os.remove(create_file_at)

    with open(create_file_at, 'w') as f:
        for path in paths:
            f.write(f"{path}\n")

    return create_file_at, paths