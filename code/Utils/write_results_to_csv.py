import os
import csv
from Utils.path_utils import path_leaf


def write_results_to_csv(test_config, results, output_dir):
    output_file_path = test_config.output_file_name

    assert os.path.isdir(output_dir)
    output_full_path = os.path.join(output_dir, output_file_path)

    with open(output_full_path, "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(",file name,rx,ry,rz,tx,ty,tz".split(','))
        for i, (file_name, line) in enumerate(results):
            writer.writerow([str(i)] + [path_leaf(file_name)] + list(line))
