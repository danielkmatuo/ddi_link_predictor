import pandas as pd

import os

def find_file(target):
    cwd = os.getcwd()
    target_path = ""

    for root, dirs, files in os.walk(cwd):
        for file in files:
            if file == target:
                target_path = os.path.join(root, file)

    if target_path == "":
        raise ValueError("Could not find the target file")

    return target_path

def create_csv_file(data: pd.DataFrame, file_name, feats_group, dir_name="test"):
    allowed_dirs = ["results", "data"]

    if dir_name not in allowed_dirs:
        raise ValueError("Not a valid directory")

    cwd = os.getcwd()
    group_dir = os.path.join(cwd, dir_name, feats_group)

    os.makedirs(group_dir, exist_ok=True)

    data_path = os.path.join(group_dir, file_name)
    data.to_csv(data_path, index=(dir_name != "data"))
