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
    allowed_dirs = ["results", "data"] # change if you want to create more directories

    if dir_name not in allowed_dirs:
        raise ValueError("Not a valid directory")

    cwd = os.getcwd()
    group_dir = os.path.join(dir_name, feats_group)

    if not os.path.exists(os.path.join(cwd, group_dir)) and dir_name != "data":
        os.mkdir(dir_name)
        os.mkdir(group_dir)
        data_path = os.path.join(cwd, group_dir, file_name)
        data.to_csv(data_path, index=False) if dir_name == "data" else data.to_csv(data_path, index=True)
        return
    else:
        data_path = os.path.join(cwd, group_dir, file_name)
        data.to_csv(data_path, index=False) if dir_name == "data" else data.to_csv(data_path, index=True)
        return 
