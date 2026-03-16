import os
import json
import numpy as np


def load_and_sample_pose(path, target_T=30, target_V=17, C=2):
    with open(path) as f:
        data = json.load(f).get("data", [])
    T = len(data)
    if T < target_T:
        return None  

    indices = np.linspace(0, T - 1, target_T).astype(int)
    pose_array = np.zeros((target_T, 1, target_V, C))

    for i, idx in enumerate(indices):
        frame = data[idx]
        if frame["skeleton"]:
            pose_raw = frame["skeleton"][0]["pose"]
            if len(pose_raw) >= target_V * C:
                pose_raw = pose_raw[:target_V * C]  # 截断为17关键点
                pose = np.array(pose_raw).reshape(target_V, C)
                pose_array[i, 0] = pose
    return pose_array


def process_folder(folder_path, label_json, target_T=30):
    with open(label_json) as f:
        label_map = json.load(f)

    x_list, y_list = [], []
    total = 0
    kept = 0

    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith(".json"):
            continue
        total += 1
        sample_id = os.path.splitext(fname)[0]
        json_path = os.path.join(folder_path, fname)

        pose_array = load_and_sample_pose(json_path, target_T=target_T)
        if pose_array is not None:
            x_list.append(pose_array)
            y_list.append(label_map[sample_id]["label_index"])
            print(f"Loaded {sample_id} with shape {pose_array.shape}, label: {label_map[sample_id]['label_index']}")
            kept += 1

    print(f" {folder_path}: Kept {kept}/{total} samples with ≥{target_T} frames")
    return np.array(x_list), np.array(y_list)


# 路径设置（修改为你自己的路径）
train_folder = "/home/awsdjikl/TrackNetV3/GCN/data/kinetics_train"
val_folder = "/home/awsdjikl/TrackNetV3/GCN/data/kinetics_val"
train_label_file = "/home/awsdjikl/TrackNetV3/GCN/data/kinetics_train_label.json"
val_label_file = "/home/awsdjikl/TrackNetV3/GCN/data/kinetics_val_label.json"

# 处理数据
x_train, y_train = process_folder(train_folder, train_label_file)
x_test, y_test = process_folder(val_folder, val_label_file)

# 保存 npz
np.savez("/home/awsdjikl/TrackNetV3/GCN/data/V1.npz", x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

print("\n🎉 Saved V1.npz")
print(f"x_train shape: {x_train.shape}, y_train: {y_train.shape}")
print(f"x_test shape: {x_test.shape}, y_test: {y_test.shape}")
