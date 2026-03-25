"""
读取视频，提取每一帧骨骼数据，预处理，按照30帧窗口滑动的形式输入到GCN中进行分类，给出分类结果。
"""

import torch
from GCN.main import Processor, get_parser
import GCN.dataset.tools as tools
from GCN import main
import numpy as np
import random
import os
import yaml

COCO_PAIRS = [
    (1, 6),
    (2, 1),
    (3, 1),
    (4, 2),
    (5, 3),
    (6, 7),
    (7, 1),
    (8, 6),
    (9, 7),
    (10, 8),
    (11, 9),
    (12, 6),
    (13, 7),
    (14, 12),
    (15, 13),
    (16, 14),
    (17, 15),
]

CONFIG_LIST = [
    "./GCN/config/j.yaml",
    "./GCN/config/b.yaml",
    "./GCN/config/jm.yaml",
    "./GCN/config/bm.yaml",
]
WORK_DIR_LIST = [
    "./GCN/work_dir/j",
    "./GCN/work_dir/b",
    "./GCN/work_dir/jm",
    "./GCN/work_dir/bm",
]
MODEL_WEIGHT_LIST = [0.06725512710722695, 2.0, 0.0001, 2.0]

MODEL_NAME = "runs-65-715.pt"

BONE_LIST = [False, True, False, True]
VEL_LIST = [False, False, True, True]
CATEGORIES = ["正手", "反手", "其他"]
WINDOW_SIZE = 30
MIN_FRAME_PERCENTAGE = 0.4

MODEL_LIST = []


def load_models():
    """懒加载模型，避免 import 时执行"""
    global MODEL_LIST
    if MODEL_LIST:
        return

    print("Loading GCN models...")
    for c, wd in zip(CONFIG_LIST, WORK_DIR_LIST):
        model_parser = get_parser()
        # 关键修正：传入空列表 []，避免解析 sys.argv (即 uvicorn 的参数)
        p = model_parser.parse_args([])
        p.phase = "test"
        p.config = c
        with open(p.config, "r") as f:
            default_arg = yaml.safe_load(f)

        # 验证参数键
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                # print("WRONG ARG: {}".format(k))
                pass  # 忽略多余参数警告

        model_parser.set_defaults(**default_arg)
        # 再次传入空列表
        model_args = model_parser.parse_args([])
        model_args.work_dir = wd
        model_args.phase = "test"
        model_args.weights = os.path.join(wd, MODEL_NAME)
        # print(model_args)

        processor = Processor(model_args)
        processor.model.eval()
        MODEL_LIST.append(processor.model)
    print(f"Loaded {len(MODEL_LIST)} models.")


def data_normalization(input, bone=False, vel=False):
    # 数据预处理
    keypoints = np.array(input)

    # 修复之前提到的 IndexError: 确保输入长度匹配 WINDOW_SIZE
    num_frames = keypoints.shape[0]
    if num_frames >= WINDOW_SIZE:
        # 取中间或最后 WINDOW_SIZE 帧，这里取最后30帧
        keypoints = keypoints[num_frames - WINDOW_SIZE : num_frames]
    else:
        # 补零
        pad = np.zeros((WINDOW_SIZE - num_frames, 17, 2), dtype=keypoints.dtype)
        keypoints = np.concatenate([pad, keypoints], axis=0)

    # keypoints 现在形状一定是 (30, 17, 2)
    # 继续原有归一化逻辑...

    # 1. 提取 x 和 y 坐标
    x = keypoints[:, :, 0]
    y = keypoints[:, :, 1]
    # 2. 计算每帧的极值
    x_min = np.min(x, axis=1, keepdims=True)
    x_max = np.max(x, axis=1, keepdims=True)
    y_min = np.min(y, axis=1, keepdims=True)
    y_max = np.max(y, axis=1, keepdims=True)
    # 3. 计算归一化范围
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_range[x_range == 0] = 1
    y_range[y_range == 0] = 1
    # 4. 归一化
    x_norm = (x - x_min) / x_range
    y_norm = (y - y_min) / y_range
    # 5. 组合
    keypoints_norm = np.stack([x_norm, y_norm], axis=2)

    data_numpy = np.zeros((WINDOW_SIZE, 1, 17, 2))
    for i in range(WINDOW_SIZE):
        data_numpy[i, 0, :, :] = keypoints_norm[i]

    data_numpy = data_numpy.transpose(3, 0, 2, 1)  # C,T,V,M

    # valid_frame_num 逻辑在全非零时可能失效，这里简化处理，假设所有帧有效如果已经padding过
    valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)

    data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, [0.5, 1], 5)

    if bone:
        bone_data_numpy = np.zeros_like(data_numpy)
        for v1, v2 in COCO_PAIRS:
            bone_data_numpy[:, :, v1 - 1] = (
                data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            )
        data_numpy = bone_data_numpy
    if vel:
        data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
        data_numpy[:, -1] = 0

    data_numpy = data_numpy - np.tile(data_numpy[:, :, 0:1, :], (1, 1, 17, 1))
    pose = torch.tensor(data_numpy).unsqueeze(0).float()
    return pose


def get_cls(input):
    """
    窗口大小为30帧，输入30帧的连续骨骼进行分类
    """
    # 确保模型已加载
    load_models()

    result = []
    # 确保MODEL_LIST有内容
    if not MODEL_LIST:
        print("Error: No models loaded!")
        return 0  # 返回默认类别索引

    for i, (model, weight, bone, vel) in enumerate(
        zip(MODEL_LIST, MODEL_WEIGHT_LIST, BONE_LIST, VEL_LIST)
    ):
        x = data_normalization(input, bone=bone, vel=vel).cuda()
        model.eval()
        # print("-" * 20)
        with torch.no_grad():  # 推理时不需要梯度
            logits = model(x)

        # print(f"Model {i+1} logits shape: {logits.size()}")
        logits = torch.nn.functional.softmax(logits, dim=1) * torch.tensor(weight)
        # print(CATEGORIES[torch.max(logits.data, 1)[1].item()])
        result.append(logits.cpu())

    result = torch.stack(result, dim=0)
    result = torch.mean(result, dim=0)
    # print(result)
    _, predict_label = torch.max(result.data, 1)
    return (
        predict_label.item()
    )  # 返回索引，外部再转CATEGORIES[idx] (注意保持和action_recognizer一致)
