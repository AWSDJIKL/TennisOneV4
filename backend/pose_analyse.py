import math
import torch
import numpy as np
import cv2
import time
import os
import tqdm
from PIL import Image

# import yolo.tennis_detector as td
# from transformers import (
#     AutoProcessor,
#     RTDetrForObjectDetection,
#     VitPoseForPoseEstimation,
# )
import six

device = "cuda"
from PIL import Image, ImageDraw, ImageFont
import requests, re
import backend.n1n_api as n1n_api

PROBLEM_PARTS = ["肘部", "腋下", "重心", "躯干", "膝部"]


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at b formed by a-b-c in degrees. Returns nan if invalid."""
    va = a - b
    vc = c - b
    na = np.linalg.norm(va)
    nc = np.linalg.norm(vc)
    if na == 0 or nc == 0:
        return float("nan")
    cosv = float(np.dot(va, vc) / (na * nc))
    cosv = max(-1.0, min(1.0, cosv))
    return math.degrees(math.acos(cosv))


def get_elbow_angle(keypoints):
    """
    计算肘部角度

    :param keypoints: Description
    """
    left_elbow_angle = _angle_deg(keypoints[5], keypoints[7], keypoints[9])
    right_elbow_angle = _angle_deg(keypoints[6], keypoints[8], keypoints[10])
    return left_elbow_angle, right_elbow_angle
    pass


def get_underarm_angle(keypoints):
    """
    计算腋下角度

    :param keypoints: Description
    """
    left_underarm_angle = _angle_deg(keypoints[7], keypoints[5], keypoints[11])
    right_underarm_angle = _angle_deg(keypoints[8], keypoints[6], keypoints[12])
    return left_underarm_angle, right_underarm_angle
    pass


def get_torso_angle(keypoints):
    """
    计算躯干角度

    :param keypoints: Description
    """
    if keypoints is None or keypoints.shape[0] < 17:
        return float("nan")
    coords = keypoints[:, :2]
    mid_shoulder = (coords[5] + coords[6]) / 2.0
    mid_hip = (coords[11] + coords[12]) / 2.0
    torso = mid_shoulder - mid_hip  # vector hip->shoulder (dx, dy)
    norm = np.linalg.norm(torso)
    if norm == 0:
        return float("nan")

    # 基本角度（相对于竖直方向，正为向右倾）
    ang_rad = math.atan2(torso[0], -torso[1])
    ang_deg = math.degrees(ang_rad)

    # 判定面向方向：优先使用 nose(0)，否则用肩膀左右关系作为退化方案
    facing_sign = 1.0
    # 如果右肩在图像右侧则视为朝右（退化启发式）
    facing_sign = 1.0 if (coords[6, 0] - coords[5, 0]) > 0 else -1.0

    # 将“右倾”为正的角度转换为“前倾”为正（若朝右为前，则保持；若朝左为前，则取相反号）
    torso_angle = ang_deg * facing_sign
    return torso_angle
    pass


def get_knee_angle(keypoints):
    """
    计算膝部角度

    :param keypoints: Description
    """

    left_knee_angle = _angle_deg(keypoints[11], keypoints[13], keypoints[15])
    right_knee_angle = _angle_deg(keypoints[12], keypoints[14], keypoints[16])
    return left_knee_angle, right_knee_angle
    pass


def normalize_keypoints_2d(kp, require_points=(5, 6, 11, 12)):
    """
    修改版 2D 归一化: 仅做 translate (pelvis 移到原点) -> scale (按躯干长度缩放)
    保留原始的倾斜角度（不执行旋转操作），这样不会破坏躯干和重心分析。
    """
    kp = np.asarray(kp, dtype=float)
    if kp.ndim == 1 and kp.size % 2 == 0:
        kp = kp.reshape(-1, 2)
    N = kp.shape[0]
    out = np.full_like(kp, np.nan, dtype=float)

    # helpers to get valid point or NaN
    def valid_point(idx):
        if idx < 0 or idx >= N:
            return None
        x, y = kp[idx]
        if np.isnan(x) or np.isnan(y):
            return None
        return np.array([x, y], dtype=float)

    l_sh = valid_point(require_points[0])
    r_sh = valid_point(require_points[1])
    l_hip = valid_point(require_points[2])
    r_hip = valid_point(require_points[3])

    # compute mid_shoulder / mid_hip robustly
    shoulders = [p for p in (l_sh, r_sh) if p is not None]
    hips = [p for p in (l_hip, r_hip) if p is not None]

    if len(hips) == 0:
        valid = kp[~np.isnan(kp).any(axis=1)]
        if valid.size == 0:
            return out
        pelvis = valid.mean(axis=0)
    else:
        pelvis = np.mean(hips, axis=0)

    if len(shoulders) == 0:
        valid = kp[~np.isnan(kp).any(axis=1)]
        mid_shoulder = (
            valid.mean(axis=0) if valid.size else pelvis + np.array([0.0, -1.0])
        )
    else:
        mid_shoulder = np.mean(shoulders, axis=0)

    # 1. 仅平移: 以 pelvis 为坐标原点 (由此计算出的重心将是相对于骨盆的相对重心)
    trans = pelvis
    kp_t = kp.copy()
    for i in range(N):
        if not np.isnan(kp_t[i, 0]) and not np.isnan(kp_t[i, 1]):
            kp_t[i] = kp_t[i] - trans

    # 2. 计算躯干长度用于尺度归一化
    torso = mid_shoulder - pelvis
    torso_len = np.linalg.norm(torso) if torso is not None else 0.0
    if torso_len < 1e-6:
        torso_len = 1.0

    # 3. 缩放并输出 (不旋转，保留躯干倾角)
    for i in range(N):
        if np.isnan(kp_t[i, 0]) or np.isnan(kp_t[i, 1]):
            out[i] = np.array([np.nan, np.nan])
        else:
            out[i] = kp_t[i] / torso_len  # 统一相对于躯干的长度单位

    return out


def get_angle_groups(keypoints_list):
    angle_groups = [[] for _ in range(9)]
    for keypoints in keypoints_list:
        if keypoints is None:
            # angle_groups.append(None)
            continue
        if keypoints.shape[0] < 17:
            # angle_groups.append(None)
            continue
        keypoints = normalize_keypoints_2d(keypoints)

        left_elbow_angle, right_elbow_angle = get_elbow_angle(keypoints)
        left_underarm_angle, right_underarm_angle = get_underarm_angle(keypoints)
        torso_angle = get_torso_angle(keypoints)
        left_knee_angle, right_knee_angle = get_knee_angle(keypoints)
        mass_x, mass_y = compute_center_of_mass(keypoints)
        angle_groups[0].append(left_elbow_angle)
        angle_groups[1].append(right_elbow_angle)

        angle_groups[2].append(left_underarm_angle)
        angle_groups[3].append(right_underarm_angle)

        angle_groups[4].append(mass_x)
        angle_groups[5].append(mass_y)

        angle_groups[6].append(torso_angle)

        angle_groups[7].append(left_knee_angle)
        angle_groups[8].append(right_knee_angle)

    return angle_groups


def compute_center_of_mass(
    keypoints, confidences=None, method="weighted", conf_thresh=0.2
):
    """
    计算单帧人体重心 (x, y)。
    keypoints: (N,2) 或可转换为该形状的数组（N 应为 17 对 COCO）
    confidences: 可选 (N,) 置信度数组，用于滤除低置信点
    method: "centroid" 或 "weighted"
    返回 (x, y) 或 (np.nan, np.nan)（无有效点时）
    """
    kp = np.asarray(keypoints, dtype=float)
    if kp.size == 0:
        return (float("nan"), float("nan"))
    if kp.ndim == 1:
        if kp.size % 2 != 0:
            return (float("nan"), float("nan"))
        kp = kp.reshape(-1, 2)
    # 常见 COCO 17 keypoints 顺序假设 (0..16)
    N = kp.shape[0]
    if N < 2:
        return (float("nan"), float("nan"))

    conf = None
    if confidences is not None:
        conf = np.asarray(confidences, dtype=float)
        if conf.ndim != 1 or conf.size != N:
            conf = None

    valid = ~np.isnan(kp[:, 0]) & ~np.isnan(kp[:, 1])
    if conf is not None:
        valid &= conf >= conf_thresh
    if not np.any(valid):
        return (float("nan"), float("nan"))

    pts = kp[valid]

    if method == "centroid":
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        return (cx, cy)

    # weighted 方法：为 17 个 COCO 关键点设置经验权重（可根据实际调整）
    # indices: 0 nose,1 l_eye,2 r_eye,3 l_ear,4 r_ear,5 l_shoulder,6 r_shoulder,7 l_elbow,8 r_elbow,
    # 9 l_wrist,10 r_wrist,11 l_hip,12 r_hip,13 l_knee,14 r_knee,15 l_ankle,16 r_ankle
    default_weights = np.array(
        [
            0.30,
            0.05,
            0.05,
            0.02,
            0.02,
            1.00,
            1.00,
            0.60,
            0.60,
            0.40,
            0.40,
            1.50,
            1.50,
            0.90,
            0.90,
            0.70,
            0.70,
        ],
        dtype=float,
    )
    if kp.shape[0] != default_weights.size:
        # 如果不是 17 点，退化为等权 centroid
        return (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))

    weights = default_weights.copy()
    mask = valid.astype(float)  # 1 for valid joints, 0 for invalid
    weights = weights * mask

    if conf is not None:
        weights *= conf  # 按置信度调节权重

    sum_w = float(np.sum(weights))
    if sum_w == 0:
        return (float("nan"), float("nan"))

    weighted_x = np.nansum(kp[:, 0] * weights) / sum_w
    weighted_y = np.nansum(kp[:, 1] * weights) / sum_w
    return (float(weighted_x), float(weighted_y))


def compute_com_trajectory(
    keypoints_list, confidences_list=None, method="weighted", conf_thresh=0.2
):
    """
    为一段序列计算每帧重心轨迹和简单稳定性指标。
    keypoints_list: list of keypoints (每项为 (N,2) 或可转换)
    confidences_list: 可选 list，与 keypoints_list 对应
    返回 dict:
      {
        "trajectory": np.ndarray(shape=(T,2)),
        "valid_mask": np.ndarray(bool, shape=(T,)),
        "std": (std_x, std_y),
        "path_length": float  # 总轨迹长度（像素）
      }
    """
    T = len(keypoints_list)
    traj = np.full((T, 2), np.nan, dtype=float)
    valid = np.zeros((T,), dtype=bool)
    for i in range(T):
        conf = None
        if confidences_list is not None:
            try:
                conf = confidences_list[i]
            except Exception:
                conf = None
        cx, cy = compute_center_of_mass(
            keypoints_list[i], confidences=conf, method=method, conf_thresh=conf_thresh
        )
        if not (np.isnan(cx) or np.isnan(cy)):
            traj[i, 0] = cx
            traj[i, 1] = cy
            valid[i] = True

    if not np.any(valid):
        return {
            "trajectory": traj,
            "valid_mask": valid,
            "std": (np.nan, np.nan),
            "path_length": 0.0,
        }

    valid_pts = traj[valid]
    std_x = float(np.std(valid_pts[:, 0]))
    std_y = float(np.std(valid_pts[:, 1]))

    # path length: 连续有效点间的欧氏距离之和
    pts = valid_pts
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    path_length = float(np.sum(diffs))

    # return {"trajectory": traj, "valid_mask": valid, "std": (std_x, std_y), "path_length": path_length}
    return traj


def sample_keypoints(keypoints_list, sample=15):
    n = len(keypoints_list)
    if n == 0:
        return []
    indices = [i * (n - 1) // (sample - 1) for i in range(sample)]
    return [keypoints_list[i] for i in indices]


def calculate_similarity_score(a, b):
    """
    更稳健的相似度计算：
    - 接受任意可转为 (T, C) 的数组（1D/2D/更高维会被折叠为 (T, C)）
    - 尝试自动处理转置/单列铺平
    - 在时间轴上对标准序列插值重采样到玩家序列长度
    - 计算 mean absolute error -> 映射到 0..100
    """
    A = np.asarray(a, dtype=float)
    B = np.asarray(b, dtype=float)

    def to_tc(x):
        if x.ndim == 0:
            return x.reshape(1, 1)
        if x.ndim == 1:
            # treat as (T, 1)
            return x.reshape(-1, 1)
        if x.ndim == 2:
            return x
        # collapse trailing dims into features: (T, ...)
        return x.reshape(x.shape[0], -1)

    A = to_tc(A)
    B = to_tc(B)

    # try transpose heuristics if columns mismatch
    if A.shape[1] != B.shape[1]:
        if B.T.shape[1] == A.shape[1]:
            B = B.T
        elif A.shape[1] == 1 and B.shape[1] > 1:
            A = np.tile(A, (1, B.shape[1]))
        elif B.shape[1] == 1 and A.shape[1] > 1:
            B = np.tile(B, (1, A.shape[1]))
        else:
            return 0  # incompatible feature dims

    T_A, C = A.shape
    T_B = B.shape[0]

    # resample B in time to match A length
    if T_B != T_A:
        x_old = np.linspace(0.0, 1.0, T_B)
        x_new = np.linspace(0.0, 1.0, T_A)
        B_res = np.empty((T_A, C), dtype=float)
        for j in range(C):
            col = B[:, j]
            if np.all(np.isnan(col)):
                B_res[:, j] = np.nan
                continue
            ok = ~np.isnan(col)
            if ok.sum() == 0:
                B_res[:, j] = np.nan
            elif ok.sum() == 1:
                B_res[:, j] = float(col[ok][0])
            else:
                B_res[:, j] = np.interp(x_new, x_old[ok], col[ok])
        B = B_res

    diff = np.abs(A - B)
    valid = ~np.isnan(diff)
    if not np.any(valid):
        return 0
    mean_abs = float(np.nanmean(diff))
    score_norm = 1.0 - (mean_abs / 90.0)
    score_norm = max(0.0, min(1.0, score_norm))
    return int(round(score_norm * 100.0))


def analyze_problem_and_suggest(angels_group, standard_group):
    # print(angels_group)
    # print(standard_group)
    messages = f"请根据我提供的用户动作角度和标准动作角度序列，分析用户的动作与标准动作的差异，并给出改进建议。从肘部、手腕、躯干、膝部和击球点5个角度分析问题和给出建议，不需要逐帧分析，回复时请直接给出问题和建议的总结，不需要给出分析过程。\n用户的动作角度序列: {angels_group}\n标准的动作序列: {standard_group}"
    url = f"http://127.0.0.1:31000/v1/chat/completions"

    data = {
        "model": "Qwen/Qwen3-VL-4B-Instruct",
        # "messages": ["hello"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": messages},
                ],
            }
        ],
        "max_tokens": 300,
    }
    response = requests.post(url, json=data)
    response_json = response.json()
    text = response_json["choices"][0]["message"]["content"]
    pattern = (
        r"(肘部|手腕|躯干|膝部|击球点)：(.*?)(?=\n\n(?:肘部|手腕|躯干|膝部|击球点)：|$)"
    )
    matches = re.findall(pattern, text, re.DOTALL)
    problems = ""
    suggests = ""
    for part, content in matches:
        content = content.strip()
        # 分离问题与建议
        if "建议" in content:
            problem, suggestion = content.split("建议", 1)
            problem = problem.strip()[:-1]
            suggestion = f"建议{suggestion.strip()}"[2:-1]
        else:
            problem = content
            suggestion = "无明确建议"
        problems += f"{part}问题: {problem}\n"
        suggests += f"{part}建议: {suggestion}\n"
    return problems, suggests


def AnalysePose(player_groups, standar_groups, video_path):
    """
    计算玩家动作与标准动作的相似度分数
    给出六维分数和动作三阶段分数

    :param player_groups: Description
    :param standar_groups: Description
    """

    # 先对玩家动作序列和标准动作序列进行采样（如15帧），得到采样后的角度序列
    # print("player_groups length:", player_groups.shape)
    player_sampled = sample_keypoints(player_groups, sample=15)
    player_groups = get_angle_groups(player_sampled)
    player_sampled = np.array(player_groups)
    print(player_sampled.shape)  # (9,15)
    print(standar_groups.shape)  # (9,15)
    sixe_scores = []
    three_stage_score = []
    # 1. 计算动作标准性分数，用玩家动作序列和标准序列的相似度作为分数
    ready_score = calculate_similarity_score(
        player_sampled[:, :5], standar_groups[:, :5]
    )
    hit_score = calculate_similarity_score(
        player_sampled[:, 5:10], standar_groups[:, 5:10]
    )
    finish_score = calculate_similarity_score(
        player_sampled[:, 10:], standar_groups[:, 10:]
    )
    total_score = calculate_similarity_score(player_sampled, standar_groups)
    three_stage_score.extend([total_score, ready_score, hit_score, finish_score])
    print("three_stage_score:", three_stage_score)
    sixe_scores.append(total_score)
    print("total_score:", total_score)
    # 2. 计算肘部分数，计算肘部角度的相似度作为肘部分数
    # player_elbow_angle = []
    # standar_elbow_angle = []
    # for i in range(len(player_sampled)):
    # player_elbow_angle.append(get_elbow_angle(player_sampled[i]))
    # standar_elbow_angle.append(get_elbow_angle(standar_groups[i]))
    elbow_score = calculate_similarity_score(
        player_sampled[:2, :], standar_groups[:2, :]
    )  # 选取肘部关键点的坐标
    sixe_scores.append(elbow_score)
    print("elbow_score:", elbow_score)
    # 3. 计算腋下分数
    # player_underarm_angle = []
    # standar_underarm_angle = []
    # for i in range(len(player_sampled)):
    #     player_underarm_angle.append(get_underarm_angle(player_sampled[i]))
    #     standar_underarm_angle.append(get_underarm_angle(standar_groups[i]))
    underarm_score = calculate_similarity_score(
        player_sampled[2:4, :], standar_groups[2:4, :]
    )  # 选取腋下关键点的坐标
    sixe_scores.append(underarm_score)
    print("underarm_score:", underarm_score)
    # 4. 姿态重心分数
    # player_mass = []
    # standar_mass = []
    # for i in range(len(player_sampled)):
    #     player_mass.append(compute_center_of_mass(player_sampled[i]))
    #     standar_mass.append(compute_center_of_mass(standar_groups[i]))
    mass_score = calculate_similarity_score(
        player_sampled[4:6, :], standar_groups[4:6, :]
    )  # 选取姿态重心关键点的坐标
    sixe_scores.append(mass_score)
    print("mass_score:", mass_score)
    # 5.躯干
    # player_torso_angle = []
    # standar_torso_angle = []
    # for i in range(len(player_sampled)):
    #     player_torso_angle.append(get_torso_angle(player_sampled[i]))
    #     standar_torso_angle.append(get_torso_angle(standar_groups[i]))
    torso_score = calculate_similarity_score(
        player_sampled[6:7, :], standar_groups[6:7, :]
    )  # 选取躯干关键点的坐标
    sixe_scores.append(torso_score)
    print("torso_score:", torso_score)
    # 6.膝部
    # player_knee_angle = []
    # standar_knee_angle = []
    # for i in range(len(player_sampled)):
    #     player_knee_angle.append(get_knee_angle(player_sampled[i]))
    #     standar_knee_angle.append(get_knee_angle(standar_groups[i]))
    knee_score = calculate_similarity_score(
        player_sampled[7:, :], standar_groups[7:, :]
    )  # 选取膝部关键点的坐标
    sixe_scores.append(knee_score)
    print("knee_score:", knee_score)

    # 对比5个分数，找出最差的部分，给出改进建议

    problem_part = PROBLEM_PARTS[
        np.argmin([elbow_score, underarm_score, mass_score, torso_score, knee_score])
    ]
    print("problem_part:", problem_part)
    # problems, suggests = analyze_problem_and_suggest(player_sampled, standar_groups)
    start_time = time.time()
    problems, suggests = n1n_api.AnalyseVideo(video_path, problem_part)
    print(f"分析问题和建议耗时: {time.time() - start_time:.2f} 秒")
    return sixe_scores, three_stage_score, problems, suggests
