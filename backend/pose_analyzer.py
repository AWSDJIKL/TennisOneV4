"""
Sport Vision — 姿态分析模块
基于 YOLO (Ultralytics) 的人体关键点检测与生物力学分析
"""

import math
import numpy as np
from collections import deque
from pathlib import Path
from typing import Optional
from ultralytics import YOLO


class PoseAnalyzer:
    """封装 YOLO Pose 模型，提供关键点提取和生物力学分析"""

    # 保持与系统其他部分兼容，使用原 MediaPipe 的索引体系作为对外接口
    # 躯干： 11-12, 11-23, 12-24, 23-24
    SKELETON_CONNECTIONS = [
        # 躯干
        (11, 12),  # 左肩-右肩
        (11, 23),  # 左肩-左髋
        (12, 24),  # 右肩-右髋
        (23, 24),  # 左髋-右髋
        # 左臂
        (11, 13),  # 左肩-左肘
        (13, 15),  # 左肘-左腕
        # 右臂
        (12, 14),  # 右肩-右肘
        (14, 16),  # 右肘-右腕
        # 左腿
        (23, 25),  # 左髋-左膝
        (25, 27),  # 左膝-左踝
        # 右腿
        (24, 26),  # 右髋-右膝
        (26, 28),  # 右膝-右踝
    ]

    # YOLO COCO 格式 ID 到系统(MediaPipe格式) ID 的映射
    YOLO_TO_MP = {
        0: 0,  # nose
        5: 11,  # left_shoulder
        6: 12,  # right_shoulder
        7: 13,  # left_elbow
        8: 14,  # right_elbow
        9: 15,  # left_wrist
        10: 16,  # right_wrist
        11: 23,  # left_hip
        12: 24,  # right_hip
        13: 25,  # left_knee
        14: 26,  # right_knee
        15: 27,  # left_ankle
        16: 28,  # right_ankle
    }

    # 关键点名称映射 (兼容系统格式)
    LANDMARK_NAMES = {
        0: "nose",
        11: "left_shoulder",
        12: "right_shoulder",
        13: "left_elbow",
        14: "right_elbow",
        15: "left_wrist",
        16: "right_wrist",
        23: "left_hip",
        24: "right_hip",
        25: "left_knee",
        26: "right_knee",
        27: "left_ankle",
        28: "right_ankle",
    }

    # 要分析的关键关节角度
    JOINT_ANGLES = {
        "left_elbow": (11, 13, 15),
        "right_elbow": (12, 14, 16),
        "left_shoulder": (13, 11, 23),
        "right_shoulder": (14, 12, 24),
        "left_knee": (23, 25, 27),
        "right_knee": (24, 26, 28),
        "left_hip": (11, 23, 25),
        "right_hip": (12, 24, 26),
    }

    def __init__(self, history_size: int = 30):
        # 初始化 YOLO Pose 模型，自动下载 yolo11n-pose.pt（如果本地没有）
        model_name = "yolo11n-pose.pt"
        import torch
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_name)
        self.model.to(device)
        self.device = device

        self.history_size = history_size
        # 关键点历史记录（用于速度/加速度计算）
        self.keypoint_history: deque = deque(maxlen=history_size)
        # 重心轨迹
        self.center_of_mass_history: deque = deque(maxlen=history_size * 2)
        self.frame_count = 0

    def process_frame(self, frame_rgb: np.ndarray) -> Optional[dict]:
        """
        处理单帧，返回分析结果
        """
        h, w = frame_rgb.shape[:2]

        # 运行 YOLOv11 推理
        # Ultralytics 接收 numpy array 默认是 BGR 格式，并将它内部转为 RGB，所以这里先把传入的 RGB 转为 BGR
        frame_bgr = frame_rgb[..., ::-1]
        # 返回结果是一个 list，取第 0 个即当前帧的结果
        results = self.model(frame_bgr, verbose=False, device=self.device)

        if (
            len(results) == 0
            or results[0].keypoints is None
            or len(results[0].keypoints.data) == 0
        ):
            return None

        # 提取第一个人（置信度最高）的关键点数据
        # kps_data 形状近似为 [17, 3]，(x, y, visibility) 像素坐标
        kps_data = results[0].keypoints.data[0].cpu().numpy()

        if len(kps_data) < 17:
            return None

        # 1. 提取关键点（转化为系统原本兼容的 MediaPipe ID 和格式）
        keypoints = []
        landmarks_dict = {}  # 内部辅助字典，用于快速查找计算角度及距离

        for yolo_id, mp_id in self.YOLO_TO_MP.items():
            if yolo_id < len(kps_data):
                x, y, conf = kps_data[yolo_id]
                keypoints.append(
                    {
                        "id": mp_id,
                        "name": self.LANDMARK_NAMES[mp_id],
                        "x": float(x),
                        "y": float(y),
                        "z": 0.0,  # YOLO 不输出 3D Z轴信息
                        "visibility": float(conf),
                    }
                )
                # 构建用于内部计算的简化对象结构
                landmarks_dict[mp_id] = type(
                    "Landmark",
                    (),
                    {"x": float(x), "y": float(y), "visibility": float(conf)},
                )

        if not keypoints:
            return None

        # 过滤低置信度
        avg_visibility = np.mean([kp["visibility"] for kp in keypoints])
        if avg_visibility < 0.3:
            return None

        # 2. 计算关节角度
        joint_angles = {}
        for name, (a, b, c) in self.JOINT_ANGLES.items():
            angle = self._calculate_angle_from_landmarks(landmarks_dict, a, b, c)
            if angle is not None:
                joint_angles[name] = round(angle, 1)

        # 3. 计算重心 (髋部的中点)
        if 23 in landmarks_dict and 24 in landmarks_dict:
            com_x = (landmarks_dict[23].x + landmarks_dict[24].x) / 2
            com_y = (landmarks_dict[23].y + landmarks_dict[24].y) / 2
        else:
            com_x, com_y = w / 2, h / 2
        center_of_mass = {"x": round(com_x, 1), "y": round(com_y, 1)}
        self.center_of_mass_history.append(center_of_mass)

        # 4. 记录关键点历史
        kp_dict = {kp["id"]: (kp["x"], kp["y"]) for kp in keypoints}
        self.keypoint_history.append(kp_dict)
        self.frame_count += 1

        # 5. 生物力学分析
        biomechanics = self._analyze_biomechanics(landmarks_dict, w, h)

        return {
            "keypoints": keypoints,
            "skeleton": self.SKELETON_CONNECTIONS,
            "joint_angles": joint_angles,
            "biomechanics": biomechanics,
            "center_of_mass": center_of_mass,
            "confidence": round(avg_visibility, 2),
            "raw_kps": kps_data.tolist()  # 额外保留 17 关键点给 GCN
        }

    def _calculate_angle_from_landmarks(
        self, landmarks_dict, a_idx, b_idx, c_idx
    ) -> Optional[float]:
        """计算三个关键点形成的角度（以 b 为顶点）"""
        try:
            if (
                a_idx not in landmarks_dict
                or b_idx not in landmarks_dict
                or c_idx not in landmarks_dict
            ):
                return None
            a = landmarks_dict[a_idx]
            b = landmarks_dict[b_idx]
            c = landmarks_dict[c_idx]
            ba = np.array([a.x - b.x, a.y - b.y])
            bc = np.array([c.x - b.x, c.y - b.y])
            cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
            angle = np.arccos(np.clip(cosine, -1.0, 1.0))
            return math.degrees(angle)
        except Exception:
            return None

    def _analyze_biomechanics(self, landmarks_dict, w: int, h: int) -> dict:
        """运动生物力学分析"""
        result = {
            "wrist_speed": 0.0,
            "body_lean": 0.0,
            "knee_bend": 0.0,
            "arm_extension": 0.0,
            "symmetry_score": 0.0,
        }

        if len(self.keypoint_history) < 2:
            return result

        prev = self.keypoint_history[-2]
        curr = self.keypoint_history[-1]

        # 手腕速度（取左右手中速度更大的）
        wrist_speeds = []
        for wrist_id in [15, 16]:
            if wrist_id in prev and wrist_id in curr:
                dx = curr[wrist_id][0] - prev[wrist_id][0]
                dy = curr[wrist_id][1] - prev[wrist_id][1]
                speed = math.sqrt(dx * dx + dy * dy)
                wrist_speeds.append(speed)
        result["wrist_speed"] = round(max(wrist_speeds) if wrist_speeds else 0, 1)

        # 身体倾斜角（脊柱与垂直线的夹角）
        try:
            if (
                11 in landmarks_dict
                and 12 in landmarks_dict
                and 23 in landmarks_dict
                and 24 in landmarks_dict
            ):
                mid_shoulder = np.array(
                    [
                        (landmarks_dict[11].x + landmarks_dict[12].x) / 2,
                        (landmarks_dict[11].y + landmarks_dict[12].y) / 2,
                    ]
                )
                mid_hip = np.array(
                    [
                        (landmarks_dict[23].x + landmarks_dict[24].x) / 2,
                        (landmarks_dict[23].y + landmarks_dict[24].y) / 2,
                    ]
                )
                spine = mid_shoulder - mid_hip
                vertical = np.array([0, -1])
                cos_angle = np.dot(spine, vertical) / (np.linalg.norm(spine) + 1e-8)
                lean_angle = math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))
                result["body_lean"] = round(lean_angle, 1)
        except Exception:
            pass

        # 膝盖弯曲度（取双膝平均）
        knee_angles = []
        for name in ["left_knee", "right_knee"]:
            a_id, b_id, c_id = self.JOINT_ANGLES[name]
            angle = self._calculate_angle_from_landmarks(
                landmarks_dict, a_id, b_id, c_id
            )
            if angle is not None:
                knee_angles.append(angle)
        result["knee_bend"] = round(180 - np.mean(knee_angles) if knee_angles else 0, 1)

        # 手臂伸展度（肘部角度，越接近 180 越伸展）
        elbow_angles = []
        for name in ["left_elbow", "right_elbow"]:
            a_id, b_id, c_id = self.JOINT_ANGLES[name]
            angle = self._calculate_angle_from_landmarks(
                landmarks_dict, a_id, b_id, c_id
            )
            if angle is not None:
                elbow_angles.append(angle)
        result["arm_extension"] = round(np.mean(elbow_angles) if elbow_angles else 0, 1)

        # 对称性评分（0-100，左右对称性）
        try:
            if (
                11 in landmarks_dict
                and 12 in landmarks_dict
                and 23 in landmarks_dict
                and 24 in landmarks_dict
            ):
                # 之前 MediaPipe 的 y 是 0~1 的归一化值，YOLO 返回的是像素值，所以在计算对称性时除以 h 变回归一化值
                left_shoulder_y = landmarks_dict[11].y / h
                right_shoulder_y = landmarks_dict[12].y / h
                left_hip_y = landmarks_dict[23].y / h
                right_hip_y = landmarks_dict[24].y / h
                shoulder_diff = abs(left_shoulder_y - right_shoulder_y)
                hip_diff = abs(left_hip_y - right_hip_y)
                asymmetry = (shoulder_diff + hip_diff) / 2
                symmetry = max(0, 100 - asymmetry * 500)
                result["symmetry_score"] = round(symmetry, 1)
        except Exception:
            result["symmetry_score"] = 0.0

        return result

    def get_trajectory(self) -> list:
        """返回重心轨迹"""
        return list(self.center_of_mass_history)

    def reset(self):
        """重置状态"""
        self.keypoint_history.clear()
        self.center_of_mass_history.clear()
        self.frame_count = 0

    def close(self):
        """释放资源"""
        pass
