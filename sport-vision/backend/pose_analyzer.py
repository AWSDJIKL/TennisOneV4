"""
Sport Vision — 姿态分析模块
基于 YOLOv11 Pose 的人体关键点检测与生物力学分析
"""

import math
import numpy as np
import torch
from ultralytics import YOLO
from collections import deque
from pathlib import Path
from typing import Optional


class PoseAnalyzer:
    """封装 YOLOv11 Pose，提供关键点提取和生物力学分析"""

    # 骨骼连接定义 (使用 COCO 17 关键点索引)
    SKELETON_CONNECTIONS = [
        # 躯干
        (5, 6),  # 左肩-右肩
        (5, 11),  # 左肩-左髋
        (6, 12),  # 右肩-右髋
        (11, 12),  # 左髋-右髋
        # 左臂
        (5, 7),  # 左肩-左肘
        (7, 9),  # 左肘-左腕
        # 右臂
        (6, 8),  # 右肩-右肘
        (8, 10),  # 右肘-右腕
        # 左腿
        (11, 13),  # 左髋-左膝
        (13, 15),  # 左膝-左踝
        # 右腿
        (12, 14),  # 右髋-右膝
        (14, 16),  # 右膝-右踝
    ]

    # 关键点名称映射 (COCO 17)
    LANDMARK_NAMES = {
        0: "nose",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle",
    }

    # 要分析的关键关节角度
    JOINT_ANGLES = {
        "left_elbow": (5, 7, 9),
        "right_elbow": (6, 8, 10),
        "left_shoulder": (7, 5, 11),
        "right_shoulder": (8, 6, 12),
        "left_knee": (11, 13, 15),
        "right_knee": (12, 14, 16),
        "left_hip": (5, 11, 13),
        "right_hip": (6, 12, 14),
    }

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        history_size: int = 30,
    ):

        # 初始化 YOLO11 姿态估计模型
        # model_path = "yolo11n-pose.pt" # 自动下载或指定本地路径
        model_path = "yolo11l-pose.pt"  # 自动下载或指定本地路径
        self.model = YOLO(model_path)

        # 显卡加速配置
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.min_detection_confidence = min_detection_confidence

        self.history_size = history_size
        # 关键点历史记录（用于速度/加速度计算）
        self.keypoint_history: deque = deque(maxlen=history_size)
        # 重心轨迹
        self.center_of_mass_history: deque = deque(maxlen=history_size * 2)
        self.frame_count = 0

    def process_frame(self, frame_rgb: np.ndarray) -> Optional[dict]:
        """
        处理单帧，返回分析结果

        Returns:
            {
                "keypoints": [{x, y, z, visibility, name}, ...],
                "skeleton": [[p1_idx, p2_idx], ...],
                "joint_angles": {name: angle_degrees, ...},
                "biomechanics": {velocity, acceleration, symmetry, ...},
                "center_of_mass": {x, y},
                "confidence": float,
            }
        """
        h, w = frame_rgb.shape[:2]

        # 检测
        results = self.model(frame_rgb, verbose=False, device=self.device)

        if (
            not results
            or not results[0].keypoints
            or results[0].keypoints.data.shape[0] == 0
        ):
            return None

        # 如果检测到多个人，取置信度最高的或者第一个
        keypoints_data = (
            results[0].keypoints.data[0].cpu().numpy()
        )  # shape: (17, 3) -> x, y, conf

        # 1. 提取关键点（像素坐标）
        keypoints = []
        for idx in self.LANDMARK_NAMES:
            if idx < len(keypoints_data):
                x, y, conf = keypoints_data[idx]
                keypoints.append(
                    {
                        "id": idx,
                        "name": self.LANDMARK_NAMES[idx],
                        "x": float(x),
                        "y": float(y),
                        "z": 0.0,  # YOLO姿态估计通常没有Z轴坐标
                        "visibility": float(conf),
                    }
                )

        if not keypoints:
            return None

        # 过滤低置信度
        avg_visibility = np.mean([kp["visibility"] for kp in keypoints])
        if avg_visibility < self.min_detection_confidence:
            return None

        # 为了计算关节角度和生物力学，构建一个与MediaPipe结构类似的对象
        class DummyLandmark:
            pass

        landmarks = []
        for i in range(17):
            lm = DummyLandmark()
            if i < len(keypoints_data):
                lm.x = float(keypoints_data[i][0] / w)
                lm.y = float(keypoints_data[i][1] / h)
                lm.z = 0.0
                lm.visibility = float(keypoints_data[i][2])
            else:
                lm.x, lm.y, lm.z, lm.visibility = 0.0, 0.0, 0.0, 0.0
            landmarks.append(lm)

        # 2. 计算关节角度
        joint_angles = {}
        for name, (a, b, c) in self.JOINT_ANGLES.items():
            angle = self._calculate_angle_from_landmarks(landmarks, a, b, c)
            if angle is not None:
                joint_angles[name] = float(round(angle, 1))

        # 3. 计算重心
        if 11 < len(landmarks) and 12 < len(landmarks):
            com_x = (landmarks[11].x + landmarks[12].x) / 2 * w
            com_y = (landmarks[11].y + landmarks[12].y) / 2 * h
        else:
            com_x, com_y = w / 2, h / 2
        center_of_mass = {"x": float(round(com_x, 1)), "y": float(round(com_y, 1))}
        self.center_of_mass_history.append(center_of_mass)

        # 4. 记录关键点历史
        kp_dict = {kp["id"]: (kp["x"], kp["y"]) for kp in keypoints}
        self.keypoint_history.append(kp_dict)
        self.frame_count += 1

        # 5. 生物力学分析
        biomechanics = self._analyze_biomechanics(landmarks, w, h)

        return {
            "keypoints": keypoints,
            "skeleton": self.SKELETON_CONNECTIONS,
            "joint_angles": joint_angles,
            "biomechanics": biomechanics,
            "center_of_mass": center_of_mass,
            "confidence": float(round(avg_visibility, 2)),
        }

    def _calculate_angle_from_landmarks(
        self, landmarks, a_idx, b_idx, c_idx
    ) -> Optional[float]:
        """计算三个关键点形成的角度（以 b 为顶点）"""
        try:
            if (
                a_idx >= len(landmarks)
                or b_idx >= len(landmarks)
                or c_idx >= len(landmarks)
            ):
                return None
            a = landmarks[a_idx]
            b = landmarks[b_idx]
            c = landmarks[c_idx]
            ba = np.array([a.x - b.x, a.y - b.y])
            bc = np.array([c.x - b.x, c.y - b.y])
            cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
            angle = np.arccos(np.clip(cosine, -1.0, 1.0))
            return math.degrees(angle)
        except Exception:
            return None

    def _analyze_biomechanics(self, landmarks, w: int, h: int) -> dict:
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
        # YOLO 左右手腕索引 9, 10
        for wrist_id in [9, 10]:
            if wrist_id in prev and wrist_id in curr:
                dx = curr[wrist_id][0] - prev[wrist_id][0]
                dy = curr[wrist_id][1] - prev[wrist_id][1]
                speed = math.sqrt(dx * dx + dy * dy)
                wrist_speeds.append(speed)
        result["wrist_speed"] = float(
            round(max(wrist_speeds) if wrist_speeds else 0.0, 1)
        )

        # 身体倾斜角（脊柱与垂直线的夹角）
        # YOLO 肩膀 5, 6，髋部 11, 12
        try:
            if (
                5 < len(landmarks)
                and 6 < len(landmarks)
                and 11 < len(landmarks)
                and 12 < len(landmarks)
            ):
                mid_shoulder = np.array(
                    [
                        (landmarks[5].x + landmarks[6].x) / 2,
                        (landmarks[5].y + landmarks[6].y) / 2,
                    ]
                )
                mid_hip = np.array(
                    [
                        (landmarks[11].x + landmarks[12].x) / 2,
                        (landmarks[11].y + landmarks[12].y) / 2,
                    ]
                )
                spine = mid_shoulder - mid_hip
                vertical = np.array([0, -1])
                cos_angle = np.dot(spine, vertical) / (np.linalg.norm(spine) + 1e-8)
                lean_angle = math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))
                result["body_lean"] = float(round(lean_angle, 1))
        except Exception:
            pass

        # 膝盖弯曲度（取双膝平均）
        knee_angles = []
        for name in ["left_knee", "right_knee"]:
            a_id, b_id, c_id = self.JOINT_ANGLES[name]
            angle = self._calculate_angle_from_landmarks(landmarks, a_id, b_id, c_id)
            if angle is not None:
                knee_angles.append(angle)
        result["knee_bend"] = float(
            round(180 - float(np.mean(knee_angles)) if knee_angles else 0.0, 1)
        )

        # 手臂伸展度（肘部角度，越接近 180 越伸展）
        elbow_angles = []
        for name in ["left_elbow", "right_elbow"]:
            a_id, b_id, c_id = self.JOINT_ANGLES[name]
            angle = self._calculate_angle_from_landmarks(landmarks, a_id, b_id, c_id)
            if angle is not None:
                elbow_angles.append(angle)
        result["arm_extension"] = float(
            round(float(np.mean(elbow_angles)) if elbow_angles else 0.0, 1)
        )

        # 对称性评分（0-100，左右对称性）
        try:
            if (
                5 < len(landmarks)
                and 6 < len(landmarks)
                and 11 < len(landmarks)
                and 12 < len(landmarks)
            ):
                left_shoulder_y = landmarks[5].y
                right_shoulder_y = landmarks[6].y
                left_hip_y = landmarks[11].y
                right_hip_y = landmarks[12].y
                shoulder_diff = abs(left_shoulder_y - right_shoulder_y)
                hip_diff = abs(left_hip_y - right_hip_y)
                asymmetry = (shoulder_diff + hip_diff) / 2
                symmetry = max(0, 100 - asymmetry * 500)
                result["symmetry_score"] = float(round(symmetry, 1))
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
