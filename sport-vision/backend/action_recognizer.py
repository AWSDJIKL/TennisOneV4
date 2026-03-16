"""
Sport Vision — 动作识别模块
基于关键点时序分析的击球动作识别引擎
"""

import math
import numpy as np
from collections import deque
from typing import Optional


class ActionRecognizer:
    """
    规则引擎动作识别器
    通过关键点的时序变化识别羽毛球/网球的典型动作
    """

    # 支持识别的动作类型
    ACTIONS = {
        "serve": {"name": "发球 Serve", "icon": "🎯", "color": "#00f0ff"},
        "smash": {"name": "扣杀 Smash", "icon": "💥", "color": "#ff3366"},
        "forehand": {"name": "正手 Forehand", "icon": "➡️", "color": "#33ff88"},
        "backhand": {"name": "反手 Backhand", "icon": "⬅️", "color": "#ffaa33"},
        "ready": {"name": "准备 Ready", "icon": "🧍", "color": "#888888"},
        "moving": {"name": "移动 Moving", "icon": "🏃", "color": "#ffdd44"},
    }

    def __init__(self, window_size: int = 1, debounce_frames: int = 1):
        """
        Args:
            window_size: 滑动窗口大小（帧数）
            debounce_frames: 动作去抖动间隔（防止同一动作重复触发）
        """
        self.window_size = window_size
        self.debounce_frames = debounce_frames
        # 关键点历史缓冲
        self.keypoint_buffer: deque = deque(maxlen=window_size)
        # 动作历史
        self.action_history: list = []
        self.last_action = "ready"
        self.last_action_frame = -debounce_frames
        self.frame_count = 0
        # 统计
        self.action_counts: dict = {k: 0 for k in self.ACTIONS}

    def update(self, keypoints: list, joint_angles: dict) -> dict:
        """
        输入当前帧的关键点和关节角度，输出识别结果

        Returns:
            {
                "action": str,
                "action_info": {name, icon, color},
                "confidence": float,
                "is_new_action": bool,
                "action_counts": dict,
                "action_history": list (recent 20),
            }
        """
        self.frame_count += 1

        # 构建关键点字典
        raw_kp_map = {}
        for kp in keypoints:
            raw_kp_map[kp["name"]] = kp
            
        # 这里进行骨骼旋转归一化，消除视角和站位朝向的偏差
        norm_kp_map = self._normalize_skeleton(raw_kp_map)
        
        self.keypoint_buffer.append(norm_kp_map)

        if len(self.keypoint_buffer) < 5:
            return self._make_result("ready", 0.5, False)

        # 识别动作
        action, confidence = self._recognize(norm_kp_map, joint_angles)

        # 去抖动
        is_new = False
        if action != self.last_action and action not in ("ready", "moving"):
            if self.frame_count - self.last_action_frame >= self.debounce_frames:
                is_new = True
                self.last_action = action
                self.last_action_frame = self.frame_count
                self.action_counts[action] = self.action_counts.get(action, 0) + 1
                self.action_history.append({
                    "action": action,
                    "frame": self.frame_count,
                    "confidence": confidence,
                })
        elif action in ("ready", "moving"):
            self.last_action = action

        return self._make_result(action, confidence, is_new)

    def _normalize_skeleton(self, kp_map: dict) -> dict:
        """旋转归一化骨骼坐标，消除相机视角/站位引发的坐标错乱问题"""
        left_hip = kp_map.get("left_hip")
        right_hip = kp_map.get("right_hip")
        left_shoulder = kp_map.get("left_shoulder")
        right_shoulder = kp_map.get("right_shoulder")
        
        # 必须有躯干基准点来计算朝向
        if left_hip and right_hip and left_shoulder and right_shoulder:
            # 取身体中心作为原点
            cx = (left_hip["x"] + right_hip["x"] + left_shoulder["x"] + right_shoulder["x"]) / 4
            cy = (left_hip["y"] + right_hip["y"] + left_shoulder["y"] + right_shoulder["y"]) / 4
            
            # 使用两侧髋部的连线来判断朝向角（网球侧身时，肩膀旋转极大，使用髋部加上肩膀平均相对更准）
            # 目标是使得 左->右 连线水平，这样身体“正面对着前方”就统一了
            dx = right_shoulder["x"] - left_shoulder["x"]
            dy = right_shoulder["y"] - left_shoulder["y"]
            
            # 由于可能背对或面对屏幕，dx 可能有正负变化
            # 但不管怎样，我们通过旋转，将其强行拉平 (水平)，把二维骨骼视角变成统一的正面视角
            angle = math.atan2(dy, dx)
            cos_a = math.cos(-angle)
            sin_a = math.sin(-angle)
            
            norm_map = {}
            for name, kp in kp_map.items():
                # 平移到原点后旋转
                tx = kp["x"] - cx
                ty = kp["y"] - cy
                rx = tx * cos_a - ty * sin_a
                ry = tx * sin_a + ty * cos_a
                norm_map[name] = {"x": rx, "y": ry, "name": name}
            return norm_map

        return dict(kp_map)

    def _recognize(self, kp: dict, angles: dict) -> tuple:
        """核心识别逻辑"""

        # 提取关键指标
        right_wrist = kp.get("right_wrist", {})
        left_wrist = kp.get("left_wrist", {})
        right_shoulder = kp.get("right_shoulder", {})
        right_elbow = kp.get("right_elbow", {})
        right_hip = kp.get("right_hip", {})
        nose = kp.get("nose", {})

        if not all([right_wrist, right_shoulder, nose]):
            return "ready", 0.3

        wrist_y = right_wrist.get("y", 0)
        shoulder_y = right_shoulder.get("y", 0)
        hip_y = right_hip.get("y", 0)
        nose_y = nose.get("y", 0)

        wrist_x = right_wrist.get("x", 0)
        shoulder_x = right_shoulder.get("x", 0)

        # 计算手腕速度（帧间差异）
        wrist_speed = self._get_wrist_speed()
        wrist_vertical_speed = self._get_wrist_vertical_speed()

        # 肘部角度
        elbow_angle = angles.get("right_elbow", 90)
        shoulder_angle = angles.get("right_shoulder", 90)

        # === 发球检测 ===
        # 特征：手腕在头顶以上 + 手臂伸展 + 从高到低的轨迹
        if (wrist_y < nose_y and
            elbow_angle > 140 and
            shoulder_angle > 120 and
            wrist_vertical_speed > 3):
            return "serve", 0.85

        # === 扣杀检测 ===
        # 特征：手腕极高位 + 快速向下挥动 + 肘部先弯后伸
        if (wrist_y < shoulder_y - 50 and
            wrist_speed > 15 and
            wrist_vertical_speed > 8):
            return "smash", 0.80

        # === 正手/反手检测 ===
        # 基于归一化后手腕相对身体的横向位置和运动方向
        # 归一化后，身体总是“面向”特定的方向（左右肩水平）
        lateral_speed = self._get_wrist_lateral_speed()
        if wrist_speed > 8:
            # 如果打球手（假设右手）在身体的右侧外扩出去，那就是正手
            if wrist_x > shoulder_x and lateral_speed > 3:
                return "forehand", 0.75
            # 如果打球手跨过了身体中线向左或者处于左侧随挥，就是反手
            elif wrist_x < shoulder_x and lateral_speed < -3:
                return "backhand", 0.70

        # === 移动检测 ===
        body_speed = self._get_body_speed()
        if body_speed > 5:
            return "moving", 0.60

        return "ready", 0.50

    def _get_wrist_speed(self) -> float:
        """计算手腕速度（像素/帧）"""
        if len(self.keypoint_buffer) < 2:
            return 0
        curr = self.keypoint_buffer[-1].get("right_wrist", {})
        prev = self.keypoint_buffer[-2].get("right_wrist", {})
        if not curr or not prev:
            return 0
        dx = curr.get("x", 0) - prev.get("x", 0)
        dy = curr.get("y", 0) - prev.get("y", 0)
        return math.sqrt(dx * dx + dy * dy)

    def _get_wrist_vertical_speed(self) -> float:
        """手腕垂直速度（正值=向下，负值=向上）"""
        if len(self.keypoint_buffer) < 3:
            return 0
        curr = self.keypoint_buffer[-1].get("right_wrist", {})
        prev = self.keypoint_buffer[-3].get("right_wrist", {})
        if not curr or not prev:
            return 0
        return (curr.get("y", 0) - prev.get("y", 0)) / 2

    def _get_wrist_lateral_speed(self) -> float:
        """手腕横向速度（正值=向右，负值=向左）"""
        if len(self.keypoint_buffer) < 3:
            return 0
        curr = self.keypoint_buffer[-1].get("right_wrist", {})
        prev = self.keypoint_buffer[-3].get("right_wrist", {})
        if not curr or not prev:
            return 0
        return (curr.get("x", 0) - prev.get("x", 0)) / 2

    def _get_body_speed(self) -> float:
        """身体整体移动速度（基于髋部）"""
        if len(self.keypoint_buffer) < 2:
            return 0
        curr_lh = self.keypoint_buffer[-1].get("left_hip", {})
        curr_rh = self.keypoint_buffer[-1].get("right_hip", {})
        prev_lh = self.keypoint_buffer[-2].get("left_hip", {})
        prev_rh = self.keypoint_buffer[-2].get("right_hip", {})
        if not all([curr_lh, curr_rh, prev_lh, prev_rh]):
            return 0
        cx = (curr_lh.get("x", 0) + curr_rh.get("x", 0)) / 2
        cy = (curr_lh.get("y", 0) + curr_rh.get("y", 0)) / 2
        px = (prev_lh.get("x", 0) + prev_rh.get("x", 0)) / 2
        py = (prev_lh.get("y", 0) + prev_rh.get("y", 0)) / 2
        return math.sqrt((cx - px) ** 2 + (cy - py) ** 2)

    def _make_result(self, action: str, confidence: float, is_new: bool) -> dict:
        action_info = self.ACTIONS.get(action, self.ACTIONS["ready"])
        return {
            "action": action,
            "action_info": action_info,
            "confidence": float(round(confidence, 2)),
            "is_new_action": is_new,
            "action_counts": dict(self.action_counts),
            "action_history": self.action_history[-20:],
        }

    def reset(self):
        """重置状态"""
        self.keypoint_buffer.clear()
        self.action_history.clear()
        self.last_action = "ready"
        self.last_action_frame = -self.debounce_frames
        self.frame_count = 0
        self.action_counts = {k: 0 for k in self.ACTIONS}
