"""
Sport Vision — 动作识别模块
基于 GCN 时序分析的击球动作识别引擎 (非阻塞缓存池滑动窗口)
"""

import sys
import os
import threading
import queue
import time
from collections import deque
from typing import Optional

# 将项目根目录注入 sys.path 以便导入根目录下的 gcn_test 和 GCN 模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import gcn_test


class ActionRecognizer:
    """
    GCN 的解耦动作识别器
    维持一个缓存池按顺序存储所有视频帧，后台线程以 30 帧窗口逐帧滑动识别
    """

    ACTIONS = {
        "forehand": {"name": "正手 Forehand", "icon": "➡️", "color": "#33ff88"},
        "backhand": {"name": "反手 Backhand", "icon": "⬅️", "color": "#ffaa33"},
        "ready": {"name": "其他 Other / 准备", "icon": "🧍", "color": "#888888"},
        "moving": {"name": "移动 Moving", "icon": "🏃", "color": "#ffdd44"},
    }

    GCN_TO_ACTION = {0: "forehand", 1: "backhand", 2: "ready"}

    def __init__(self, window_size: int = 30, skip_frames_cd: int = 60):
        """
        Args:
            window_size: 必须为 30 左右（针对本 GCN 预处理逻辑），滑动窗口大小（帧数）
            skip_frames_cd: 识别到动作后的冷却帧数
        """
        self.window_size = window_size
        self.skip_frames_cd = skip_frames_cd

        # 相机处理模式专用
        self.camera_handler = None
        self.pose_analyzer = None

        # 帧传递队列（主线程 -> GCN 工作线程）
        self.frame_queue = queue.Queue(maxsize=1000)

        # 滑动窗口缓存池
        self.keypoint_buffer = deque(maxlen=window_size)

        # 统计数据与状态保存（用于供前端实时快速查询）
        self.action_history = []
        self.action_counts = {k: 0 for k in self.ACTIONS}

        self.current_action = "ready"
        self.is_new_action_flag = False
        self.frame_count = 0
        self.skip_counter = 0

        # 最近一次新动作对应的视频切片信息
        self.latest_clip = None

        # 初始化 GCN 模型并在后台长期运行监听
        try:
            gcn_test.load_models()
        except Exception as e:
            print(f"Warning: Failed to load GCN models: {e}")

        self.running = True
        self.worker_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.worker_thread.start()

    def set_camera_mode(self, camera_handler, pose_analyzer):
        """进入相机实时采样模式：忽略 frame_queue，主动从相机缓存中截取帧去分析"""
        self.camera_handler = camera_handler
        self.pose_analyzer = pose_analyzer

    def _normalize_video_url(self, saved_path: str) -> str:
        """
        把 save_action_video 返回的路径规范成前端可访问 URL
        例如:
            video/123_forehand.mp4 -> ./video/123_forehand.mp4
        """
        if not saved_path:
            return ""

        normalized = str(saved_path).replace("\\", "/").strip()

        if normalized.startswith("./video/"):
            return normalized

        if normalized.startswith("video/"):
            return "./" + normalized

        filename = os.path.basename(normalized)
        return f"./video/{filename}"

    def _process_loop(self):
        """后台线程：不断从缓存池中读取帧数据并执行 GCN 模型推断"""
        while self.running:
            # 相机模式单独处理
            if (
                getattr(self, "camera_handler", None) is not None
                and getattr(self, "pose_analyzer", None) is not None
            ):
                self._process_camera_mode()
                time.sleep(0.01)
                continue

            try:
                frame_kps_xy = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # 冷却期：清空缓存并等待
            if self.skip_counter > 0:
                self.skip_counter -= 1
                self.keypoint_buffer.clear()
                if self.skip_counter == 0:
                    self.current_action = "ready"
                continue

            # 将读取到的帧压入滑动窗口
            self.keypoint_buffer.append(frame_kps_xy)

            # 当收集满一个完整分析窗口
            if len(self.keypoint_buffer) == self.window_size:
                input_data = list(self.keypoint_buffer)

                try:
                    cls_idx = gcn_test.get_cls(input_data)
                    action = self.GCN_TO_ACTION.get(cls_idx, "ready")
                except Exception as e:
                    print(f"GCN Inference error: {e}")
                    action = "ready"

                # 实现逐帧滑动
                self.keypoint_buffer.popleft()

                # 识别成有效动作
                if action in ["forehand", "backhand"]:
                    self.skip_counter = self.skip_frames_cd
                    self.current_action = action
                    self.action_counts[action] += 1

                    self.action_history.append(
                        {
                            "frame": self.frame_count,
                            "action": action,
                            "name": self.ACTIONS.get(action, {}).get("name", action),
                        }
                    )

                    # 普通视频模式当前不生成本地切片文件
                    self.latest_clip = None

                    # 通知主线程：出现了一个新动作
                    self.is_new_action_flag = True
                else:
                    self.current_action = action

    def _process_camera_mode(self):
        """从相机 handler 获取最近一秒等距 30 帧进行连续预测"""
        if self.skip_counter > 0:
            self.skip_counter -= 1
            if self.skip_counter == 0:
                self.current_action = "ready"
            time.sleep(0.05)
            self.keypoint_buffer.clear()
            return

        sampled = self.camera_handler.get_latest_second_samples(self.window_size)
        if len(sampled) < self.window_size:
            time.sleep(0.05)
            return

        input_data = []
        for ts, frame in sampled:
            pose_result = self.pose_analyzer.process_frame(frame)
            if (
                pose_result
                and "raw_kps" in pose_result
                and pose_result["raw_kps"] is not None
            ):
                raw_kps = pose_result["raw_kps"]
                input_data.append([[kp[0], kp[1]] for kp in raw_kps])
            else:
                input_data.append([[0.0, 0.0] for _ in range(17)])

        try:
            cls_idx = gcn_test.get_cls(input_data)
            action = self.GCN_TO_ACTION.get(cls_idx, "ready")
        except Exception as e:
            print(f"GCN Inference error: {e}")
            action = "ready"

        if action in ["forehand", "backhand"]:
            self.skip_counter = self.skip_frames_cd
            self.current_action = action
            self.action_counts[action] += 1
            self.frame_count += 1

            # 从这一秒采样窗口向前多带 1 秒，做一个更完整的动作切片
            start_ts = float(sampled[0][0]) - 0.7
            end_ts = float(sampled[-1][0])

            saved_path = ""
            try:
                saved_path = self.camera_handler.save_action_video(
                    action, start_ts, end_ts
                )
            except Exception as e:
                print(f"save_action_video error: {e}")
                saved_path = ""

            video_url = self._normalize_video_url(saved_path)
            clip_id = os.path.basename(saved_path) if saved_path else ""

            # 关键：每次新动作都绑定它自己唯一的切片信息
            self.latest_clip = {
                "clip_id": clip_id,
                "video_url": video_url,
                "saved_path": saved_path,
                "start_ts": round(start_ts, 3),
                "end_ts": round(end_ts, 3),
            }

            self.action_history.append(
                {
                    "frame": self.frame_count,
                    "action": action,
                    "name": self.ACTIONS.get(action, {}).get("name", action),
                    "video_url": video_url,
                    "clip_id": clip_id,
                    "start_ts": round(start_ts, 3),
                    "end_ts": round(end_ts, 3),
                }
            )

            print(f"[clip] action={action}, file={saved_path}, url={video_url}")

            self.is_new_action_flag = True
        else:
            self.current_action = action

    def update(self, pose_result: Optional[dict]) -> dict:
        """
        主渲染管线调用的超快非阻塞接口。
        拿到姿态直接推到后台工作池，立刻返回最后已知状态即可，无需阻塞等待。
        """
        if getattr(self, "camera_handler", None) is not None:
            # 相机模式下，后台主动拉取帧进行分析
            is_new = self.is_new_action_flag
            if self.is_new_action_flag:
                self.is_new_action_flag = False
            return self._make_result(self.current_action, 0.95, is_new)

        self.frame_count += 1

        # 将结构转化为 [17, 2] 给 GCN
        if (
            pose_result
            and "raw_kps" in pose_result
            and pose_result["raw_kps"] is not None
        ):
            raw_kps = pose_result["raw_kps"]
            frame_kps_xy = [[kp[0], kp[1]] for kp in raw_kps]
        else:
            frame_kps_xy = [[0.0, 0.0] for _ in range(17)]

        # 压入后台队列（不阻塞）
        try:
            self.frame_queue.put_nowait(frame_kps_xy)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            self.frame_queue.put_nowait(frame_kps_xy)

        is_new = self.is_new_action_flag
        if self.is_new_action_flag:
            self.is_new_action_flag = False

        return self._make_result(self.current_action, 0.95, is_new)

    def _make_result(self, action: str, confidence: float, is_new: bool) -> dict:
        return {
            "action": action,
            "action_info": self.ACTIONS.get(action, self.ACTIONS["ready"]),
            "confidence": confidence,
            "is_new_action": is_new,
            "action_counts": self.action_counts,
            "action_history": self.action_history[-20:],
            # 只在“新动作”这一帧返回 clip
            # 前端创建 timeline-item 时立刻绑定，不靠动作名猜测
            "clip": self.latest_clip if is_new else None,
        }

    def reset(self):
        """重启应用时的缓冲清洗"""
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        self.keypoint_buffer.clear()
        self.action_history.clear()
        self.frame_count = 0
        self.skip_counter = 0
        self.current_action = "ready"
        self.is_new_action_flag = False
        self.latest_clip = None

        for k in self.action_counts:
            self.action_counts[k] = 0

    def close(self):
        """安全停止并退出子线程"""
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
