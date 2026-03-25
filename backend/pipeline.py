"""
Sport Vision — CV 处理流水线
串联姿态分析、动作识别、可视化的核心流水线
"""

import cv2
import base64
import time
import subprocess
from pathlib import Path
from collections import deque
from typing import Optional, AsyncGenerator

from backend.pose_analyzer import PoseAnalyzer
from backend.action_recognizer import ActionRecognizer
from backend.visualizer import Visualizer


BASE_DIR = Path(__file__).resolve().parent.parent
CLIPS_DIR = BASE_DIR / "clips"
CLIPS_DIR.mkdir(exist_ok=True)


class Pipeline:
    """视频分析流水线"""

    def __init__(self):
        self.pose_analyzer = PoseAnalyzer()
        self.action_recognizer = ActionRecognizer()
        self.visualizer = Visualizer()
        self.is_running = False

    async def process_video(
        self, video_path: str, target_fps: int = 24, skip_frames: int = 1
    ) -> AsyncGenerator[dict, None]:
        """
        处理视频并逐帧 yield 分析结果（异步生成器）
        普通视频模式下，会在识别到新动作时自动生成切片视频。
        切片完成并转码后，才把 clip 返回给前端。
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            yield {"error": f"Cannot open video: {video_path}"}
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 限制输出尺寸（保持比例，最大宽度 960）
        max_width = 960
        if frame_width > max_width:
            scale = max_width / frame_width
            target_w = max_width
            target_h = int(frame_height * scale)
        else:
            target_w = frame_width
            target_h = frame_height

        self.is_running = True
        self.pose_analyzer.reset()
        self.action_recognizer.reset()
        self.visualizer.reset()

        frame_count = 0
        frame_interval = 1.0 / target_fps

        # ===== demo/upload 自动切片相关 =====
        pre_roll_seconds = 1.0
        post_roll_seconds = 1.0
        pre_roll_frames = max(1, int(video_fps * pre_roll_seconds))
        post_roll_frames = max(1, int(video_fps * post_roll_seconds))

        # 保存最近一段原始帧（用于前置片段）
        frame_buffer = deque(maxlen=pre_roll_frames)

        # 正在写入中的切片
        active_clips = []

        # 已经写完并转码完成、等待发给前端的切片
        ready_clips = []

        def transcode_clip(temp_path: str, final_path: str) -> bool:
            """把 OpenCV 写出的临时 mp4 转成浏览器更兼容的 H.264 mp4"""
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i", temp_path,
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart",
                        final_path,
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            except Exception as e:
                print(f"[clip transcode error] {temp_path} -> {final_path}: {e}")
                return False

        def close_finished_clips():
            finished = []

            for clip in active_clips:
                if clip["remaining_post_frames"] <= 0:
                    try:
                        clip["writer"].release()
                    except Exception:
                        pass

                    info = clip["info"]
                    temp_path = info["temp_path"]
                    final_path = info["saved_path"]

                    ok = transcode_clip(temp_path, final_path)
                    if ok:
                        try:
                            Path(temp_path).unlink(missing_ok=True)
                        except Exception:
                            pass

                        info.pop("temp_path", None)
                        ready_clips.append(info)

                    finished.append(clip)

            for clip in finished:
                active_clips.remove(clip)

        def start_new_clip(action_name: str, current_frame_idx: int):
            timestamp_ms = int(time.time() * 1000)
            base_name = f"{timestamp_ms}_{action_name}_{current_frame_idx}"
            temp_filename = f"{base_name}_raw.mp4"
            final_filename = f"{base_name}.mp4"

            temp_path = CLIPS_DIR / temp_filename
            final_path = CLIPS_DIR / final_filename

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(temp_path), fourcc, video_fps, (target_w, target_h))

            # 先写入前置缓存
            for buffered_frame in frame_buffer:
                writer.write(buffered_frame)

            clip_start_frame = max(1, current_frame_idx - len(frame_buffer) + 1)
            clip_end_frame = current_frame_idx + post_roll_frames

            clip_info = {
                "clip_id": final_filename,
                "video_url": f"/clips/{final_filename}",
                "saved_path": str(final_path),
                "temp_path": str(temp_path),
                "start_frame": clip_start_frame,
                "end_frame": clip_end_frame,
                "start_time": round(clip_start_frame / video_fps, 3),
                "end_time": round(clip_end_frame / video_fps, 3),
                "action": action_name,
            }

            active_clips.append(
                {
                    "writer": writer,
                    "remaining_post_frames": post_roll_frames,
                    "info": clip_info,
                }
            )

        def build_completed_action_result(clip_info: dict, base_action_result: Optional[dict]):
            action_name = clip_info["action"]
            action_info = self.action_recognizer.ACTIONS.get(
                action_name, self.action_recognizer.ACTIONS["ready"]
            )

            action_counts = {}
            action_history = []
            confidence = 0.95

            if base_action_result:
                action_counts = dict(base_action_result.get("action_counts", {}))
                action_history = list(base_action_result.get("action_history", []))
                confidence = base_action_result.get("confidence", 0.95)

            action_history.append(
                {
                    "frame": frame_count,
                    "action": action_name,
                    "name": action_info["name"],
                    "clip_id": clip_info["clip_id"],
                    "video_url": clip_info["video_url"],
                    "start_time": clip_info["start_time"],
                    "end_time": clip_info["end_time"],
                }
            )

            return {
                "action": action_name,
                "action_info": action_info,
                "confidence": confidence,
                "is_new_action": True,
                "action_counts": action_counts,
                "action_history": action_history[-20:],
                "clip": clip_info,
            }

        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % skip_frames != 0:
                    continue

                start_time = time.time()

                # 缩放
                if frame.shape[1] != target_w:
                    frame = cv2.resize(frame, (target_w, target_h))

                # 原始 BGR 帧先入缓存，用于切片
                frame_buffer.append(frame.copy())

                # 正在录制的切片继续写当前帧
                for clip in active_clips:
                    clip["writer"].write(frame)
                    clip["remaining_post_frames"] -= 1

                # 收尾并转码完成的切片进入 ready 队列
                close_finished_clips()

                # RGB 转换（MediaPipe 需要 RGB）
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 1. 姿态分析
                pose_result = self.pose_analyzer.process_frame(frame_rgb)

                # 2. 动作识别
                action_result = self.action_recognizer.update(pose_result)

                # 3. 识别到新动作时，只负责启动切片
                if (
                    action_result
                    and action_result.get("is_new_action")
                    and action_result.get("action") in ("forehand", "backhand")
                ):
                    start_new_clip(action_result["action"], frame_count)

                # 4. 如果有已完成切片，就作为“可播放动作事件”返回给前端
                if ready_clips:
                    completed_clip = ready_clips.pop(0)
                    action_result = build_completed_action_result(completed_clip, action_result)
                elif action_result:
                    action_result["clip"] = None
                    action_result["is_new_action"] = False

                # 5. 可视化渲染
                rendered = self.visualizer.render_frame(frame, pose_result, action_result)

                # 编码为 JPEG base64
                _, buffer = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_base64 = base64.b64encode(buffer).decode("utf-8")

                # 构建输出
                progress = frame_count / total_frames if total_frames > 0 else 0

                yield {
                    "frame_base64": frame_base64,
                    "frame_number": frame_count,
                    "total_frames": total_frames,
                    "fps": round(video_fps, 1),
                    "width": target_w,
                    "height": target_h,
                    "pose": self._sanitize_pose(pose_result),
                    "action": action_result,
                    "progress": round(min(progress, 1.0), 3),
                    "heatmap_data": self.pose_analyzer.get_trajectory(),
                }

                # 控制帧率
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    import asyncio
                    await asyncio.sleep(sleep_time)

            # 视频结束后，把剩余活跃切片全部收尾、转码并补发
            for clip in active_clips:
                try:
                    clip["writer"].release()
                    info = clip["info"]
                    temp_path = info["temp_path"]
                    final_path = info["saved_path"]

                    ok = transcode_clip(temp_path, final_path)
                    if ok:
                        try:
                            Path(temp_path).unlink(missing_ok=True)
                        except Exception:
                            pass
                        info.pop("temp_path", None)
                        ready_clips.append(info)
                except Exception as e:
                    print(f"[clip final transcode error] {e}")

            active_clips.clear()

            while ready_clips:
                completed_clip = ready_clips.pop(0)
                action_result = build_completed_action_result(completed_clip, None)

                blank_frame = frame_buffer[-1] if frame_buffer else None
                if blank_frame is None:
                    break

                rendered = self.visualizer.render_frame(blank_frame.copy(), None, action_result)
                _, buffer = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_base64 = base64.b64encode(buffer).decode("utf-8")

                yield {
                    "frame_base64": frame_base64,
                    "frame_number": frame_count,
                    "total_frames": total_frames,
                    "fps": round(video_fps, 1),
                    "width": target_w,
                    "height": target_h,
                    "pose": None,
                    "action": action_result,
                    "progress": 1.0,
                    "heatmap_data": self.pose_analyzer.get_trajectory(),
                }

        finally:
            cap.release()
            for clip in active_clips:
                try:
                    clip["writer"].release()
                except Exception:
                    pass
            self.is_running = False

    def _sanitize_pose(self, pose_result: Optional[dict]) -> Optional[dict]:
        """清理姿态数据以便 JSON 序列化"""
        if not pose_result:
            return None

        return {
            "keypoints": pose_result["keypoints"],
            "joint_angles": pose_result["joint_angles"],
            "biomechanics": pose_result["biomechanics"],
            "center_of_mass": pose_result["center_of_mass"],
            "confidence": pose_result["confidence"],
        }

    def stop(self):
        """停止处理"""
        self.is_running = False

    def close(self):
        """释放所有资源"""
        self.stop()
        self.pose_analyzer.close()
        if hasattr(self.action_recognizer, "close"):
            self.action_recognizer.close()

    async def process_camera(self, target_fps: int = 20) -> AsyncGenerator[dict, None]:
        """处理实时工业相机数据流"""
        import asyncio
        from backend.cam import CameraHandler
        from backend.pose_analyzer import PoseAnalyzer

        try:
            camera_handler = CameraHandler()
            camera_handler.start_capture()
        except Exception as e:
            yield {"error": f"相机启动失败: {str(e)}"}
            return

        try:
            pose_analyzer_for_gcn = PoseAnalyzer()
        except Exception as e:
            yield {"error": f"PoseAnalyzer初始化失败: {str(e)}"}
            camera_handler.release_camera()
            return

        self.is_running = True
        self.pose_analyzer.reset()
        self.action_recognizer.reset()
        self.visualizer.reset()

        self.action_recognizer.set_camera_mode(camera_handler, pose_analyzer_for_gcn)

        frame_count = 0
        frame_interval = 1.0 / target_fps

        try:
            while self.is_running:
                start_time = time.time()

                latest = camera_handler.latest_frame
                if latest is None:
                    await asyncio.sleep(0.01)
                    continue

                timestamp, frame_rgb = latest

                frame_height, frame_width = frame_rgb.shape[:2]
                max_width = 960
                if frame_width > max_width:
                    scale = max_width / frame_width
                    target_w = max_width
                    target_h = int(frame_height * scale)
                    frame_rgb_resized = cv2.resize(frame_rgb, (target_w, target_h))
                else:
                    target_w, target_h = frame_width, frame_height
                    frame_rgb_resized = frame_rgb.copy()

                frame_bgr = cv2.cvtColor(frame_rgb_resized, cv2.COLOR_RGB2BGR)

                frame_count += 1

                pose_result = self.pose_analyzer.process_frame(frame_rgb_resized)
                action_result = self.action_recognizer.update(pose_result)

                rendered = self.visualizer.render_frame(frame_bgr, pose_result, action_result)

                _, buffer = cv2.imencode(".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_base64 = base64.b64encode(buffer).decode("utf-8")

                yield {
                    "frame_base64": frame_base64,
                    "frame_number": frame_count,
                    "total_frames": 0,
                    "fps": target_fps,
                    "width": target_w,
                    "height": target_h,
                    "pose": self._sanitize_pose(pose_result),
                    "action": action_result,
                    "progress": 0.0,
                    "heatmap_data": self.pose_analyzer.get_trajectory(),
                }

                elapsed = time.time() - start_time
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield {"error": f"处理流异常: {str(e)}"}
        finally:
            self.action_recognizer.set_camera_mode(None, None)
            camera_handler.release_camera()
            self.is_running = False
            pose_analyzer_for_gcn.close()