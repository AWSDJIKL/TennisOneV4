"""
Sport Vision — CV 处理流水线
串联姿态分析、动作识别、可视化的核心流水线
"""

import cv2
import base64
import numpy as np
import time
import asyncio
from pathlib import Path
from typing import Optional, AsyncGenerator

from backend.pose_analyzer import PoseAnalyzer
from backend.action_recognizer import ActionRecognizer
from backend.visualizer import Visualizer


class Pipeline:
    """视频分析流水线"""

    def __init__(self):
        self.pose_analyzer = PoseAnalyzer()
        self.action_recognizer = ActionRecognizer()
        self.visualizer = Visualizer()
        self.is_running = False

    async def process_video(
        self,
        video_path: str,
        target_fps: int = 24,
        skip_frames: int = 1,
        buffer_seconds: float = 5.0,
    ) -> AsyncGenerator[dict, None]:
        """
        处理视频并逐帧 yield 分析结果（异步生成器）
        使用生产者-消费者模式设置缓冲区，保证网页前端渲染时的极度平滑流畅。
        """
        import os

        is_gx_camera = video_path == "gx_camera"

        cap = None
        gx_cam = None
        total_frames = -1
        video_fps = target_fps
        frame_width = 1920
        frame_height = 1080

        if is_gx_camera:
            # 相机相关

            import gxipy as gx

            device_manager = gx.DeviceManager()
            dev_num, dev_info_list = device_manager.update_all_device_list()

            if dev_num == 0:
                yield {"error": "No USB camera found"}
                return
            strSN = dev_info_list[0].get("sn")
            gx_cam = device_manager.open_device_by_sn(strSN)
            # try:
            #     # gx_cam.ExposureAuto.set(gx.GxAutoEntry.OFF)
            #     gx_cam.ExposureTime.set(30000.0)
            #     gx_cam.BalanceWhiteAuto.set(gx.GxAutoEntry.CONTINUOUS)
            # except Exception as e:
            #     print(f"Warning: failed to set camera parameters: {e}")
            # gx_cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
            # gx_cam.stream_on()

            # 设置相机参数
            try:
                # ---- 关闭自动曝光 ----
                gx_cam.ExposureAuto.set(gx.GxAutoEntry.OFF)

                # ---- 设置曝光时间 (单位: 微秒) ----
                exposure_time = 4300.0  # 30ms
                gx_cam.ExposureTime.set(exposure_time)
                print(f"Exposure time set to {exposure_time} us")

                # ---- 自动白平衡 ----
                gx_cam.BalanceWhiteAuto.set(gx.GxAutoEntry.CONTINUOUS)
                gx_cam.AcquisitionFrameRateMode.set(gx.GxSwitchEntry.ON)
                gx_cam.AcquisitionFrameRate.set(230.0)
                framerate_get = (
                    gx_cam.CurrentAcquisitionFrameRate.get()
                )  # 获取当前采集的帧率
                print(f"Current camera frame rate: {framerate_get} FPS")

                # （可选）关闭自动增益，如果有需要：
                # cam.GainAuto.set(gx.GxAutoEntry.OFF)
                # cam.Gain.set(10.0)
            except Exception as e:
                print(f"Warning: failed to set camera parameters: {e}")

            # 连续采集模式
            gx_cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
            gx_cam.stream_on()

        else:
            # 输入视频相关
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                yield {"error": f"Cannot open video: {video_path}"}
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

        frame_interval = 1.0 / target_fps

        if is_gx_camera:
            buffer_size = 1  # 实时摄像头不需要缓冲
        else:
            buffer_size = max(1, int(buffer_seconds * target_fps))

        # 预留点队列空间
        frame_queue = asyncio.Queue(maxsize=buffer_size + 10)
        buffer_ready = asyncio.Event()

        async def producer():
            # 增加这句，显式声明需要修改的外层变量
            nonlocal frame_width, frame_height, target_w, target_h
            frame_count = 0
            try:
                while self.is_running:
                    # 显式让出 CPU 控制权，保证消费者有计算资源发送帧
                    await asyncio.sleep(0.001)

                    if is_gx_camera:
                        raw_image = gx_cam.data_stream[0].get_image()
                        if raw_image is None:
                            continue

                        # 1. 从原始图像对象转换为彩色 RGB 对象
                        rgb_image = raw_image.convert("RGB")
                        if rgb_image is None:
                            continue

                        # 2. 注意：从 rgb_image 获取 numpy 彩色矩阵
                        img_rgb = rgb_image.get_numpy_array()

                        # 3. 将 RGB 转成 OpenCV 管道期望的 BGR 格式
                        frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                        ret = True

                        # 动态更新宽高及缩放目标尺寸
                        if frame_width == 1920:
                            frame_width = frame.shape[1]
                            frame_height = frame.shape[0]
                            if frame_width > max_width:
                                scale = max_width / frame_width
                                target_w = max_width
                                target_h = int(frame_height * scale)
                            else:
                                target_w = frame_width
                                target_h = frame_height
                    else:
                        ret, frame = cap.read()
                        if not ret:
                            break

                    frame_count += 1
                    if frame_count % skip_frames != 0:
                        continue

                    if frame.shape[1] != target_w:
                        frame = cv2.resize(frame, (target_w, target_h))
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    pose_result = self.pose_analyzer.process_frame(frame_rgb)
                    action_result = None
                    if pose_result:
                        action_result = self.action_recognizer.update(
                            pose_result["keypoints"], pose_result["joint_angles"]
                        )

                    rendered = self.visualizer.render_frame(
                        frame, pose_result, action_result
                    )
                    _, buffer = cv2.imencode(
                        ".jpg", rendered, [cv2.IMWRITE_JPEG_QUALITY, 80]
                    )
                    frame_base64 = base64.b64encode(buffer).decode("utf-8")

                    progress = frame_count / total_frames if total_frames > 0 else 0

                    result = {
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

                    # 加入队列，如果超过队列最大长度，会自动阻塞当前协程，从而起到反压 (Backpressure) 的作用
                    await frame_queue.put(result)

                    if not buffer_ready.is_set() and frame_queue.qsize() >= buffer_size:
                        buffer_ready.set()

            except Exception as e:
                import traceback

                traceback.print_exc()
                await frame_queue.put({"error": f"Producer core error: {str(e)}"})
            finally:
                await frame_queue.put(None)

        producer_task = asyncio.create_task(producer())

        try:
            # 消费端：等待初始化缓冲装满
            while self.is_running and not buffer_ready.is_set():
                if producer_task.done():
                    break
                await asyncio.sleep(0.1)

            # 消费端：固定平滑的时钟间隔往前端输出
            while self.is_running:
                start_time = time.time()

                item = await frame_queue.get()
                if item is None:
                    break
                if "error" in item:
                    yield item
                    break

                yield item

                # 按照严格时间发送
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        finally:
            self.is_running = False
            producer_task.cancel()
            if is_gx_camera and gx_cam:
                try:
                    gx_cam.stream_off()
                    gx_cam.close()
                except Exception:
                    pass
            elif cap:
                cap.release()

    def _sanitize_pose(self, pose_result: Optional[dict]) -> Optional[dict]:
        """清理姿态数据以便 JSON 序列化"""
        if not pose_result:
            return None
        # 移除大体积的骨骼连接信息（前端已有）
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
