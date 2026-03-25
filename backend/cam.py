import gxipy as gx
import cv2
import threading
import time
import os
from datetime import datetime
from collections import deque

FRAME_RATE = 227

# 确保视频保存目录存在
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_DIR = os.path.join(PROJECT_ROOT, "video")
os.makedirs(VIDEO_DIR, exist_ok=True)


class CameraHandler:
    def __init__(self):
        self.camera = None
        # 你的需求原是 60，但会导致内存溢出(OOM)，现在改成 5 (5秒也完全足够截取最新1秒或前后的数据)
        self.history_buffer = deque(maxlen=int(FRAME_RATE * 5))
        self.is_capturing = False
        self.capture_thread = None
        self.latest_frame = None  # 仅保存元组 (timestamp, frame) 给抽样用
        self.initialize_camera()

    def initialize_camera(self):
        self.device_manager = gx.DeviceManager()
        dev_num, dev_info_list = self.device_manager.update_all_device_list()
        if dev_num == 0:
            print("没有发现已连接的相机 (No camera found)")
            return

        strSN = dev_info_list[0].get("sn")
        self.camera = self.device_manager.open_device_by_sn(strSN)

        try:
            self.camera.ExposureAuto.set(gx.GxAutoEntry.OFF)
            self.camera.ExposureTime.set(3000.0)
            self.camera.DeviceLinkThroughputLimitMode.set(gx.GxSwitchEntry.OFF)
            self.camera.BalanceWhiteAuto.set(gx.GxAutoEntry.CONTINUOUS)
            self.camera.AcquisitionFrameRateMode.set(gx.GxSwitchEntry.ON)
            self.camera.AcquisitionFrameRate.set(FRAME_RATE)
            self.camera.Gain.set(20.0)
        except Exception as e:
            print(f"Warning: failed to set camera parameters: {e}")

        self.camera.TriggerMode.set(gx.GxSwitchEntry.OFF)
        self.camera.stream_on()

    def start_capture(self):
        if self.camera is None:
            raise RuntimeError("相机未准备就绪")

        self.is_capturing = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _capture_loop(self):
        while self.is_capturing:
            try:
                raw_image = self.camera.data_stream[0].get_image()
                if raw_image is not None:
                    rgb_image = raw_image.convert("RGB")
                    if rgb_image is not None:
                        numpy_image = rgb_image.get_numpy_array()
                        if numpy_image is not None:
                            timestamp = time.time()
                            self.history_buffer.append((timestamp, numpy_image))
                            self.latest_frame = (timestamp, numpy_image)
                else:
                    time.sleep(0.001)
            except Exception as e:
                # 出现异常时短暂停顿，防止死循环占满CPU或刷屏
                time.sleep(0.01)

    def get_latest_second_samples(self, target_frames=30):
        # 截取最新1秒（约 FRAME_RATE 帧）进行平均抽样
        snapshot = list(self.history_buffer)
        if not snapshot:
            return []

        now = time.time()
        start_target = now - 1.0

        # 寻找最近一秒内的所有帧
        valid_frames = [f for f in snapshot if f[0] >= start_target]

        # 如果不到1秒的帧，则直接使用当前所有有效帧
        if not valid_frames:
            valid_frames = snapshot

        # 平均采样30帧
        total = len(valid_frames)
        if total == 0:
            return []

        indices = [int(i * total / target_frames) for i in range(target_frames)]
        sampled = [valid_frames[i] for i in indices]

        return sampled

    def save_action_video(
        self, action_type: str, start_ts: float, end_ts: float
    ) -> str:
        """
        根据动作时间窗口保存视频（同步 + 转码，确保浏览器可播放）
        """
    
        frames_snapshot = list(self.history_buffer)
        if not frames_snapshot:
            return ""
    
        valid_frames = [
            (ts, frm) for ts, frm in frames_snapshot if start_ts <= ts <= end_ts
        ]
    
        if not valid_frames:
            print("No frames found for the given time window.")
            return ""
    
        first_ts = valid_frames[0][0]
        dt_str = datetime.fromtimestamp(float(first_ts)).strftime("%Y_%m_%d_%H_%M_%S")
    
        action_ch = "正手" if action_type == "forehand" else "反手"
    
        base_name = f"{dt_str}_{action_ch}"
        temp_filename = f"{base_name}_raw.mp4"
        final_filename = f"{base_name}.mp4"
    
        temp_path = os.path.join(VIDEO_DIR, temp_filename)
        final_path = os.path.join(VIDEO_DIR, final_filename)
    
        frames = [f[1] for f in valid_frames]
    
        # ===== 1️⃣ 同步写临时视频 =====
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            temp_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            FRAME_RATE,
            (w, h),
        )
    
        for frm in frames:
            writer.write(cv2.cvtColor(frm, cv2.COLOR_RGB2BGR))
    
        writer.release()
    
        print(f"[camera] raw clip saved: {temp_path}")
    
        # ===== 2️⃣ ffmpeg 转码（关键）=====
        try:
            import subprocess
    
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
    
            # 删除临时文件
            try:
                os.remove(temp_path)
            except Exception:
                pass
    
            print(f"[camera] final clip saved: {final_path}")
    
        except Exception as e:
            print(f"[camera] ffmpeg error: {e}")
            return ""
    
        # ===== 3️⃣ 只在完全完成后返回 =====
        return f"video/{final_filename}"

    def release_camera(self):
        self.is_capturing = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if hasattr(self, "camera") and self.camera is not None:
            try:
                self.camera.stream_off()
                self.camera.close()
            except Exception:
                pass
            self.camera = None
        if hasattr(self, "device_manager") and self.device_manager is not None:
            # Depending on gxipy version, some require close()
            self.device_manager = None
