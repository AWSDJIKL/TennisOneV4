import sys
import platform
import gxipy as gx
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import datetime


if __name__ == "__main__":
    # 检测操作系统
    is_windows = platform.system().lower() == "windows"

    # 创建设备管理器
    device_manager = gx.DeviceManager()

    # 枚举设备
    dev_num, dev_info_list = device_manager.update_all_device_list()
    if dev_num == 0:
        print("No camera found")
        sys.exit(1)

    # 打开设备
    strSN = dev_info_list[0].get("sn")
    cam = device_manager.open_device_by_sn(strSN)
    # framerate_get = cam.CurrentAcquisitionFrameRate.get()  # 获取当前采集的帧率
    # print(f"Current camera frame rate: {framerate_get} FPS")
    # 设置相机参数
    try:
        # ---- 关闭自动曝光 ----
        cam.ExposureAuto.set(gx.GxAutoEntry.OFF)

        # ---- 设置曝光时间 (单位: 微秒) ----
        exposure_time = 4300.0  # 30ms
        cam.ExposureTime.set(exposure_time)
        print(f"Exposure time set to {exposure_time} us")

        # ---- 自动白平衡 ----
        cam.DeviceLinkThroughputLimitMode.set(gx.GxSwitchEntry.OFF)
        cam.BalanceWhiteAuto.set(gx.GxAutoEntry.CONTINUOUS)
        cam.AcquisitionFrameRateMode.set(gx.GxSwitchEntry.ON)
        cam.AcquisitionFrameRate.set(230.0)
        framerate_get = cam.CurrentAcquisitionFrameRate.get()  # 获取当前采集的帧率
        print(f"Current camera frame rate: {framerate_get} FPS")

        # （可选）关闭自动增益，如果有需要：
        # cam.GainAuto.set(gx.GxAutoEntry.OFF)
        cam.Gain.set(24.0)
    except Exception as e:
        print(f"Warning: failed to set camera parameters: {e}")

    # 连续采集模式
    cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
    cam.stream_on()
    # print("Streaming started... Press 'q' to quit")

    while True:
        try:
            # 获取一帧图像
            raw_image = cam.data_stream[0].get_image()
            if raw_image is None:
                print("Warning: failed to get image")
                continue
            rgb_image = raw_image.convert("RGB")  # 从彩色原始图像获取RGB图像
            if rgb_image is None:
                print("Warning: failed to convert to RGB")
                continue

            # rgb_image.image_improvement(color_correction_param, contrast_lut, gamma_lut)  # 实现图像增强

            numpy_image = rgb_image.get_numpy_array()  # 从RGB图像数据创建numpy数组
            if numpy_image is None:
                print("Warning: failed to convert RGB image to numpy array")
                continue

            # img = Image.fromarray(numpy_image, "RGB")  # 展示获取的图像
            # img.show()
            # mtime = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

            # img.save(
            #     r"D:\image\\" + str(i) + str("-") + mtime + ".jpg"
            # )  # 保存图片到本地
            # # 转换为OpenCV格式 (BGR)
            # img = raw_image.get_numpy_array()
            # img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 如果是单通道，转换为3通道

            # 显示图像
            cv2.imwrite("./Camera Stream.jpg", numpy_image)
            break
        except Exception as e:
            print(f"Error during streaming: {e}")
            break
    # 释放资源
    cam.stream_off()
