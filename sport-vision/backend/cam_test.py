# 导入所需库
import cv2
import time
from datetime import datetime

# ====================== 配置区域 ======================
# 摄像头IP地址
IP_ADDRESS = "192.168.101.209"
# RTSP端口，海康威视通常为554
RTSP_PORT = "554"
# 摄像头用户名 (替换为你的实际用户名)
USERNAME = "admin"
# 摄像头密码 (替换为你的实际密码)
PASSWORD = "fds147258"
# 录制的视频保存路径及文件名前缀
OUTPUT_FILE_PREFIX = "hikvision_recording"
# 每次录制的时长（秒），例如每60秒保存为一个文件
RECORD_DURATION = 3
# =====================================================

# 构建RTSP URL (海康威视主码流路径通常为 /h264/ch1/main/av_stream)
# 如果子码流是 /h264/ch1/sub/av_stream，可根据需要替换
rtsp_url = f"rtsp://{USERNAME}:{PASSWORD}@{IP_ADDRESS}:{RTSP_PORT}/h264/ch1/main/av_stream"
print(f"尝试连接: {rtsp_url}")

# 创建VideoCapture对象来连接摄像头
cap = cv2.VideoCapture(rtsp_url)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("错误：无法打开摄像头，请检查网络、IP、用户名、密码和RTSP路径。")
    exit()

# 获取视频的原始属性（帧宽度、高度、帧率）
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
# 如果获取的fps为0或无效，可以手动设置一个常见值，例如25
if fps <= 0:
    fps = 25.0
    print(f"警告：未能获取到有效帧率，手动设置为 {fps} fps")

print(f"视频流信息: 分辨率 {frame_width}x{frame_height}, 帧率 {fps:.2f}")

# 定义视频编码器 (使用mp4v编码，生成MP4文件)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 主循环：用于按时间段分割录制文件
while True:
    # 生成带时间戳的文件名，避免覆盖
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{OUTPUT_FILE_PREFIX}_{timestamp}.mp4"
    print(f"开始录制新文件: {output_filename}")

    # 创建VideoWriter对象，准备写入视频
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print("错误：无法创建视频文件，请检查磁盘空间或写入权限。")
        break

    # 记录本次分段录制的开始时间
    start_time = time.time()

    # 内层循环：持续读取帧，直到达到设定的录制时长
    while (time.time() - start_time) < RECORD_DURATION:
        ret, frame = cap.read()  # 读取一帧

        # 检查是否成功读取帧
        if not ret:
            print("警告：无法读取视频帧，可能流已中断。尝试重新连接...")
            # 尝试重新连接
            cap.release()
            time.sleep(2)
            cap.open(rtsp_url)
            if not cap.isOpened():
                print("错误：重新连接摄像头失败。")
                break
            # 重置开始时间，避免立即结束
            start_time = time.time()
            continue

        # 将当前帧写入输出文件
        out.write(frame)

        # 可选：在窗口中显示实时画面（按'q'键可退出）
        # cv2.imshow('Recording...', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # 分段录制结束，释放当前的VideoWriter
    out.release()
    print(f"文件 {output_filename} 录制完成。")

    # 如果在内层循环中按了'q'，则退出主循环
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# 所有录制结束后，释放摄像头资源并关闭窗口
cap.release()
cv2.destroyAllWindows()
print("录制程序已退出。")