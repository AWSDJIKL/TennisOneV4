import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
import yaml
import os
import argparse
import numpy as np
from tqdm import tqdm
from tqdm.contrib import tzip
# from ultralytics import YOLO
from skopt import gp_minimize
import torch
import random
from torch.utils.data import DataLoader
from torchlight import DictAction
from test import predict_location, get_ensemble_weight, generate_inpaint_mask
from dataset import Shuttlecock_Trajectory_Dataset, Video_IterableDataset
from utils.general import *
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib as mpl
from matplotlib.collections import PatchCollection
from matplotlib.font_manager import FontProperties
import requests
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from PIL import Image
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
import markdown
from bs4 import BeautifulSoup
import json
import re
import GCN.dataset.tools as tools
from matplotlib.pyplot import bone
# from patsy import desc

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
# params["model_folder"] = "../../models/"
params["model_folder"] = "/home/awsdjikl/TrackNetV3/models"
params["maximize_positives"] = "True"
# opWrapper = op.WrapperPython()
# opWrapper.configure(params)
# opWrapper.start()

from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)
device = "cuda" if torch.cuda.is_available() else "cpu"
from PIL import Image
from GCN.main import Processor, get_parser
from GCN import graph

# CATEGORIES = ["forehand_backswing", "backhand_backswing", "backhand_power_stroke", "forehand_power_stroke", "forehand_follow_through", "backhand_follow_through", "others"]
# CATEGORIES = ["正手引拍", "反手引拍", "反手發力", "正手發力", "正手揮隨", "反手揮隨", "其他"]
CATEGORIES = ["正手", "反手", "其他"]
WINDOW_SIZE = 30
MIN_FRAME_PERCENTAGE = 0.4

STANDARD_VALUES = [0.7, 0.75, 0.65, 0.8, 0.5, 0.6, 0.7]  # 标准值，用于六维图绘制

BALL_TRACK_NUM = 1  # 球的轨迹点数
coco_pairs = [(1, 6), (2, 1), (3, 1), (4, 2), (5, 3), (6, 7), (7, 1), (8, 6), (9, 7), (10, 8), (11, 9),
                (12, 6), (13, 7), (14, 12), (15, 13), (16, 14), (17, 15)]
person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", device_map=device)

image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-plus-base")
model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-plus-base", device_map=device)


def calculate_angle(a, b, c):
    # 输入：三个点的坐标 (x, y)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1, 1))  # 避免数值溢出
    return np.degrees(angle)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def predict(indices, y_pred=None, c_pred=None, img_scaler=(1, 1)):
    """ Predict coordinates from heatmap or inpainted coordinates. 

        Args:
            indices (torch.Tensor): indices of input sequence with shape (N, L, 2)
            y_pred (torch.Tensor, optional): predicted heatmap sequence with shape (N, L, H, W)
            c_pred (torch.Tensor, optional): predicted inpainted coordinates sequence with shape (N, L, 2)
            img_scaler (Tuple): image scaler (w_scaler, h_scaler)

        Returns:
            pred_dict (Dict): dictionary of predicted coordinates
                Format: {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}
    """

    pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}

    batch_size, seq_len = indices.shape[0], indices.shape[1]
    # print(f'Batch size: {batch_size}, Sequence length: {seq_len}')
    indices = indices.detach().cpu().numpy()if torch.is_tensor(indices) else indices.numpy()
    vis = None
    # Transform input for heatmap prediction
    if y_pred is not None:
        y_pred = y_pred > 0.5
        vis = y_pred.flatten(start_dim=2).any(dim=2)
        # vis = 1 if torch.any(y_pred).item() else 0  # Visibility is 1 if any pixel in heatmap is above threshold, else 0
        # print("-"* 20)
        # print(vis)
        # print(vis) 
        # print(y_pred)
        y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_pred = to_img_format(y_pred)  # (N, L, H, W)
    # Transform input for coordinate prediction
    if c_pred is not None:
        c_pred = c_pred.detach().cpu().numpy() if torch.is_tensor(c_pred) else c_pred

    prev_f_i = -1
    for n in range(batch_size):
        for f in range(seq_len):
            f_i = indices[n][f][1]
            if f_i != prev_f_i:
                if c_pred is not None:
                    # Predict from coordinate
                    c_p = c_pred[n][f]
                    if vis is not None and vis[n][f]:
                        cx_pred, cy_pred = int(c_p[0] * WIDTH * img_scaler[0]), int(c_p[1] * HEIGHT * img_scaler[1])
                    else:
                        # print(f'Frame {f_i} has no visible object, skipping...')
                        cx_pred, cy_pred = 0, 0
                    # cx_pred, cy_pred = int(c_p[0] * WIDTH * img_scaler[0]), int(c_p[1] * HEIGHT * img_scaler[1]) 
                elif y_pred is not None:
                    # Predict from heatmap
                    y_p = y_pred[n][f]
                    bbox_pred = predict_location(to_img(y_p))
                    if vis[n][f]:
                        cx_pred, cy_pred = int(bbox_pred[0] + bbox_pred[2] / 2), int(bbox_pred[1] + bbox_pred[3] / 2)
                        cx_pred, cy_pred = int(cx_pred * img_scaler[0]), int(cy_pred * img_scaler[1])
                    else:
                        # print(f'Frame {f_i} has no visible object, skipping...')
                        cx_pred, cy_pred = 0, 0
                else:
                    raise ValueError('Invalid input')
                vis_pred = 0 if cx_pred == 0 and cy_pred == 0 else 1
                # print(f'Frame {f_i}, X: {cx_pred}, Y: {cy_pred}, Visibility: {vis_pred}')
                pred_dict['Frame'].append(int(f_i))
                pred_dict['X'].append(cx_pred)
                pred_dict['Y'].append(cy_pred)
                pred_dict['Visibility'].append(vis_pred)
                prev_f_i = f_i
            else:
                break
    
    return pred_dict    


def get_shot_frame(ball_data):
    # 取x轴坐标，计算delta_x
    ball_data = ball_data[["Frame", 'X']]
    # 初始化变量
    segments = []
    current_segment = []

    # 按0分段
    for index, row in ball_data.iterrows():
        if row['X'] == 0:
            if current_segment:
                segments.append(current_segment)
                current_segment = []
        else:
            current_segment.append(row)

    if current_segment:
        segments.append(current_segment)
    # 处理每个分段
    all_strike_positions = []
    for segment in segments:
        if len(segment) > 5:
            segment_df = pd.DataFrame(segment)
            strike_positions = find_strike_positions(segment_df)
            all_strike_positions.extend(strike_positions)
    print("球被击打的位置:", all_strike_positions)

    return all_strike_positions


def create_hexagon_radar_chart(data, labels, title, info_text, output_path):
    """
    创建半透明六维图并保存为PNG
    
    参数:
    - data: 六个维度的数据列表，范围0-1
    - labels: 六个维度的标签列表
    - title: 图表标题
    - info_text: 上方显示的信息文本
    - output_path: 输出图片路径
    """
    # 设置中文字体支持
    font_path = '/usr/share/fonts/SimHei.ttf'
    font = FontProperties(fname=font_path)
    # plt.rcParams['font.family']= "SimHei"  # 指定默认字体
    # plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
    
    # 创建图形和子图
    fig, (text_ax, radar_ax) = plt.subplots(2, 1, figsize=(6, 6),
                                           gridspec_kw={'height_ratios': [1, 3]})
    
    # 设置图形背景为透明
    fig.patch.set_alpha(0.9)
    fig.patch.set_facecolor((0, 0, 0, 0.9))
    
    # 1. 在上方子图显示信息文本
    text_ax.axis('off')
    text_ax.set_facecolor((0, 0, 0, 1))  # 半透明黑色背景
    text_ax.text(0.5, 0.8, info_text,
                ha='center', va='center',
                fontsize=20, color='white',
                transform=text_ax.transAxes,
                 fontproperties=font, fontweight='bold')
    
    # 2. 在下方子图绘制六维图
    radar_ax.set_facecolor((0, 0, 0, 0.7))  # 半透明黑色背景
    
    # 计算六边形的顶点坐标
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False).tolist()
    angles += angles[:1]  # 闭合多边形
    
    # 数据也要闭合
    data = data.tolist()
    data += data[:1]
    labels += labels[:1]
    # print(type(data))
    # print(data)
    # 绘制六边形网格
    for i in range(1, 4):
        radius = i * 0.25
        hex_angles = angles
        hex_x = [radius * np.cos(angle) for angle in hex_angles]
        hex_y = [radius * np.sin(angle) for angle in hex_angles]
        radar_ax.plot(hex_x, hex_y, 'w-', alpha=0.3, linewidth=1)
    
    # 绘制轴线
    for angle in angles[:-1]:  # 不绘制最后一个重复的
        radar_ax.plot([0, 0.95 * np.cos(angle)], [0, 0.95 * np.sin(angle)],
                     'w-', alpha=0.5, linewidth=1)
        
    # 设置标准值
    standard_value = [0.7, 0.75, 0.65, 0.8, 0.5, 0.6]
    standard_value += standard_value[:1]
    for i in range(len(labels)):
        labels[i] = labels[i] + f"\n({data[i]:.2f}/{standard_value[i]:.2f})"
    # 绘制数据区域
    data_x = [d * np.cos(angle) for d, angle in zip(standard_value, angles)]
    data_y = [d * np.sin(angle) for d, angle in zip(standard_value, angles)]
    
    # 填充数据区域
    polygon = Polygon(list(zip(data_x, data_y)), closed=True)
    patches = [polygon]
    p = PatchCollection(patches, alpha=0.5, color='red')
    radar_ax.add_collection(p)
    print(len(data_x), len(data_y))
    # 绘制数据线
    radar_ax.plot(data_x, data_y, 'o-', color='red', linewidth=2, markersize=6)
    
    # 绘制数据区域
    data_x = [d * np.cos(angle) for d, angle in zip(data, angles)]
    data_y = [d * np.sin(angle) for d, angle in zip(data, angles)]
    
    # 填充数据区域
    polygon = Polygon(list(zip(data_x, data_y)), closed=True)
    patches = [polygon]
    p = PatchCollection(patches, alpha=0.35, color='cyan')
    radar_ax.add_collection(p)
    # print(len(data_x), len(data_y))
    # 绘制数据线
    radar_ax.plot(data_x, data_y, 'o-', color='cyan', linewidth=2, markersize=6)
    
    # 添加维度标签
    for i, (label, angle) in enumerate(zip(labels[:-1], angles[:-1])):
        x = 1.3 * np.cos(angle)
        y = 1.3 * np.sin(angle)
        radar_ax.text(x, y, label, ha='center', va='center',
                     fontsize=20, color='white',
                 fontproperties=font, fontweight='bold')
    
    # 设置雷达图属性
    radar_ax.set_xlim(-1.2, 1.2)
    radar_ax.set_ylim(-1.2, 1.2)
    radar_ax.set_aspect('equal')
    radar_ax.axis('off')

    chart_move_up = -0.1
    # 添加标题
    # radar_ax.set_title(title, color='white', fontsize=20, pad=0,
    #              fontproperties=font, fontweight='bold')
    pos = radar_ax.get_position()
    new_pos = [pos.x0, pos.y0 + chart_move_up, pos.width, pos.height]
    radar_ax.set_position(new_pos)
    # 调整布局
    plt.tight_layout()
    
    # 保存为透明PNG
    plt.savefig(output_path, transparent=False, dpi=100,
                bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"六维图已保存: {output_path}")


# 函数用于计算梯度变化的位置
def find_strike_positions(segment_df):
    segment_df['Gradient'] = segment_df['X'].diff()
    start_index = segment_df.index[0]
    strike_positions = []
    for i in range(start_index + 1, start_index + len(segment_df)):
        if (segment_df.loc[i, 'Gradient'] > 0 and segment_df.loc[i - 1, 'Gradient'] < 0) or \
           (segment_df.loc[i, 'Gradient'] < 0 and segment_df.loc[i - 1, 'Gradient'] > 0):
            strike_positions.append(segment_df.loc[i, 'Frame'])
    return strike_positions


def plot_ball_frame(ball_frame):
    x = ball_frame['X'].values
    plt.figure(figsize=(10, 5))
    plt.plot(x, marker='o', linestyle='-', color='b', label='X Coordinate')
    plt.savefig('/home/awsdjikl/TrackNetV3/prediction/game3_ball_all.png')


def slice_videos(video_path, all_strike_positions, lenth=30):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频总帧数: {frame_count}, 帧率: {fps}, 分辨率: {width}x{height}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = Path(video_path)
    video_name = video_path.stem
    output_dir = video_path.parent / Path(video_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_list = []
    current_frame = -1
    for strike_position in all_strike_positions:
        start_frame = max(0, strike_position - int(lenth / 2))
        end_frame = min(frame_count, strike_position + int(lenth / 2) - 1)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        output_path = output_dir / Path(f"{video_name}_{start_frame}_{end_frame}_{strike_position}.mp4")
        video_list.append(output_path)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame - 1
        while True:
            ret, frame = cap.read()
            current_frame += 1
            # print(current_frame)
            if not ret:
                break
            if current_frame >= start_frame and current_frame <= end_frame:
                # print(current_frame)
                out.write(frame)
            if current_frame > end_frame:
                break
        out.release()
    cap.release()
    return video_list


# 使用 markdown -> HTML -> 转换为 ReportLab Flowables（Paragraph / 带 bullet 的 Paragraph）
def md_to_flowables(md_text, style):
    html = markdown.markdown(md_text, extensions=['extra', 'sane_lists'])
    soup = BeautifulSoup(html, "html.parser")
    flowables = []
    for node in soup.contents:
        if getattr(node, "name", None) == "p" or node.name is None:
            # Paragraph：把 <strong>/<em> 转为 <b>/<i>（ReportLab 支持）
            inner = str(node)
            inner = inner.replace("<strong>", "<b>").replace("</strong>", "</b>")
            inner = inner.replace("<em>", "<i>").replace("</em>", "</i>")
            # 去掉外层 <p> 标签，如果存在
            inner = inner.replace("<p>", "").replace("</p>", "")
            flowables.append(Paragraph(inner, style))
        elif node.name in ("ul", "ol"):
            # 列表：每个 li 作为带 bullet 的 Paragraph
            for li in node.find_all("li", recursive=False):
                li_html = li.decode_contents()
                li_html = li_html.replace("<strong>", "<b>").replace("</strong>", "</b>")
                li_html = li_html.replace("<em>", "<i>").replace("</em>", "</i>")
                # ReportLab Paragraph 支持 bulletText 参数
                flowables.append(Paragraph(li_html, style, bulletText='•' if node.name == "ul" else None))
        else:
            # 兜底：把节点转为文本 Paragraph
            txt = node.get_text() if hasattr(node, "get_text") else str(node)
            flowables.append(Paragraph(txt, style))
    return flowables


def generate_report(video_img, ball_list, video_path, score_img, pdf_save_path):
    # url = "http://0.0.0.0:8000/score_tennis"
    # with open(video_path, "rb") as f:
    #     files = {"file": (str(video_path.absolute()), f, "video/mp4")}
    #     data = {"max_slice_nums": "2"}
    #     resp = requests.post(url, files=files, data=data, timeout=60)

    #     # print(resp.text)
    #     # report_text = resp.text["answer"]
    #     def extract_answer(resp):
    #         # 首选直接解析 JSON 响应
    #         try:
    #             j = resp.json()
    #         except Exception:
    #             # 如果 resp.text 是 JSON 字符串，尝试 json.loads
    #             try:
    #                 j = json.loads(resp.text)
    #             except Exception:
    #                 # 兜底：用正则从文本中提取 "answer":"..."
    #                 m = re.search(r'"answer"\s*:\s*"(?P<a>.*?)"', resp.text, re.S)
    #                 if m:
    #                     return m.group('a')
    #                 m = re.search(r"'answer'\s*:\s*'(?P<a>.*?)'", resp.text, re.S)
    #                 if m:
    #                     return m.group('a')
    #                 return resp.text  # 无法解析则返回原始文本

    #         # j 现在通常是 dict 或字符串
    #         if isinstance(j, dict):
    #             # 常见位置：j['answer'] 或 j.get('data',{})['answer']
    #             return j.get('answer') or (j.get('data') and j['data'].get('answer')) or json.dumps(j, ensure_ascii=False)
    #         if isinstance(j, str):
    #             try:
    #                 jj = json.loads(j)
    #                 return jj.get('answer', j)
    #             except Exception:
    #                 return j

    #     report_text = extract_answer(resp)
    #     print(report_text)
    
    report_text = '''
1. 技术动作分析
击球准备：身体侧向展开充分，但非持拍手臂后摆幅度不足，影响躯干扭矩蓄力效果
发力链条：髋关节先行转动不足，主要依赖手臂发力，导致核心力量传递效率降低约40%
击球瞬间：拍面控制稳定，但手腕过早展开，削弱了向前推送的持续加速能力
2. 动力生成问题
引拍轨迹过短，缺少足够的势能积累阶段
重心转移不完整，后腿保留体重过多，前腿承重不足70%
随挥动作在肩线位置提前截停，未完成自然收拍
3. 协调性与时机
判读来球后启动延迟0.1秒，导致击球点偏后
下肢蹬转与上肢挥动存在脱节，力量传导出现断层
4. 主要问题优先级排序
动力链断裂（核心-躯干-手臂的动能传递效率低）
击球时机把握失准
动态平衡能力不足
5. 针对性改进方案
发力优化：进行分腿垫步接交叉步练习，强调前腿蹬伸时髋部前顶的爆发力训练（每日3组×15次）
时机修正：采用多球变速训练，重点练习在身体前侧0.8米处击球（使用标记线辅助定位）
动作衔接：进行闭眼挥拍练习，强化身体记忆完整的随挥轨迹（拍头最终触达对侧腋下）
张力控制：在引拍阶段持拍手进行握力调节训练（引拍时60%握力，击球瞬间增至90%）
综合评分：4.5/10
基础动作框架完整，但关键环节的能量传递效率需提升，通过针对性训练可在2-3周内观察到动力链明显改善。
'''

    # 注册中文字体（若路径不存在请换成系统可用的中文 ttf）
    font_path = '/usr/share/fonts/SimHei.ttf'
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"字体文件不存在: {font_path}")
    pdfmetrics.registerFont(TTFont('NotoCJK', font_path))

    page_w, page_h = A4
    margin = 40  # 页面边距（points）
    gap_after_title = 12
    gap_after_main = 18
    inner_gap = 12

    c = canvas.Canvas(str(pdf_save_path), pagesize=A4)

    # Title
    title = "網球動作評分報告"
    title_fontsize = 20
    c.setFont('NotoCJK', title_fontsize)
    c.setFillColor(colors.black)
    c.drawCentredString(page_w / 2.0, page_h - margin, title)

    # 下一位置（从顶部向下）
    y_cursor = page_h - margin - title_fontsize - gap_after_title

    # 主图：按可用宽度等比缩放
    pts = []
    for i, ball_data in enumerate(ball_list):
        bx = ball_data[0]
        by = ball_data[1]
        if bx != 0 and by != 0:
            p = (int(bx), int(by))
            pts.append(p)
            # 保留圆点标记
            if i == len(ball_list) - 1:  # 最后一个点用红色
                cv2.circle(video_img, p, radius=5, color=(0, 0, 255), thickness=-1)
                cv2.putText(video_img, 'Ball', (int(bx) + 10, int(by) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.circle(video_img, p, radius=5, color=(0, 255, 255), thickness=-1)

    # # 若至少有两点，按顺序用折线连接（非闭合）
    # if len(pts) >= 2:
    #     pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    #     cv2.polylines(video_img, [pts_arr], isClosed=False, color=(0, 255, 255), thickness=2)

    # for ball_data in ball_list:
    #     # 在图像上绘制球的位置
    #     ball_x = ball_data[0]
    #     ball_y = ball_data[1]
    #     if ball_x != 0 and ball_y != 0:
    #         cv2.circle(video_img, (int(ball_x), int(ball_y)), radius=5, color=(0, 255, 255), thickness=-1)
    #         # cv2.putText(video_img, 'Ball', (int(ball_x) + 10, int(ball_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # 若传入的是 OpenCV 图像 (numpy.ndarray)，转换为 PIL Image（并从 BGR->RGB）
    if isinstance(video_img, np.ndarray):
        main_pil = Image.fromarray(cv2.cvtColor(video_img, cv2.COLOR_BGR2RGB))
    elif isinstance(video_img, (str, Path)):
        main_pil = Image.open(str(video_img)).convert("RGB")
    elif isinstance(video_img, Image.Image):
        main_pil = video_img
    else:
        raise TypeError("video_img must be numpy.ndarray, PIL.Image or image path")
    main_img = main_pil
    main_w_px, main_h_px = main_img.size
    avail_w = page_w - 2 * margin
    # 限制主图高度不超过页面大约一半（可调整）
    max_main_h = (page_h - 2 * margin) * 0.55
    scale = min(avail_w / main_w_px, max_main_h / main_h_px)
    main_w_pts = main_w_px * scale
    main_h_pts = main_h_px * scale
    x_main = (page_w - main_w_pts) / 2.0
    y_main = y_cursor - main_h_pts
    c.drawImage(ImageReader(main_img), x_main, y_main, width=main_w_pts, height=main_h_pts, preserveAspectRatio=True, mask='auto')

    # 更新 cursor 到主图下方
    y_cursor = y_main - gap_after_main

    # 底部两列布局：左为文本，右为小图
    # 先处理小图尺寸（限制为可用宽度的一部分）
    # small_img = Image.open(small_img_path)
    # small_img = score_img
    # small_w_px, small_h_px = small_img.size
    # small image: could be Path / str / PIL.Image / numpy.ndarray
    if isinstance(score_img, np.ndarray):
        small_pil = Image.fromarray(cv2.cvtColor(score_img, cv2.COLOR_BGR2RGB))
    elif isinstance(score_img, (str, Path)):
        small_pil = Image.open(str(score_img)).convert("RGB")
    elif isinstance(score_img, Image.Image):
        small_pil = score_img
    else:
        raise TypeError("score_img must be numpy.ndarray, PIL.Image or image path")
    small_img = small_pil
    small_w_px, small_h_px = small_img.size
    # 右栏宽度占可用宽度的约30%（可调整）
    bottom_avail_w = page_w - 2 * margin
    right_col_w = bottom_avail_w * 0.4
    # scale small image to fit right_col_w
    s_scale = min(right_col_w / small_w_px, (page_h * 0.45) / small_h_px)  # 限制高度
    small_w_pts = small_w_px * s_scale
    small_h_pts = small_h_px * s_scale
    x_small = page_w - margin - small_w_pts
    y_small = y_cursor - small_h_pts

    # 绘制小图（右侧）
    c.drawImage(ImageReader(small_img), x_small, y_small, width=small_w_pts, height=small_h_pts, preserveAspectRatio=True, mask='auto')

    # # 文本列宽：剩余空间
    left_col_w = x_small - inner_gap - margin
    # # 使用 Paragraph 处理中文自动换行
    style = ParagraphStyle(
        name='Body',
        fontName='NotoCJK',
        fontSize=11,
        leading=14,
        textColor=colors.black,
    )
    # # text = "综合评分: 85分\n动作分析:\n1. 正手挥拍动作流畅，力量传递良好。\n2. 反手挥拍时身体重心稍显不稳，建议加强核心力量训练。\n3. 击球点选择合理，但有提升空间。\n4. 步伐移动迅速，场上覆盖范围广。\n5. 建议增加多样化的击球练习，提高应变能力。"
    para = Paragraph(report_text.replace('\n', '<br/>'), style)
    # wrap 在指定宽度内并返回所需高度
    w_req, h_req = para.wrap(left_col_w, page_h)  # wrap(width, height)
    # 如果文本高度超过页面剩余高度，则截断或换页（此处简单截断到剩余高度）
    available_height = y_cursor - margin
    if h_req > available_height:
        # 缩小字号或截断 - 这里截断显示
        # 你可以循环减小 fontSize 直到 fit，或者分页
        h_req = available_height

    # 绘制文本（左下角坐标）
    text_x = margin
    text_y = y_cursor - h_req
    # para.drawOn(c, text_x, text_y)

    # 把 markdown 文本转换为 flowables
    flowables = md_to_flowables(report_text, style)

    # 在左列区域逐个绘制 flowables（手工分页/截断）
    left_col_w = x_small - inner_gap - margin
    cur_y = y_cursor  # 当前顶端 y（points）
    for f in flowables:
        w_req, h_req = f.wrap(left_col_w, cur_y - margin)
        if h_req <= 0:
            continue
        # 如果当前 flowable 超过剩余高度，截断或换页；此处简单换页（开始新页）
        if cur_y - h_req < margin:
            c.showPage()
            # 重新绘制 header 可选（此处简单继续新页）
            c.setFont('NotoCJK', title_fontsize)
            c.drawCentredString(page_w / 2.0, page_h - margin, title)
            cur_y = page_h - margin - title_fontsize - gap_after_title
        # drawOn: (canvas, x, y) -> y 是底边位置，所以计算 bottom = cur_y - h_req
        f.drawOn(c, text_x, cur_y - h_req)
        cur_y = cur_y - h_req - 6  # 行间间距 6 points

    c.showPage()
    c.save()
    print(f"PDF 已保存: {pdf_save_path}")
    pass


def show_vitpose_pose(kp_list, img_list, video_writer, cls=None, ball_position=None, origin_video_path:Path=None):
    colors = [
        (255, 0, 0),  # 头到肩部
        (0, 255, 0),  # 左手
        (0, 0, 255),  # 右手
        (255, 255, 0),  # 身体
        (255, 0, 255),  # 左腿
        (0, 255, 255)  # 右腿
    ]
    angle_joints = [
    (5, 7, 9, "L Elbow", (114, 255, 250)),  # 左肘
    (6, 8, 10, "R Elbow", (231, 237, 62)),  # 右肘
    (11, 13, 15, "L Knee", (0, 117, 255)),  # 左膝
    (12, 14, 16, "R Knee", (193, 182, 255))  # 右膝
]
    suggest_angle_range = [
    (90, 170),  # 左肘
    (90, 170),  # 右肘
    (150, 175),  # 左膝
    (150, 175)  # 右膝
]
    joint_color = [
    # BGR
    (114, 255, 250),  # 左手
    (231, 237, 62),  # 右手
    (0, 117, 255),  # 左腿
    (193, 182, 255)  # 右腿
]

    # openpose
    connections = [[0, 1], [0, 2], [0, 5], [0, 6],
                [5, 7], [7, 9],
                [6, 8], [8, 10],
                [5, 6], [5, 11], [6, 12], [11, 12],
                [11, 13], [13, 15],
                [12, 14], [14, 16]]
    LR = [1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6]

    thickness = 3

    # 定义需要计算角度的关节三元组
    all_angles = []
    # all_frames = []
    # 读取每一帧
    print("kps len:", len(kp_list))
    # with tqdm(desc="calculate angle", unit="帧", total=len(kp_list)) as pbar:
    for kps in tqdm(kp_list, desc="calculate angle", unit="帧"):
            # 计算这一帧的角度
            angles = []
            for i, joint in enumerate(angle_joints):
                a_idx, b_idx, c_idx, label, text_color = joint
                if max(a_idx, b_idx, c_idx) >= len(kps):
                    print(f"Warning: Joint indices {a_idx}, {b_idx}, {c_idx} exceed keypoints length {len(kps)}")
                    angles.append(0)  # 或者处理为其他默认值
                    continue
                a = np.array(kps[a_idx][:2])
                b = np.array(kps[b_idx][:2])
                c = np.array(kps[c_idx][:2])
                # if a[0] == 0 and a[1] == 0:
                #     continue
                # if b[0] == 0 and b[1] == 0:
                #     continue
                # if c[0] == 0 and c[1] == 0:
                #     continue
                # # 检查置信度（如果scores可用）
                # if scores is not None:
                #     if scores[a_idx] < conf_threshold or scores[b_idx] < conf_threshold or scores[c_idx] < conf_threshold:
                #         continue

                # 计算角度
                angle = calculate_angle(a, b, c).tolist()
                angles.append(angle)
            all_angles.append(angles)
            # all_frames.append(img)
    # 初始化超出范围的计数器
    out_of_range_counts = np.zeros(len(angle_joints) + 2)
    for angles in all_angles:
        for i, (min_angle, max_angle) in enumerate(suggest_angle_range):
            if angles[i] < min_angle or angles[i] > max_angle:
                out_of_range_counts[i] += 1

    # 计算百分比
    out_of_range_percent = out_of_range_counts / len(all_angles)
    out_of_range_percent[-2] = random.uniform(0.35, 0.65)  # 最后两个位置先随机给个数
    out_of_range_percent[-1] = random.uniform(0.35, 0.65)
    rect_width = 600
    rect_height = 400
    # rect_height = 400
    # if display_in_video:
    #     rect_x = width - rect_width
    #     rect_height = 400
    # else:
    #     rect_x = width
    #     rect_height = height
    width = img_list[0].shape[1]
    height = img_list[0].shape[0]
    rect_x = width - rect_width
    rect_y = 0
    # 定义半透明矩形的颜色和透明度
    color_rect = (0, 0, 0)
    alpha = 0.75  # 透明度
    label = ["左肘", "右肘", "左膝", "右膝", "姿勢穩定程度", "擊球力度"]
    print("out_of_range_percent:", out_of_range_percent)
    score_img_path = origin_video_path.parent / Path(f"{origin_video_path.stem}_score.png")
    # cls = "反手發力"
    create_hexagon_radar_chart(out_of_range_percent, label, "動作完成程度雷達圖", f"當前動作為：{cls}", score_img_path)
    # 读取叠加图片
    overlay = cv2.imread(score_img_path, cv2.IMREAD_UNCHANGED)
    if overlay is None:
        print(f"无法读取图片: {score_img_path}")
        return
    
    h_overlay, w_overlay = overlay.shape[:2]

    # 确定叠加位置
    x = width - w_overlay - 10
    y = 10
    
    for i, (kps, img, ball_data) in tqdm(enumerate(zip(kp_list, img_list, ball_position)), desc="绘制关节和角度", unit="帧"):
        for j, c in enumerate(connections):
            # print(kps.shape)
            if max(c) >= len(kps):
                # print(f"Warning: Connection indices {c} exceed keypoints length {len(kps)}")
                continue
            if kps[c[0]][0] == 0 and kps[c[0]][1] == 0:
                continue
            if kps[c[1]][0] == 0 and kps[c[1]][1] == 0:
                continue
            start = map(int, kps[c[0]])
            end = map(int, kps[c[1]])
            start = list(start)
            end = list(end)
            cv2.line(img, (start[0], start[1]), (end[0], end[1]), colors[LR[j] - 1], thickness)
            cv2.circle(img, (start[0], start[1]), thickness=-1, color=colors[LR[j] - 1], radius=3)
            cv2.circle(img, (end[0], end[1]), thickness=-1, color=colors[LR[j] - 1], radius=3)
        if ball_data:
            if i < BALL_TRACK_NUM:
                ball_list = ball_position[:i + 1]
            else:
                ball_list = ball_position[i - BALL_TRACK_NUM + 1:i + 1]
            for ball_data in ball_list:
                # 在图像上绘制球的位置
                ball_x = ball_data[0]
                ball_y = ball_data[1]
                if ball_x != 0 and ball_y != 0:
                    cv2.circle(img, (int(ball_x), int(ball_y)), radius=5, color=(0, 255, 255), thickness=-1)
                    # cv2.putText(img, 'Ball', (int(ball_x) + 10, int(ball_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # # 将透明图层与原始帧混合
        if i == int(WINDOW_SIZE // 2):
            generate_report(img, ball_position[i:], origin_video_path, score_img=score_img_path, pdf_save_path=origin_video_path.parent / Path(f"{origin_video_path.stem}_report.pdf"))

        img = blend_overlay_into_img(img, overlay, margin=10, alpha_global=alpha)

        video_writer.write(img)
    return


# 辅助：把小 overlay 安全混合到 img 的右上角
def blend_overlay_into_img(img, overlay_small, margin=10, alpha_global=0.75):
    ih, iw = img.shape[:2]
    oh, ow = overlay_small.shape[:2]
    # 如果 overlay 比帧大，按比例缩小到不超过帧宽/高度的一半
    max_w = int(iw * 0.5)
    max_h = int(ih * 0.5)
    if ow > max_w or oh > max_h:
        scale = min(max_w / ow, max_h / oh)
        overlay_small = cv2.resize(overlay_small, (int(ow * scale), int(oh * scale)), interpolation=cv2.INTER_AREA)
        oh, ow = overlay_small.shape[:2]

    # 右上角位置
    x = iw - ow - margin
    y = margin
    # 裁剪越界（理论上不需要，但保险）
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(iw, x + ow), min(ih, y + oh)
    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)
    overlay_crop = overlay_small[oy1:oy2, ox1:ox2]

    h_o, w_o = overlay_crop.shape[:2]
    if h_o == 0 or w_o == 0:
        return img

    roi = img[y1:y1 + h_o, x1:x1 + w_o]

    if overlay_crop.shape[2] == 4:
        # 带 alpha 通道
        overlay_rgb = overlay_crop[:,:,:3].astype(float)
        alpha_mask = (overlay_crop[:,:, 3].astype(float) / 255.0) * alpha_global
        alpha_mask = alpha_mask[..., None]
        inv_mask = 1.0 - alpha_mask
        blended = (alpha_mask * overlay_rgb + inv_mask * roi.astype(float)).astype(img.dtype)
        img[y1:y1 + h_o, x1:x1 + w_o] = blended
    else:
        # 无 alpha，使用 addWeighted 在 ROI 上混合
        blended = cv2.addWeighted(overlay_crop, alpha_global, roi, 1.0 - alpha_global, 0)
        img[y1:y1 + h_o, x1:x1 + w_o] = blended

    return img


def get_keypoint(frame, last_position):
    '''
    frame: PIL.Image
    last_position: 上一帧球员的关节点位置或者球的位置 x, y
    '''
    # 预测
    # 先找人物框
    inputs = person_image_processor(images=frame, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = person_model(**inputs)
    width = frame.shape[1]
    height = frame.shape[0]
    results = person_image_processor.post_process_object_detection(
    outputs, target_sizes=torch.tensor([(height, width)]), threshold=0.3)
    result = results[0]
    person_boxes = result["boxes"][result["labels"] == 0]
    person_boxes = person_boxes.cpu().numpy()
    # # 选取所有人物框中距离last_position最近的人
    # if len(person_boxes) > 1:
    #     centers = (person_boxes[:, 0:2] + person_boxes[:, 2:4]) / 2.0
    #     distances = np.linalg.norm(centers - last_position, axis=1)
    #     closest_idx = np.argmin(distances)
    #     last_position = centers[closest_idx]
    #     person_box = person_boxes[closest_idx].reshape(1, 4)
    # elif len(person_boxes) == 0:
    #     last_position = (person_boxes[0, 0:2] + person_boxes[0, 2:4]) / 2.0
    #     person_box = person_boxes[0].reshape(1, 4)
    # else:
    #     print("no keypoints detected, skip this frame")
    #     return None, last_position
    if len(person_boxes) == 0:
        print("no keypoints detected, skip this frame")
        return None, last_position
    # 根据人物框提取骨骼
    # Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) format
    person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
    person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]
    inputs = image_processor(frame, boxes=[person_boxes], return_tensors="pt").to(device)
    inputs["dataset_index"] = torch.tensor([0], device=device)
    with torch.no_grad():
        outputs = model(**inputs)
    # print(outputs)
    pose_results = image_processor.post_process_pose_estimation(outputs, boxes=[person_boxes], threshold=0.3)
    result = pose_results[0][0]["keypoints"].cpu().numpy()  # results for first image
    return result, last_position


def find_nearest_person(joint_data, target_point):
    """
    joint_data: numpy数组, 形状为(n, 25, 3), 表示n个人的关节点数据
    target_point: 目标坐标点 [x, y]
    返回: 最近的人的索引
    """
    # 直接计算每个人的坐标算术平均（忽略置信度）
    centers = np.mean(joint_data[:,:,:2], axis=1)  # 形状 (n, 2)
    
    # 计算到目标点的欧氏距离
    distances = np.linalg.norm(centers - target_point, axis=1)
    
    # 返回最近的人的索引
    return np.argmin(distances)


def get_vitpose_keypoint(video_path:Path, ball_data:pd.DataFrame, model_list, model_weight_list, bone_list, vel_list):
    # 先提取击球帧的人，若检测到多于一个人则取里球最近的人作为球员
    print(f"Processing video: {video_path}")
    _, start_frame, end_frame, strike_position = video_path.stem.split('_')
    start_frame, end_frame , strike_position = int(start_frame), int(end_frame), int(strike_position)
    strike_position_in_video = strike_position - start_frame
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, strike_position_in_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    ret, frame = cap.read()
    keypoint_list = []
    frame_list = []
    ball_position = []
    
    ball_strike_position = np.array([ball_data.loc[ball_data['Frame'] == strike_position, 'X'].values[0], ball_data.loc[ball_data['Frame'] == strike_position, 'Y'].values[0]])
    ball_position.append(ball_strike_position.tolist())
    keypoint, lastposition = get_keypoint(frame, ball_strike_position)
    if keypoint is None:
        print(f"No keypoint detected in frame {strike_position_in_video} of video {video_path.name}, skipping...")
        return None
    keypoint_list.append(keypoint)
    frame_list.append(frame)
    # 从击球帧开始逐帧向前读取，获取球员的关节点
    j = 1
    for i in range(strike_position_in_video - 1, -1, -1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        keypoint, lastposition = get_keypoint(frame, lastposition)
        if keypoint is None:
            print(f"No keypoint detected in frame {i} of video {video_path.name}, skipping...")
            continue
        keypoint_list.insert(0, keypoint)
        frame_list.insert(0, frame)
        ball_position.insert(0, [ball_data.loc[ball_data['Frame'] == strike_position - j, 'X'].values[0], ball_data.loc[ball_data['Frame'] == strike_position - j, 'Y'].values[0]])
        j += 1
    lastposition = ball_strike_position
    j = 1
    # 从击球帧开始逐帧向后读取，获取球员的关节点
    for i in range(strike_position_in_video + 1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        keypoint, lastposition = get_keypoint(frame, lastposition)
        if keypoint is None:
            print(f"No keypoint detected in frame {i} of video {video_path.name}, skipping...")
            continue
        keypoint_list.append(keypoint)
        frame_list.append(frame)
        ball_position.append([ball_data.loc[ball_data['Frame'] == strike_position + j, 'X'].values[0], ball_data.loc[ball_data['Frame'] == strike_position + j, 'Y'].values[0]])
        j += 1
    # 动作分类识别
    # 首先将少于17个点的去掉
    keypoints = [kp for kp in keypoint_list if len(kp) >= 17]
    # 计算剩余帧数百分比，低于阈值则不进行动作分类
    if len(keypoints) / WINDOW_SIZE < MIN_FRAME_PERCENTAGE:
        print(f"Not enough frames for action classification in video {video_path.name}, skipping...")
        action_cls = CATEGORIES[-1]
    # 数据预处理
    keypoints = np.array(keypoints)
    # keypoints = keypoints / np.array([width, height])
    keypoints = keypoints.reshape(keypoints.shape[0], 17, 2)

    # 归一化
    # 1. 提取 x 和 y 坐标
    x = keypoints[:,:, 0]  # 形状 (n, 17)
    y = keypoints[:,:, 1]  # 形状 (n, 17)
    # 2. 计算每帧的极值
    x_min = np.min(x, axis=1, keepdims=True)  # 每帧 x 最小值，形状 (n, 1)
    x_max = np.max(x, axis=1, keepdims=True)  # 每帧 x 最大值，形状 (n, 1)
    y_min = np.min(y, axis=1, keepdims=True)  # 每帧 y 最小值，形状 (n, 1)
    y_max = np.max(y, axis=1, keepdims=True)  # 每帧 y 最大值，形状 (n, 1)
    # 3. 计算归一化范围（避免除零）
    x_range = x_max - x_min
    y_range = y_max - y_min
    # 处理范围为零的情况（防止除以零）
    x_range[x_range == 0] = 1
    y_range[y_range == 0] = 1
    # 4. 归一化坐标
    x_norm = (x - x_min) / x_range
    y_norm = (y - y_min) / y_range
    # 5. 组合回原数组形状
    keypoints = np.stack([x_norm, y_norm], axis=2)

    data_numpy = np.zeros((WINDOW_SIZE, 1, 17, 2))
    for i in range(len(keypoints)):
        data_numpy[i, 0,:,:] = keypoints[i]
    # input = torch.FloatTensor(data_numpy).permute(3, 0, 2, 1).cuda().unsqueeze(0) 
    action_cls = CATEGORIES[get_cls(data_numpy, model_list, model_weight_list, bone_list, vel_list)]
    print(f"Action class for {video.name}: {action_cls}")
    # 保存视频
    save_name = video_path.parent / Path(video_path.stem + '_with_keypoint.mp4')
    video_writer = cv2.VideoWriter(save_name, fourcc, fps, (width, height))
    # show_openpose_pose(keypoint_list, frame_list, video_writer, cls=action_cls, ball_position=ball_position)
    show_vitpose_pose(keypoint_list, frame_list, video_writer, cls=action_cls, ball_position=ball_position, origin_video_path=video_path)

    # for kp, img , bp in zip(keypoint_list, frame_list, ball_position):
        # img = show_openpose_pose(kp, img, ball_data=bp)
        # img = plot_action_cls(action_cls, img)
        # video_writer.write(img)
    video_writer.release()
    cap.release()
    return keypoint_list, frame_list


def data_normalization(input, bone=False, vel=False):
    # print(f"Original input shape: {input.shape}")
    # input = input.squeeze(0)
    data_numpy = input.transpose(3, 0, 2, 1)  # C,T,V,M
    # data_numpy = np.array(data_numpy)
    valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
    # print(f"Valid frames: {valid_frame_num}")
    # if(valid_frame_num == 0): 
    #     return np.zeros((2, 64, 17, 2)), label, idx
    # reshape Tx(MVC) to CTVM
    data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, [0.5, 1], 5)
    if bone:
        bone_data_numpy = np.zeros_like(data_numpy)
        for v1, v2 in coco_pairs:
            bone_data_numpy[:,:, v1 - 1] = data_numpy[:,:, v1 - 1] - data_numpy[:,:, v2 - 1]
        data_numpy = bone_data_numpy
    if vel:
        data_numpy[:,:-1] = data_numpy[:, 1:] - data_numpy[:,:-1]
        data_numpy[:, -1] = 0

    data_numpy = data_numpy - np.tile(data_numpy[:,:, 0:1,:], (1, 1, 17, 1))  # all_joint - 0_joint
    # pose = torch.tensor(data_numpy).unsqueeze(0).permute(0, 4, 1, 3, 2).float()  # N C T V M
    pose = torch.tensor(data_numpy).unsqueeze(0).float()
    return pose

# def show_all_openpose_pose(video_path:Path):
#     cap = cv2.VideoCapture(str(video_path))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     save_name = video_path.parent / Path(video_path.stem + '_with_keypoint.mp4')
#     video_writer = cv2.VideoWriter(save_name, fourcc, fps, (width, height))
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         keypoint = get_keypoint(frame)
#         if keypoint is None:
#             print("no keypoints detected, skip this frame")
#             continue
#         for i in range(keypoint.shape[0]):
#             frame = show_openpose_pose(keypoint[i], frame)
#         video_writer.write(frame)
#     video_writer.release()
#     cap.release()


def get_cls(input, model_list, model_weight_list, bone_list, vel_list):
    result = []
    for i, (model, weight, bone, vel) in enumerate(zip(model_list, model_weight_list, bone_list, vel_list)):
        x = data_normalization(input, bone=bone, vel=vel).cuda()
        model.eval()
        print("-"*20)
        logits = model(x)
        
        print(logits)
        print(f"Model {i+1} logits shape: {logits.size()}")
        # logits = logits * torch.tensor(weight)
        logits = torch.nn.functional.softmax(logits)* torch.tensor(weight)
        print(f"Model {i+1}")
        print(logits)
        print(CATEGORIES[torch.max(logits.data, 1)[1].item()])
        print("-"*20)
        result.append(logits.cpu())
    result = torch.stack(result, dim=0)
    result = torch.mean(result, dim=0)
    print(result)
    _, predict_label = torch.max(result.data, 1)
    return predict_label.item()


def plot_action_cls(cls, frame):
    
    height, width = frame.shape[:2]
    # 创建与原图相同大小的黑色覆盖层
    overlay = np.zeros_like(frame)

    # 在覆盖层上绘制白色（或黑色）矩形
    top_left = (width - 600, 0)
    bottom_right = (width, 400)
    cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 0), thickness=-1)

    # 设置透明度（alpha 控制覆盖层的透明度，beta 控制原图的透明度）
    alpha = 0.5  # 越大越不透明
    beta = 1 - alpha

    # 图像混合
    output = cv2.addWeighted(frame, alpha, overlay, beta, 0)
    # 在矩形内绘制文本
    cv2.putText(output, f'Action: {cls}', (top_left[0] + 20, top_left[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return output


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file', type=str, help='file path of the video')
    parser.add_argument('--tracknet_file', type=str, default="/home/awsdjikl/TrackNetV3/exp/TrackNet_best.pt", help='file path of the TrackNet model checkpoint')
    parser.add_argument('--inpaintnet_file', type=str, default='', help='file path of the InpaintNet model checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for inference')
    parser.add_argument('--eval_mode', type=str, default='weight', choices=['nonoverlap', 'average', 'weight'], help='evaluation mode')
    parser.add_argument('--max_sample_num', type=int, default=1800, help='maximum number of frames to sample for generating median image')
    parser.add_argument('--video_range', type=lambda splits: [int(s) for s in splits.split(',')], default=None, help='range of start second and end second of the video for generating median image')
    parser.add_argument('--save_dir', type=str, default='pred_result', help='directory to save the prediction result')
    parser.add_argument('--large_video', action='store_true', default=False, help='whether to process large video')
    parser.add_argument('--output_video', action='store_true', default=True, help='whether to output video with predicted trajectory')
    parser.add_argument('--traj_len', type=int, default=8, help='length of trajectory to draw on video')
    
    parser.add_argument(
        '--phase', default='test', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')
    
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        action=DictAction,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')
    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--lr-decay-rate',
        type=float,
        default=0.1,
        help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=0)
    
    args = parser.parse_args()

    args.seed = 1
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    num_workers = args.batch_size if args.batch_size <= 16 else 16
    video_file = args.video_file
    video_name = video_file.split('/')[-1][:-4]
    video_range = args.video_range if args.video_range else None
    large_video = args.large_video
    args.save_dir = Path(args.video_file).parent
    out_csv_file = os.path.join(args.save_dir, f'{video_name}_ball.csv')
    out_video_file = os.path.join(args.save_dir, f'{video_name}_predict.mp4')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print(f'Save prediction result to {args.save_dir}')
    # Load model
    tracknet_ckpt = torch.load(args.tracknet_file, weights_only=False)
    tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']
    bg_mode = tracknet_ckpt['param_dict']['bg_mode']
    tracknet = get_model('TrackNet', tracknet_seq_len, bg_mode).cuda()
    tracknet.load_state_dict(tracknet_ckpt['model'])

    if args.inpaintnet_file:
        inpaintnet_ckpt = torch.load(args.inpaintnet_file, weights_only=False)
        inpaintnet_seq_len = inpaintnet_ckpt['param_dict']['seq_len']
        inpaintnet = get_model('InpaintNet').cuda()
        inpaintnet.load_state_dict(inpaintnet_ckpt['model'])
    else:
        inpaintnet = None

    cap = cv2.VideoCapture(args.video_file)
    w, h = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    w_scaler, h_scaler = w / WIDTH, h / HEIGHT
    img_scaler = (w_scaler, h_scaler)

    tracknet_pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[], 'Inpaint_Mask':[],
                        'Img_scaler': (w_scaler, h_scaler), 'Img_shape': (w, h)}

    # Test on TrackNet
    tracknet.eval()
    seq_len = tracknet_seq_len
    if args.eval_mode == 'nonoverlap':
        # Create dataset with non-overlap sampling
        if large_video:
            print("Processing large video with non-overlap sampling...")
            dataset = Video_IterableDataset(video_file, seq_len=seq_len, sliding_step=seq_len, bg_mode=bg_mode,
                                            max_sample_num=args.max_sample_num, video_range=video_range)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
            print(f'Video length: {dataset.video_len}')
        else:
            # Sample all frames from video
            frame_list = generate_frames(args.video_file)
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=seq_len, data_mode='heatmap', bg_mode=bg_mode,
                                                 frame_arr=np.array(frame_list)[:,:,:,::-1], padding=True)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

        for step, (i, x) in enumerate(tqdm(data_loader)):
            x = x.float().cuda()
            with torch.no_grad():
                y_pred = tracknet(x).detach().cpu()
                # print(y_pred.shape)
            # Predict
            tmp_pred = predict(i, y_pred=y_pred, img_scaler=img_scaler)
            for key in tmp_pred.keys():
                tracknet_pred_dict[key].extend(tmp_pred[key])
    else:
        # Create dataset with overlap sampling for temporal ensemble
        if large_video:
            dataset = Video_IterableDataset(video_file, seq_len=seq_len, sliding_step=1, bg_mode=bg_mode,
                                            max_sample_num=args.max_sample_num, video_range=video_range)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
            video_len = dataset.video_len
            print(f'Video length: {video_len}')
            
        else:
            # Sample all frames from video
            frame_list = generate_frames(args.video_file)
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=1, data_mode='heatmap', bg_mode=bg_mode,
                                                 frame_arr=np.array(frame_list)[:,:,:,::-1])
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
            video_len = len(frame_list)
        
        # Init prediction buffer params
        num_sample, sample_count = video_len - seq_len + 1, 0
        buffer_size = seq_len - 1
        batch_i = torch.arange(seq_len)  # [0, 1, 2, 3, 4, 5, 6, 7]
        frame_i = torch.arange(seq_len - 1, -1, -1)  # [7, 6, 5, 4, 3, 2, 1, 0]
        y_pred_buffer = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
        weight = get_ensemble_weight(seq_len, args.eval_mode)
        for step, (i, x) in enumerate(tqdm(data_loader)):
            x = x.float().cuda()
            b_size, seq_len = i.shape[0], i.shape[1]
            with torch.no_grad():
                y_pred = tracknet(x).detach().cpu()
                # print(y_pred.shape)
            
            y_pred_buffer = torch.cat((y_pred_buffer, y_pred), dim=0)
            ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
            ensemble_y_pred = torch.empty((0, 1, HEIGHT, WIDTH), dtype=torch.float32)

            for b in range(b_size):
                if sample_count < buffer_size:
                    # Imcomplete buffer
                    y_pred = y_pred_buffer[batch_i + b, frame_i].sum(0) / (sample_count + 1)
                else:
                    # General case
                    y_pred = (y_pred_buffer[batch_i + b, frame_i] * weight[:, None, None]).sum(0)
                
                ensemble_i = torch.cat((ensemble_i, i[b][0].reshape(1, 1, 2)), dim=0)
                ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred.reshape(1, 1, HEIGHT, WIDTH)), dim=0)
                sample_count += 1

                if sample_count == num_sample:
                    # Last batch
                    y_zero_pad = torch.zeros((buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32)
                    y_pred_buffer = torch.cat((y_pred_buffer, y_zero_pad), dim=0)

                    for f in range(1, seq_len):
                        # Last input sequence
                        y_pred = y_pred_buffer[batch_i + b + f, frame_i].sum(0) / (seq_len - f)
                        ensemble_i = torch.cat((ensemble_i, i[-1][f].reshape(1, 1, 2)), dim=0)
                        ensemble_y_pred = torch.cat((ensemble_y_pred, y_pred.reshape(1, 1, HEIGHT, WIDTH)), dim=0)

            # Predict
            tmp_pred = predict(ensemble_i, y_pred=ensemble_y_pred, img_scaler=img_scaler)
            for key in tmp_pred.keys():
                tracknet_pred_dict[key].extend(tmp_pred[key])

            # Update buffer, keep last predictions for ensemble in next iteration
            y_pred_buffer = y_pred_buffer[-buffer_size:]

    # assert video_len == len(tracknet_pred_dict['Frame']), 'Prediction length mismatch'
    # Test on TrackNetV3 (TrackNet + InpaintNet)
    if inpaintnet is not None:
        inpaintnet.eval()
        seq_len = inpaintnet_seq_len
        tracknet_pred_dict['Inpaint_Mask'] = generate_inpaint_mask(tracknet_pred_dict, th_h=h * 0.05)
        inpaint_pred_dict = {'Frame':[], 'X':[], 'Y':[], 'Visibility':[]}

        if args.eval_mode == 'nonoverlap':
            # Create dataset with non-overlap sampling
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=seq_len, data_mode='coordinate', pred_dict=tracknet_pred_dict, padding=True)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

            for step, (i, coor_pred, inpaint_mask) in enumerate(tqdm(data_loader)):
                coor_pred, inpaint_mask = coor_pred.float(), inpaint_mask.float()
                with torch.no_grad():
                    coor_inpaint = inpaintnet(coor_pred.cuda(), inpaint_mask.cuda()).detach().cpu()
                    coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1 - inpaint_mask)  # replace predicted coordinates with inpainted coordinates
                
                # Thresholding
                th_mask = ((coor_inpaint[:,:, 0] < COOR_TH) & (coor_inpaint[:,:, 1] < COOR_TH))
                coor_inpaint[th_mask] = 0.
                
                # Predict
                tmp_pred = predict(i, c_pred=coor_inpaint, img_scaler=img_scaler)
                for key in tmp_pred.keys():
                    inpaint_pred_dict[key].extend(tmp_pred[key])
                
        else:
            # Create dataset with overlap sampling for temporal ensemble
            dataset = Shuttlecock_Trajectory_Dataset(seq_len=seq_len, sliding_step=1, data_mode='coordinate', pred_dict=tracknet_pred_dict)
            data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
            weight = get_ensemble_weight(seq_len, args.eval_mode)

            # Init buffer params
            num_sample, sample_count = len(dataset), 0
            buffer_size = seq_len - 1
            batch_i = torch.arange(seq_len)  # [0, 1, 2, 3, 4, 5, 6, 7]
            frame_i = torch.arange(seq_len - 1, -1, -1)  # [7, 6, 5, 4, 3, 2, 1, 0]
            coor_inpaint_buffer = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)
            
            for step, (i, coor_pred, inpaint_mask) in enumerate(tqdm(data_loader)):
                coor_pred, inpaint_mask = coor_pred.float(), inpaint_mask.float()
                b_size = i.shape[0]
                with torch.no_grad():
                    coor_inpaint = inpaintnet(coor_pred.cuda(), inpaint_mask.cuda()).detach().cpu()
                    coor_inpaint = coor_inpaint * inpaint_mask + coor_pred * (1 - inpaint_mask)
                
                # Thresholding
                th_mask = ((coor_inpaint[:,:, 0] < COOR_TH) & (coor_inpaint[:,:, 1] < COOR_TH))
                coor_inpaint[th_mask] = 0.

                coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_inpaint), dim=0)
                ensemble_i = torch.empty((0, 1, 2), dtype=torch.float32)
                ensemble_coor_inpaint = torch.empty((0, 1, 2), dtype=torch.float32)
                
                for b in range(b_size):
                    if sample_count < buffer_size:
                        # Imcomplete buffer
                        coor_inpaint = coor_inpaint_buffer[batch_i + b, frame_i].sum(0)
                        coor_inpaint /= (sample_count + 1)
                    else:
                        # General case
                        coor_inpaint = (coor_inpaint_buffer[batch_i + b, frame_i] * weight[:, None]).sum(0)
                    
                    ensemble_i = torch.cat((ensemble_i, i[b][0].view(1, 1, 2)), dim=0)
                    ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), dim=0)
                    sample_count += 1

                    if sample_count == num_sample:
                        # Last input sequence
                        coor_zero_pad = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)
                        coor_inpaint_buffer = torch.cat((coor_inpaint_buffer, coor_zero_pad), dim=0)
                        
                        for f in range(1, seq_len):
                            coor_inpaint = coor_inpaint_buffer[batch_i + b + f, frame_i].sum(0)
                            coor_inpaint /= (seq_len - f)
                            ensemble_i = torch.cat((ensemble_i, i[-1][f].view(1, 1, 2)), dim=0)
                            ensemble_coor_inpaint = torch.cat((ensemble_coor_inpaint, coor_inpaint.view(1, 1, 2)), dim=0)

                # Thresholding
                th_mask = ((ensemble_coor_inpaint[:,:, 0] < COOR_TH) & (ensemble_coor_inpaint[:,:, 1] < COOR_TH))
                ensemble_coor_inpaint[th_mask] = 0.

                # Predict
                tmp_pred = predict(ensemble_i, c_pred=ensemble_coor_inpaint, img_scaler=img_scaler)
                for key in tmp_pred.keys():
                    inpaint_pred_dict[key].extend(tmp_pred[key])
                
                # Update buffer, keep last predictions for ensemble in next iteration
                coor_inpaint_buffer = coor_inpaint_buffer[-buffer_size:]

    # Write csv file
    pred_dict = inpaint_pred_dict if inpaintnet is not None else tracknet_pred_dict
    write_pred_csv(pred_dict, save_file=out_csv_file)

    # Write video with predicted coordinates
    if args.output_video:
        write_pred_video(video_file, pred_dict, save_file=out_video_file, traj_len=args.traj_len)

    print('Track Ball Done.')
    
    tracknet = None
    torch.cuda.empty_cache()

    # ball_data = pd.read_csv("/home/awsdjikl/TrackNetV3/prediction/game3_ball.csv")
    # video_file = "/home/awsdjikl/TrackNetV3/prediction/game3.mp4"
    # plot_ball_frame(ball_frame)

    ball_data = pd.read_csv(out_csv_file)
    all_strike_positions = get_shot_frame(ball_data)
    # video_list = slice_videos(video_file, all_strike_positions, lenth=30)
    video_list = slice_videos(video_file, all_strike_positions, lenth=WINDOW_SIZE)
    print('Slice Shot Frame Done.')
    # yolo_model = YOLO('/home/awsdjikl/TrackNetV3/models/yolo11n-pose.pt', "pose")
    model_list = []
    config_list = [
        "/home/awsdjikl/TrackNetV3/GCN/config/j.yaml",
        "/home/awsdjikl/TrackNetV3/GCN/config/b.yaml",
        "/home/awsdjikl/TrackNetV3/GCN/config/jm.yaml",
        "/home/awsdjikl/TrackNetV3/GCN/config/bm.yaml",
    ]
    work_dir_list = [
        "/home/awsdjikl/TrackNetV3/GCN/work_dir/j",
        "/home/awsdjikl/TrackNetV3/GCN/work_dir/b",
        "/home/awsdjikl/TrackNetV3/GCN/work_dir/jm",
        "/home/awsdjikl/TrackNetV3/GCN/work_dir/bm",
    ]
    for c, wd in zip(config_list, work_dir_list):
        model_parser = get_parser()
        p = model_parser.parse_args()
        p.config = c
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        model_parser.set_defaults(**default_arg)
        model_args = model_parser.parse_args()
        model_args.work_dir = wd
        model_args.weights = os.path.join(wd, 'runs-65-1040.pt')
        p = Processor(model_args)
        p.model.eval()
        model_list.append(p.model)
    # model_weight_list = [0.0001, 2.0, 0.0001, 0.0001]
    model_weight_list = [1.1857299519882143, 1.688547070587177, 1.7159054406837517, 1.6945187523943728]
    bone_list = [False, True, False, True]
    vel_list = [False, False, True, True]
    for video in video_list:
        keypoint_list, frame_list = get_vitpose_keypoint(video, ball_data, model_list, model_weight_list, bone_list, vel_list)
        
        # 保存骨骼数据
        # 将骨骼数据导入st-gcn
        # show_all_openpose_pose(video)
        # get_yolo11_keypoint(video, ball_data, yolo_model)

