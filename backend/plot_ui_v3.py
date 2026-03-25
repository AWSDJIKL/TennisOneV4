import os
import time
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont  # 新增导入
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation,
)

# import yolo.tennis_detector as td
import backend.pose_analyse as pa
import backend.n1n_api
import gcn_test as gt

device = "cuda"
access_token = (
    "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"  # 如果模型私有，需要填入访问token
)
person_image_processor = AutoProcessor.from_pretrained(
    "PekingU/rtdetr_r50vd_coco_o365", token=access_token
)
person_model = RTDetrForObjectDetection.from_pretrained(
    "PekingU/rtdetr_r50vd_coco_o365", device_map=device, token=access_token
)

image_processor = AutoProcessor.from_pretrained(
    "usyd-community/vitpose-plus-base", token=access_token
)
model = VitPoseForPoseEstimation.from_pretrained(
    "usyd-community/vitpose-plus-base", device_map=device, token=access_token
)

import math

CANVAS_W, CANVAS_H = 1920, 1080
REGION_W, REGION_H = CANVAS_W // 3, CANVAS_H // 2
ABILITY_NAMES = ["动作标准性", "肘部", "腋下", "姿态重心", "躯干", "膝部"]
_last_similarity = 0.0
FONT_PATHS = [
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansSC-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]


FEDERER_STANDARD = {
    "forehand_right": np.load("./backend/Federer_forehand_right_angles.npy"),
    "forehand_left": np.load("./backend/Federer_forehand_left_angles.npy"),
    "backhand_right": np.load("./backend/Federer_backhand_right_angles.npy"),
    "backhand_left": np.load("./backend/Federer_backhand_left_angles.npy"),
}
TAI_STANDARD = {
    "forehand_right": np.load("./backend/Tai_forehand_right_angles.npy"),
    "forehand_left": np.load("./backend/Tai_forehand_left_angles.npy"),
    "backhand_right": np.load("./backend/Tai_backhand_right_angles.npy"),
    "backhand_left": np.load("./backend/Tai_backhand_left_angles.npy"),
}

_font_cache = {}

# 颜色控制（RGBA）
COLOR_BAR_LOW = (255, 0, 0, 140)  # 更透明
COLOR_BAR_MID = (240, 255, 0, 140)
COLOR_BAR_HIGH = (0, 200, 0, 140)
COLOR_BAR_BG = (200, 200, 200, 110)
COLOR_BAR_BORDER = (60, 60, 60, 255)

HEX_BASE_COLOR = (160, 160, 160, 100)  # 更透明
HEX_FILL_COLOR = (39, 245, 230, 200)
HEX_BORDER_COLOR = (255, 255, 255, 10)
HEX_TICK_COLOR = (120, 120, 120, 255)


def _get_font(size: int):
    """Try known paths first, then use fc-list to pick a CJK font, else fallback to default."""
    global _chosen_font_path
    if size in _font_cache:
        return _font_cache[size]
    # try known explicit paths
    for p in FONT_PATHS:
        try:
            if os.path.exists(p):
                f = ImageFont.truetype(p, size)
                _font_cache[size] = f
                _chosen_font_path = p
                return f
        except Exception:
            continue
    # try fc-list to find a zh/CJK font (if fontconfig available)
    try:
        import subprocess

        cmd = "fc-list :lang=zh -f '%{file}\\n' | head -n 1"
        out = subprocess.check_output(["bash", "-lc", cmd], stderr=subprocess.DEVNULL)
        fp = out.decode().strip()
        if fp and os.path.exists(fp):
            f = ImageFont.truetype(fp, size)
            _font_cache[size] = f
            _chosen_font_path = fp
            return f
    except Exception:
        pass
    # final fallback (limited unicode support)
    f = ImageFont.load_default()
    _font_cache[size] = f
    _chosen_font_path = None
    return f


def _rgba_to_bgr(rgba):
    r, g, b, _ = rgba
    return (b, g, r)


def _alpha(rect_alpha):
    return rect_alpha[3] / 255.0


def _draw_rect_alpha(img, pt1, pt2, rgba):
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, _rgba_to_bgr(rgba), -1)
    cv2.addWeighted(overlay, _alpha(rgba), img, 1 - _alpha(rgba), 0, img)


def _fill_poly_alpha(img, pts, rgba, thickness=-1):
    overlay = img.copy()
    if thickness == -1:
        cv2.fillPoly(overlay, [pts], _rgba_to_bgr(rgba))
    else:
        cv2.polylines(overlay, [pts], True, _rgba_to_bgr(rgba), thickness)
    cv2.addWeighted(overlay, _alpha(rgba), img, 1 - _alpha(rgba), 0, img)


def PlotOriginalVideo(frame_list):
    """
    直接将原视频的帧列表显示在窗口的左上角区域。将原本视频帧缩放到适合显示区域的大小，保持宽高比不变。
    """
    canvas = np.full((REGION_H, REGION_W, 3), 255, dtype=np.uint8)  # 白底
    if not frame_list:
        return canvas
    frame = frame_list[-1]
    if isinstance(frame, Image.Image):
        frame_np = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    else:
        frame_np = frame.copy()
    h, w = frame_np.shape[:2]
    scale = min(REGION_W / w, REGION_H / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame_np, (new_w, new_h))
    x0 = (REGION_W - new_w) // 2
    y0 = (REGION_H - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    cv2.rectangle(canvas, (0, 0), (REGION_W - 1, REGION_H - 1), (200, 200, 200), 2)
    return canvas


def PlotHexagon(hexagon_data: np.ndarray):
    """
    绘制六边形。
    输入为6维能力数值的numpy数组，大小为(6,1).能力数值范围为0-100.
    六边形底色为灰色，边框为黑色，填充为天青色。
    能力值从六边形最上方开始逆时针排序，顺序为[动作标准性，肘部，手腕，姿态稳定性，躯干，膝部]。
    在六边形的每个顶点上显示对应的能力数值和能力名称，字体为黑色，大小适中。
    """
    canvas = np.full((REGION_H, REGION_W, 3), 255, dtype=np.uint8)  # 白底
    center = (REGION_W // 2, REGION_H // 2)
    R = int(min(REGION_W, REGION_H) * 0.35)
    angles = [math.radians(-90 + 60 * i) for i in range(6)]
    base_pts = []
    for a in angles:
        x = int(center[0] + R * math.cos(a))
        y = int(center[1] + R * math.sin(a))
        base_pts.append((x, y))
    base_pts_np = np.array(base_pts, np.int32)

    # 底色与边框（带透明度）
    _fill_poly_alpha(canvas, base_pts_np, HEX_BASE_COLOR)
    cv2.polylines(canvas, [base_pts_np], True, _rgba_to_bgr(HEX_BORDER_COLOR), 3)

    # 刻度线（4 等分）
    for ratio in [0.25, 0.5, 0.75, 1.0]:
        tick_pts = []
        for a in angles:
            x = int(center[0] + R * ratio * math.cos(a))
            y = int(center[1] + R * ratio * math.sin(a))
            tick_pts.append((x, y))
        tick_pts_np = np.array(tick_pts, np.int32)
        _fill_poly_alpha(canvas, tick_pts_np, HEX_TICK_COLOR, thickness=1)

    vals = (
        np.clip(hexagon_data.reshape(-1), 0, 100)
        if hexagon_data is not None
        else np.zeros(6)
    )
    val_pts = []
    for a, v in zip(angles, vals):
        r = R * (v / 100.0)
        x = int(center[0] + r * math.cos(a))
        y = int(center[1] + r * math.sin(a))
        val_pts.append((x, y))
    val_pts_np = np.array(val_pts, np.int32)
    _fill_poly_alpha(canvas, val_pts_np, HEX_FILL_COLOR)
    cv2.polylines(canvas, [val_pts_np], True, _rgba_to_bgr(HEX_BORDER_COLOR), 2)

    # 文字进一步外移
    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    for pt, name, v in zip(base_pts, ABILITY_NAMES, vals):
        dx, dy = pt[0] - center[0], pt[1] - center[1]
        norm = math.hypot(dx, dy) + 1e-6
        offset = 70  # 外移距离增大
        tx = pt[0] + int(dx / norm * offset)
        ty = pt[1] + int(dy / norm * offset)
        draw.text(
            (tx - 16, ty - 16), f"{name}:{int(v)}", font=_get_font(20), fill=(0, 0, 0)
        )
    canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return canvas


def PlotScore(score: np.ndarray):
    """
    绘制分数。
    分数数组大小为(5,1)，分别为[总分数，预备动作分数，击球动作分数，挥拍动作分数]。分数范围为0-100.
    在最上方为标题“总分”，右边写总分数，字体较大
    下方纵向排列显示其他四项分数，格式为“击球点 进度条 分数”，字体较小。
    四项分数标题与分数中间带有进度条显示分数占比，进度条为灰色底色，填充颜色由分数决定，低于60为红色，60-85为黄色，高于85为绿色。
    """
    canvas = np.full((REGION_H, REGION_W, 3), 255, dtype=np.uint8)

    scores = np.zeros(5, dtype=np.float32)
    if score is not None:
        flat = score.reshape(-1)
        scores[: min(5, len(flat))] = flat[:5]
    scores = np.clip(scores, 0, 100)
    scores_int = scores.astype(int)

    labels = ["预备动作", "击球动作", "挥拍动作"]
    bar_x, bar_w, bar_h = 140, REGION_W - 220, 28
    start_y = 150
    step_y = 70
    for i, name in enumerate(labels, start=1):
        y = start_y + (i - 1) * step_y
        val = scores_int[i]
        if val < 60:
            color = COLOR_BAR_LOW
        elif val < 85:
            color = COLOR_BAR_MID
        else:
            color = COLOR_BAR_HIGH
        _draw_rect_alpha(canvas, (bar_x, y), (bar_x + bar_w, y + bar_h), COLOR_BAR_BG)
        filled_w = int(bar_w * (val / 100.0))
        _draw_rect_alpha(canvas, (bar_x, y), (bar_x + filled_w, y + bar_h), color)
        cv2.rectangle(
            canvas,
            (bar_x, y),
            (bar_x + bar_w, y + bar_h),
            _rgba_to_bgr(COLOR_BAR_BORDER),
            2,
        )

    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text((20, 40), "总分", font=_get_font(36), fill=(0, 0, 0))
    draw.text(
        (REGION_W - 220, 40), f"{scores_int[0]}", font=_get_font(54), fill=(34, 177, 76)
    )
    for i, name in enumerate(labels, start=1):
        y = start_y + (i - 1) * step_y
        val = scores_int[i]
        draw.text((20, y), name, font=_get_font(24), fill=(0, 0, 0))
        draw.text((bar_x + bar_w + 15, y), f"{val}", font=_get_font(24), fill=(0, 0, 0))
    canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return canvas


def PlotSimilarity(player_frame_list, standard_frame_list, frame_idx: int):
    """
    绘制相似度。
    将玩家动作帧序列与标准动作帧序列左右拼接播放，保持相同帧高度。
    若某一序列已播完，则停留在其最后一帧静止，直到另一序列播放完毕。
    """
    canvas = np.full((REGION_H, REGION_W, 3), 255, dtype=np.uint8)  # 白底
    target_w = REGION_W // 2
    target_h = REGION_H

    def _pick(frames, idx):
        if not frames:
            return None
        if idx < len(frames):
            return frames[idx]
        # 超出长度则返回最后一帧
        return frames[-1]

    def _place(frame, x0):
        nonlocal canvas
        if frame is None:
            cv2.rectangle(
                canvas, (x0, 0), (x0 + target_w - 1, target_h - 1), (220, 220, 220), -1
            )
            return
        if isinstance(frame, Image.Image):
            img = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        else:
            img = frame.copy()
        h, w = img.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        dx = (target_w - new_w) // 2
        dy = (target_h - new_h) // 2
        canvas[dy : dy + new_h, x0 + dx : x0 + dx + new_w] = resized
        cv2.rectangle(
            canvas, (x0, 0), (x0 + target_w - 1, target_h - 1), (200, 200, 200), 2
        )

    player_frame = _pick(player_frame_list, frame_idx)
    standard_frame = _pick(standard_frame_list, frame_idx)

    _place(player_frame, 0)
    _place(standard_frame, target_w)

    # 标注文本（用 PIL 避免中文乱码）
    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text((20, 20), "玩家动作", font=_get_font(24), fill=(0, 0, 0))
    draw.text((target_w + 20, 20), "标准动作", font=_get_font(24), fill=(0, 0, 0))
    canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return canvas


def PlotSkeleton(skeleton_data):
    """
    绘制骨骼。
    按照输入的骨骼数据绘制人体骨骼，并添加投影阴影提升立体感。
    """
    canvas = np.full((REGION_H, REGION_W, 3), 255, dtype=np.uint8)
    if skeleton_data is None:
        pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        # ImageDraw.Draw(pil_img).text((20, 40), "未检测到骨骼", font=_get_font(28), fill=(255, 0, 0))
        # ImageDraw.Draw(pil_img).text((20, 40), "", font=_get_font(28), fill=(255, 0, 0))
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    kpts = np.array(skeleton_data, dtype=np.float32)
    if kpts.ndim != 2 or kpts.shape[0] < 17:
        pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        # ImageDraw.Draw(pil_img).text((20, 40), "关键点格式错误", font=_get_font(28), fill=(255, 0, 0))
        ImageDraw.Draw(pil_img).text((20, 40), "", font=_get_font(28), fill=(255, 0, 0))
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 归一化到区域
    x_min, y_min = kpts[:, 0].min(), kpts[:, 1].min()
    x_max, y_max = kpts[:, 0].max(), kpts[:, 1].max()
    w = max(1e-3, x_max - x_min)
    h = max(1e-3, y_max - y_min)
    scale = min((REGION_W * 0.8) / w, (REGION_H * 0.8) / h)
    kpts_scaled = (kpts - np.array([x_min, y_min])) * scale
    kpts_scaled += np.array([REGION_W * 0.1, REGION_H * 0.1])

    edges = [
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 6),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (12, 14),
        (13, 15),
        (14, 16),
        (5, 1),
        (6, 2),
        (1, 2),
        (1, 0),
        (2, 0),
        (0, 3),
        (0, 4),
        (3, 5),
        (4, 6),
    ]

    # ---------- 阴影生成 ----------
    body_mask = np.zeros((REGION_H, REGION_W), dtype=np.uint8)
    # 实心肢体与关节点
    for a, b in edges:
        pa = tuple(np.int32(kpts_scaled[a]))
        pb = tuple(np.int32(kpts_scaled[b]))
        limb_len = max(1.0, np.hypot(pb[0] - pa[0], pb[1] - pa[1]))
        thickness = int(max(14, limb_len * 0.35))
        cv2.line(body_mask, pa, pb, 255, thickness, lineType=cv2.LINE_AA)
    for pt in kpts_scaled:
        cv2.circle(
            body_mask,
            tuple(np.int32(pt)),
            int(max(14, w * scale * 0.03)),
            255,
            -1,
            lineType=cv2.LINE_AA,
        )

    # 膨胀/闭合填充
    ksize = max(5, int(min(REGION_W, REGION_H) * 0.035) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    body_mask = cv2.dilate(body_mask, kernel, iterations=1)

    # 仿射压扁错位（锚点脚踝）
    ankle_idx = [15, 16]
    ank_pts = [kpts_scaled[i] for i in ankle_idx if i < len(kpts_scaled)]
    if len(ank_pts):
        base_y = int(max([p[1] for p in ank_pts]))
        base_x = int(np.mean([p[0] for p in ank_pts]))
    else:
        base_y = int(REGION_H * 0.8)
        base_x = int(REGION_W * 0.5)
    scale_y = 0.42
    shear = -0.35
    small_offset = int(max(4, h * scale * 0.06))
    M = np.array(
        [
            [1.0, shear, -shear * base_y],
            [0.0, scale_y, base_y * (1.0 - scale_y) + small_offset],
        ],
        dtype=np.float32,
    )
    warped = cv2.warpAffine(
        body_mask,
        M,
        (REGION_W, REGION_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # 模糊并 alpha 混合
    blur_k = int(max(21, (int(w * scale * 0.06) | 1)))
    if blur_k % 2 == 0:
        blur_k += 1
    warped_blur = cv2.GaussianBlur(warped, (blur_k, blur_k), 0)
    shadow_opacity = 0.18
    alpha_mask = (warped_blur.astype(np.float32) / 255.0) * shadow_opacity
    alpha_mask = np.expand_dims(alpha_mask, 2)
    shadow_color = np.array([25, 25, 25], dtype=np.float32)
    canvas_f = canvas.astype(np.float32)
    canvas_f = canvas_f * (1.0 - alpha_mask) + shadow_color * alpha_mask
    canvas = np.clip(canvas_f, 0, 255).astype(np.uint8)

    # ---------- 绘制骨骼线与点 ----------
    for a, b in edges:
        pa = tuple(np.int32(kpts_scaled[a]))
        pb = tuple(np.int32(kpts_scaled[b]))
        cv2.line(canvas, pa, pb, (200, 50, 50), 2, lineType=cv2.LINE_AA)
    for pt in kpts_scaled:
        cv2.circle(
            canvas, tuple(np.int32(pt)), 4, (50, 200, 50), -1, lineType=cv2.LINE_AA
        )

    return canvas


def PlotProblemsSuggest(problems_text: str, suggest_text: str):
    """
    按照问题提示的格式要求，在窗口的右下角区域先绘制问题文本，字体为红色。换行后绘制建议文本，字体为绿色。
    支持根据区域宽度自动换行（针对中文按字符拆分）。
    """
    canvas = np.full((REGION_H, REGION_W, 3), 255, dtype=np.uint8)
    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    left_x = 20
    right_padding = 20
    max_w = REGION_W - left_x - right_padding

    def wrap_text(text: str, font):
        lines = []
        for para in text.splitlines():
            if para == "":
                lines.append("")
                continue
            cur = ""
            for ch in para:
                test = cur + ch
                try:
                    bbox = draw.textbbox((0, 0), test, font=font)
                    w = bbox[2] - bbox[0]
                except Exception:
                    w = font.getsize(test)[0]
                if w <= max_w:
                    cur = test
                else:
                    if cur == "":
                        # single char too wide (rare), still append
                        lines.append(test)
                        cur = ""
                    else:
                        lines.append(cur)
                        cur = ch
            if cur:
                lines.append(cur)
        return lines

    # problems (red)
    prob_font = _get_font(18)
    prob_lines = wrap_text(problems_text or "", prob_font)
    y = 40
    line_h = None
    try:
        # prefer font metrics
        ascent, descent = prob_font.getmetrics()
        line_h = ascent + descent + 6
    except Exception:
        line_h = 28

    for line in prob_lines:
        draw.text((left_x, y), line, font=prob_font, fill=(255, 0, 0))
        y += line_h

    # gap before suggestions
    y += 12

    # suggestions (green)
    sug_font = _get_font(18)
    sug_lines = wrap_text(suggest_text or "", sug_font)
    for line in sug_lines:
        draw.text((left_x, y), line, font=sug_font, fill=(0, 180, 0))
        y += line_h

    canvas = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return canvas


def _find_common_length(len_a: int, len_b: int, max_scale: int):
    """
    Try to find a common length L where L = len_a * s_a = len_b * s_b and 1 <= s_a,s_b <= max_scale.
    If no exact common multiple exists within the scale bound, fall back to the closest pair (abs diff minimal)
    while keeping both scales <= max_scale. Returns (target_len, scale_a, scale_b).
    """
    best = None  # (diff, target_len, s_a, s_b)

    # First try exact common multiples inside the bound (prefer the smallest L)
    for s_a in range(1, max_scale + 1):
        L = len_a * s_a
        if L % len_b != 0:
            continue
        s_b = L // len_b
        if s_b <= max_scale:
            candidate = (0, L, s_a, s_b)
            if best is None or candidate[1] < best[1]:
                best = candidate

    # If no exact solution, search for the closest lengths reachable within the scale cap
    if best is None:
        for s_a in range(1, max_scale + 1):
            target_a = len_a * s_a
            for s_b in range(1, max_scale + 1):
                target_b = len_b * s_b
                diff = abs(target_a - target_b)
                target_len = max(target_a, target_b)
                candidate = (diff, target_len, s_a, s_b)
                if best is None:
                    best = candidate
                    continue
                # Prefer smaller diff, then smaller target length
                if diff < best[0] or (diff == best[0] and target_len < best[1]):
                    best = candidate

    _, target_len, s_a, s_b = best
    return target_len, s_a, s_b


def _stretch_frames_evenly(frames, target_len: int):
    """
    Evenly replicate frames so the sequence reaches target_len.
    Each source frame is repeated either floor or ceil times to distribute the stretch smoothly.
    """
    if not frames or target_len <= 0:
        return []
    n = len(frames)
    if n == target_len:
        return list(frames)

    ratio = target_len / n
    stretched = []
    # Use cumulative rounding to keep the distribution even
    for i in range(n):
        start = int(round(i * ratio))
        end = int(round((i + 1) * ratio))
        repeat = max(1, end - start)
        stretched.extend([frames[i]] * repeat)

    # Guard against off-by-one due to rounding
    if len(stretched) < target_len:
        stretched.extend([frames[-1]] * (target_len - len(stretched)))
    elif len(stretched) > target_len:
        stretched = stretched[:target_len]
    return stretched


def ExtractSkeleton(frames):
    """
    1. 读取视频，逐帧处理。
    2. 对每一帧使用person_model检测人体，获取人体边界框。
    3. 将边界框内的图像裁剪出来，输入到model中进行骨骼估计，获取骨骼坐标。
    4. 将骨骼坐标保存到一个列表中，最终返回整个视频的骨骼数据列表。
    """

    skeleton_data = []
    person_frame_list = []
    for frame in frames:
        kpts, boxes = get_keypoint(frame)
        if kpts is None:
            print("未检测到骨骼，跳过该帧")
            continue
        skeleton_data.append(kpts)
        person_frame_list.append(
            ExtractPersonFrame(frame, boxes[0])
            if boxes is not None and len(boxes) > 0
            else None
        )
    return skeleton_data, person_frame_list


def ExtractPersonFrame(frame, person_boxe):
    """
    根据人物边界框裁剪出人物图像。
    """
    x1, y1, w, h = person_boxe.astype(int)
    x2, y2 = x1 + w, y1 + h
    person_frame = frame.crop((x1, y1, x2, y2))
    return person_frame


def get_keypoint(frame):
    """
    frame: PIL.Image
    last_position: 上一帧球员的关节点位置或者球的位置 x, y
    """
    # 预测
    # 先找人物框
    inputs = person_image_processor(images=frame, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = person_model(**inputs)
    # ensure width/height read from image
    np_frame = np.array(frame)
    height, width = np_frame.shape[0], np_frame.shape[1]
    results = person_image_processor.post_process_object_detection(
        outputs, target_sizes=torch.tensor([(height, width)]), threshold=0.3
    )
    result = results[0]
    person_boxes = result["boxes"][result["labels"] == 0]
    person_boxes = person_boxes.cpu().numpy()
    if len(person_boxes) == 0:
        print("no keypoints detected, skip this frame")
        return None, None
    # 根据人物框提取骨骼
    # Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) format
    person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
    person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]
    inputs = image_processor(frame, boxes=[person_boxes], return_tensors="pt").to(
        device
    )
    inputs["dataset_index"] = torch.tensor([0], device=device)
    with torch.no_grad():
        outputs = model(**inputs)
    # print(outputs)
    pose_results = image_processor.post_process_pose_estimation(
        outputs, boxes=[person_boxes], threshold=0.3
    )
    keypoint_result = (
        pose_results[0][0]["keypoints"].cpu().numpy()
    )  # results for first image
    if keypoint_result.shape[0] < 17:
        print("keypoints detected but less than 17, skip this frame")
        return None, person_boxes
    # print("keypoints shape:", keypoint_result.shape)  # should be (17, 2)
    return keypoint_result, person_boxes


def PlotAll(original_video_path, standard_video_path, standard_pose):
    """
    1. 读取原视频，提取骨骼，计算相似度和分数。
    2. 绘制一个1920*1080大小的视频上，画面背景为白色，将画面均匀分割为6个区域，分别显示原视频（左上角）、六边形（左下）、分数（中上）、相似度（中下）、人体骨骼（右上）和问题提示（右下）。
    所有内容自适应缩放大小。
    """
    global _last_similarity

    cap = cv2.VideoCapture(original_video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {original_video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()

    # 计算骨骼
    skeleton_list, person_frame_list = ExtractSkeleton(frames)
    # 过滤掉关键点数量不足17个的帧
    keypoints = keypoints = [kp for kp in skeleton_list if len(kp) >= 17]
    print(f"总帧数: {len(frames)}, 有效骨骼帧数: {len(keypoints)}")
    # 动作分类
    numpy_data = np.array(keypoints)
    numpy_data = numpy_data[
        int(len(numpy_data) / 2 - 15) : int(len(numpy_data) / 2 + 15)
    ]
    # print(f"用于分类的骨骼数据形状: {numpy_data.shape}")
    # start_time = time.time()

    # action_cls = gt.get_cls(numpy_data)
    # if action_cls == 0:
    #     print("动作分类结果: 正手")
    #     standard_video_path = (
    #         "./standar_video/Federer/forehand_left/standard_1_action_008.mp4"
    #     )
    #     standard_pose = FEDERER_STANDARD["forehand_left"]
    # elif action_cls == 1:
    #     print("动作分类结果: 反手")
    #     standard_video_path = (
    #         "./standar_video/Federer/backhand_left/standard_1_action_008.mp4"
    #     )
    #     standard_pose = FEDERER_STANDARD["backhand_left"]
    # else:
    #     print("动作分类结果: 未知")
    # # print(f"动作分类结果: {action_cls}")
    # print(f"动作分类耗时: {time.time() - start_time:.2f} 秒")
    standard_pose = np.load(standard_pose)
    # 分析动作
    sixe_scores, three_stage_score, problems, suggests = pa.AnalysePose(
        skeleton_list, standard_pose, original_video_path
    )

    standard_frame_list = []
    cap = cv2.VideoCapture(standard_video_path)
    if not cap.isOpened():
        print(f"Cannot open standard video: {standard_video_path}")
        return
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        standard_frame_list.append(Image.fromarray(frame_rgb))
    cap.release()

    # 对齐玩家与标准帧序列长度：允许最多 slow_factor 倍的均匀复制
    slow_factor = 3  # 最大扩张倍数
    player_frame_num = len(person_frame_list)
    standard_frame_num = len(standard_frame_list)
    print(f"原视频帧数: {player_frame_num}, 标准视频帧数: {standard_frame_num}")
    print(f"尝试对齐帧数，允许最大扩张倍数: {slow_factor}x")
    target_len, player_scale, standard_scale = _find_common_length(
        player_frame_num, standard_frame_num, slow_factor
    )
    print(f"目标帧数: {target_len}")
    # 将相关序列全部扩张到相同长度，保证内容完整且等长
    player_frames_aligned = _stretch_frames_evenly(person_frame_list, target_len)
    player_skeleton_aligned = _stretch_frames_evenly(skeleton_list, target_len)
    player_full_frames_aligned = _stretch_frames_evenly(frames, target_len)
    standard_frames_aligned = _stretch_frames_evenly(standard_frame_list, target_len)

    if target_len == 0:
        print("No frames to process after alignment.")
        return

    out = cv2.VideoWriter(
        f"{original_video_path.split('.')[0]}_output.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (CANVAS_W, CANVAS_H),
    )
    pbar = tqdm(range(target_len), total=target_len, desc="绘制视频", unit="帧")
    for idx in pbar:
        skeleton = (
            player_skeleton_aligned[idx] if idx < len(player_skeleton_aligned) else None
        )
        canvas = np.full((CANVAS_H, CANVAS_W, 3), 255, dtype=np.uint8)

        # 左上：原视频（原速播放，播完后静止最后一帧）
        orig_idx = min(idx + 1, len(frames))  # 至多取到最后一帧
        region_orig = PlotOriginalVideo(frames[:orig_idx])
        canvas[0:REGION_H, 0:REGION_W] = region_orig

        # 中上：分数（此处用占位示例，可替换为真实得分数组 shape(5,)）
        scores = np.array(three_stage_score, dtype=np.float32)
        region_score = PlotScore(scores)
        canvas[0:REGION_H, REGION_W : REGION_W * 2] = region_score

        # 右上：骨骼
        region_skeleton = PlotSkeleton(skeleton)
        canvas[0:REGION_H, REGION_W * 2 : CANVAS_W] = region_skeleton

        # 左下：六边形（占位示例，可替换为真实6维能力值）
        hex_vals = np.array(sixe_scores, dtype=np.float32)
        region_hex = PlotHexagon(hex_vals)
        canvas[REGION_H:CANVAS_H, 0:REGION_W] = region_hex

        # 中下：相似度
        sim_idx = idx
        region_sim = PlotSimilarity(
            player_frames_aligned, standard_frames_aligned, sim_idx
        )
        canvas[REGION_H:CANVAS_H, REGION_W : REGION_W * 2] = region_sim

        # 右下：问题与建议
        # problems_text = "肘部问题:肘部角度偏屈，击球时未充分伸展，影响击球力量与稳定性\n手腕问题:手腕过度屈曲，导致击球时控制力不足\n姿态稳定性:身体重心不稳，影响动作连贯性\n躯干问题:躯干旋转不足，限制了击球范围\n膝部问题:膝部未充分弯曲，降低了动作的弹性和力量"
        # suggest_text = "射部建议:在击球前适度伸展时部，保持自然发力路径\n躯干建议:调整躯干姿态，保持中立或适度后仰，增强击球稳定性\n膝部建议:在击球前活度伸展膝部，保持稳定支撑与发力路径\n击球点建议:调整击球点位置，使其更靠近身体前方或正前方，以提高击球精度与力量传导"

        region_tip = PlotProblemsSuggest(problems, suggests)
        canvas[REGION_H:CANVAS_H, REGION_W * 2 : CANVAS_W] = region_tip

        out.write(canvas)

    out.release()
    print("Finished. Saved to output.mp4")
    return scores[0], problems, suggests


if __name__ == "__main__":
    # original_video_path = "test.mp4"
    # standard_video_path = "forehand.mp4"
    # standard_video_path = "backhand.mp4"
    # PlotAll(original_video_path, standard_video_path)

    # det = td.detector.TennisActionDetector()

    # results = det.process_video(video_path="video/standard_1.mp4", output_dir="yolo_test")

    # # 建立标准数据库，保存为.npy文件
    # root_dirs = ["video/Federer", "video/Tai"]
    # sub_dirs = ["forehand_right", "forehand_left", "backhand_right", "backhand_left"]
    # for root_dir in root_dirs:
    #     for sub_dir in sub_dirs:
    #         full_dir = os.path.join(root_dir, sub_dir)
    #         angles_list = []
    #         for video_name in os.listdir(full_dir):
    #             video_path = os.path.join(full_dir, video_name)
    #             frame_list = []
    #             cap = cv2.VideoCapture(video_path)
    #             while True:
    #                 ret, frame_bgr = cap.read()
    #                 if not ret:
    #                     break
    #                 frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    #                 frame_list.append(Image.fromarray(frame_rgb))
    #             keypoint_list, _ = ExtractSkeleton(frame_list)
    #             keypoint_list = pa.sample_keypoints(keypoint_list, sample=15)
    #             angles = pa.get_angle_groups(keypoint_list)
    #             angles = np.array(angles, dtype=np.float32)
    #             print(angles.shape)
    #             angles_list.append(angles)
    #         angles_array = np.array(angles_list)
    #         print(f"angles_array shape for {full_dir}:", angles_array.shape)
    #         angles_array = np.mean(angles_array, axis=0)
    #         print(f"Mean angles_array shape for {full_dir}:", angles_array.shape)
    #         npy_file = f"{root_dir.split('/')[-1]}_{sub_dir}_angles.npy"
    #         np.save(npy_file, angles_array)
    #         print(f"Saved standard angles to {npy_file}")
    # print("标准数据库建立完毕，开始处理视频")

    federer_standard = {
        "forehand_right": np.load("Federer_forehand_right_angles.npy"),
        "forehand_left": np.load("Federer_forehand_left_angles.npy"),
        "backhand_right": np.load("Federer_backhand_right_angles.npy"),
        "backhand_left": np.load("Federer_backhand_left_angles.npy"),
    }
    tai_standard = {
        "forehand_right": np.load("Tai_forehand_right_angles.npy"),
        "forehand_left": np.load("Tai_forehand_left_angles.npy"),
        "backhand_right": np.load("Tai_backhand_right_angles.npy"),
        "backhand_left": np.load("Tai_backhand_left_angles.npy"),
    }
    gt.load_models()
    # results = det.process_video(video_path="video/tai.mp4", output_dir="yolo_test")
    # print("提取击球帧")
    start_time = time.time()
    # results = det.process_video(video_path="video/zhou.mp4", output_dir="yolo_test")
    # results = det.process_video(video_path="video/tai.mp4", output_dir="yolo_test")
    # print(f"提取完毕，用时{time.time() - start_time}秒")
    angle_results = []

    # for action in results:
    #     print("action_id:", action.get("action_id"))
    #     print("hit_frame:", action["hit_frame"])
    #     print("hit_timestamp:", action["hit_timestamp"])
    #     print("confidence:", action.get("confidence"))
    #     print("clip_path:", action.get("clip_path"))
    # angle_ = get_keypoints(action["clip_path"], n_samples=15)
    # angle_results.append(angle_)

    # process_video("test.mp4", "output_video.mp4")

    # process_video(results[2]["clip_path"], "yolo_test/tai_vit_pose_test.mp4")

    # print(federer_standard["forehand_left"].shape)
    # for i, action in enumerate(results):
    #     # process_video(action["clip_path"], f"yolo_test/zhou_{i}_vit_pose_test.mp4")
    #     # process_video(action["clip_path"], f"yolo_test/tai_{i}_vit_pose_test.mp4")
    #     PlotAll(
    #         action["clip_path"],
    #         "./standar_video/Federer/forehand_left/standard_1_action_008.mp4",
    #         federer_standard["forehand_left"],
    #     )
    #     break

    # pass
