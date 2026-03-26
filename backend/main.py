"""
Sport Vision — FastAPI 应用入口
提供 REST API + WebSocket 实时分析流
"""

import os
import json
import uuid
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.pipeline import Pipeline
import backend.plot_ui_v3 as puv3


# 路径配置
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "uploads"
DEMO_DIR = BASE_DIR / "demo_videos"
CLIPS_DIR = BASE_DIR / "clips"
VIDEO_DIR = BASE_DIR / "video"

UPLOAD_DIR.mkdir(exist_ok=True)
DEMO_DIR.mkdir(exist_ok=True)
CLIPS_DIR.mkdir(exist_ok=True)
VIDEO_DIR.mkdir(exist_ok=True)

# FastAPI 应用
app = FastAPI(title="Sport Vision", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 活跃的处理流水线
active_pipelines: dict[str, Pipeline] = {}


# ============ REST API ============


@app.get("/api/demos")
async def list_demos():
    """列出所有可用的 Demo 视频"""
    demos = []
    for ext in ["*.mp4", "*.avi", "*.mov", "*.webm"]:
        for f in DEMO_DIR.glob(ext):
            demos.append(
                {
                    "id": f.stem,
                    "name": f.stem.replace("_", " ").replace("-", " ").title(),
                    "filename": f.name,
                    "size_mb": round(f.stat().st_size / (1024 * 1024), 1),
                }
            )
    return {"demos": demos}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """上传视频文件"""
    allowed = {".mp4", ".avi", ".mov", ".webm", ".mkv"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported format: {ext}. Allowed: {allowed}"},
        )

    # 保存文件
    file_id = str(uuid.uuid4())[:8]
    save_path = UPLOAD_DIR / f"{file_id}{ext}"
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    return {
        "id": file_id,
        "filename": file.filename,
        "path": str(save_path),
        "size_mb": round(len(content) / (1024 * 1024), 1),
    }


# ============ 分析报告 API ============


@app.post("/api/report")
async def generate_report(payload: dict):
    """
    接收前端传来的切片视频地址，返回分析报告（当前为示例版）
    """

    video_url = payload.get("video_url")
    clip_id = payload.get("clip_id")
    action_name = payload.get("action_name")

    if not video_url:
        return JSONResponse(status_code=400, content={"error": "video_url is required"})
    print(BASE_DIR)
    video_url = str(BASE_DIR) + video_url
    print(video_url)
    if "Forehand" in action_name:
        hex_vals, problems, suggests = puv3.PlotAll(
            video_url,
            "./standar_video/Federer/forehand_left/federer (1).mp4",
            "./backend/Federer_forehand_left_angles.npy",
        )
    else:
        hex_vals, problems, suggests = puv3.PlotAll(
            video_url,
            "./standar_video/Federer/backhand_left/federer (1).mp4",
            "./backend/Federer_backhand_left_angles.npy",
        )
    hex_vals = hex_vals.tolist()
    # ===== 这里可以替换成你自己的分析逻辑 =====
    # 比如：调用模型 / 读取视频 / 分析动作

    # report = {
    #     "clip_id": clip_id,
    #     "video_url": video_url,
    #     "action": action_name,
    #     "analysis": {
    #         "动作类型": action_name,
    #         "稳定性评分": 0.87,
    #         "节奏评价": "动作节奏良好",
    #         "建议": ["注意击球点提前", "保持身体重心稳定", "挥拍轨迹可以更流畅"],
    #     },
    # }
    report = {
        "action": action_name,
        "score": hex_vals,
        "问题": problems,
        "建议": suggests,
    }

    return {"ok": True, "report": report}


# ============ WebSocket ============


@app.websocket("/ws/analyze")
async def websocket_analyze(websocket: WebSocket):
    """
    WebSocket 实时分析端点

    客户端发送:
        {"type": "start", "source": "demo", "id": "badminton_rally"}
        {"type": "start", "source": "upload", "path": "/path/to/video"}
        {"type": "start", "source": "camera"}
        {"type": "stop"}

    服务端推送:
        {"type": "frame", "data": {...}}
        {"type": "complete", "summary": {...}}
        {"type": "error", "message": "..."}
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]
    pipeline: Optional[Pipeline] = None

    try:
        while True:
            # 接收客户端消息
            msg = await websocket.receive_text()
            data = json.loads(msg)

            if data.get("type") == "start":
                # 停止之前的流水线
                if pipeline:
                    pipeline.stop()

                # 确定视频路径
                video_path = None
                if data.get("source") == "demo":
                    demo_id = data.get("id", "")
                    for ext in [".mp4", ".avi", ".mov", ".webm"]:
                        candidate = DEMO_DIR / f"{demo_id}{ext}"
                        if candidate.exists():
                            video_path = str(candidate)
                            break
                elif data.get("source") == "upload":
                    video_path = data.get("path", "")

                if data.get("source") != "camera" and (
                    not video_path or not Path(video_path).exists()
                ):
                    await websocket.send_json(
                        {"type": "error", "message": f"Video not found: {video_path}"}
                    )
                    continue

                # 创建新的 pipeline 并开始处理
                pipeline = Pipeline()
                active_pipelines[session_id] = pipeline

                try:
                    if data.get("source") == "camera":
                        await websocket.send_json(
                            {
                                "type": "started",
                                "session_id": session_id,
                                "source": "camera",
                            }
                        )

                        async for result in pipeline.process_camera(target_fps=20):
                            if "error" in result:
                                await websocket.send_json(
                                    {"type": "error", "message": result["error"]}
                                )
                                break
                            await websocket.send_json({"type": "frame", "data": result})
                    else:
                        await websocket.send_json(
                            {
                                "type": "started",
                                "session_id": session_id,
                                "video": video_path,
                                "source": data.get("source"),
                            }
                        )

                        async for result in pipeline.process_video(
                            video_path,
                            target_fps=20,
                            skip_frames=1,
                        ):
                            if "error" in result:
                                await websocket.send_json(
                                    {"type": "error", "message": result["error"]}
                                )
                                break
                            await websocket.send_json({"type": "frame", "data": result})

                    await websocket.send_json(
                        {"type": "complete", "session_id": session_id}
                    )

                except Exception as e:
                    await websocket.send_json({"type": "error", "message": str(e)})

            elif data.get("type") == "stop":
                if pipeline:
                    pipeline.stop()
                    await websocket.send_json(
                        {
                            "type": "stopped",
                            "session_id": session_id,
                        }
                    )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if pipeline:
            pipeline.close()
        active_pipelines.pop(session_id, None)


# ============ 静态文件 ============

# 前端静态文件
app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")


# # Demo 原视频访问
# @app.get("/demo_videos/{filename}")
# async def serve_demo_video(filename: str):
#     path = DEMO_DIR / filename
#     if path.exists():
#         return FileResponse(path)
#     return JSONResponse(status_code=404, content={"error": "not found"})


# # 相机模式切片视频访问
# @app.get("/video/{filename}")
# async def serve_action_video(filename: str):
#     path = VIDEO_DIR / filename
#     if path.exists():
#         return FileResponse(path)
#     return JSONResponse(status_code=404, content={"error": "not found"})


# # demo / upload 自动切片视频访问
# @app.get("/clips/{filename}")
# async def serve_clip_video(filename: str):
#     path = CLIPS_DIR / filename
#     if path.exists():
#         return FileResponse(path)
#     return JSONResponse(status_code=404, content={"error": "not found"})
@app.api_route("/demo_videos/{filename}", methods=["GET", "HEAD"])
async def serve_demo_video(filename: str):
    path = DEMO_DIR / filename
    if path.exists():
        return FileResponse(path)
    return JSONResponse(status_code=404, content={"error": "not found"})


@app.api_route("/video/{filename}", methods=["GET", "HEAD"])
async def serve_action_video(filename: str):
    path = VIDEO_DIR / filename
    if path.exists():
        return FileResponse(path)
    return JSONResponse(status_code=404, content={"error": "not found"})


@app.api_route("/clips/{filename}", methods=["GET", "HEAD"])
async def serve_clip_video(filename: str):
    path = CLIPS_DIR / filename
    if path.exists():
        return FileResponse(path)
    return JSONResponse(status_code=404, content={"error": "not found"})


# 首页
@app.get("/")
async def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
