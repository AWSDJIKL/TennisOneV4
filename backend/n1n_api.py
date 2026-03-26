import requests
import base64
import json
import time

# ================= 配置区 =================
with open("config.json", "r") as f:
    config = json.load(f)
API_KEY = config.get("n1n_api_key")
URL = "https://api.n1n.ai/v1beta/models/gemini-3.1-pro-preview:generateContent"
# video_path = "game3.mp4"  # 替换成你的视频文件路径
# ==========================================

# # 1. 读取视频文件并将其转化为 Base64 编码
# with open(video_path, "rb") as video_file:
#     video_base64 = base64.b64encode(video_file.read()).decode("utf-8")

# # 2. 构造 JSON 请求体
# payload = {
#     "contents": [
#         {
#             "role": "user",
#             "parts": [
#                 {"inline_data": {"mime_type": "video/mp4", "data": video_base64}},
#                 {"text": "请用一句话总结这个击球动作中的问题，30字以内。"},
#             ],
#         }
#     ]
# }

# # 3. 设置请求头（通常 Google API 风格需要 x-goog-api-key 或是 Authorization 头部）
# headers = {
#     "Content-Type": "application/json",
#     # 不同的中转商可能要求的 header 不同，常见的有两种：
#     "Authorization": f"Bearer {api_key}",  # 很多兼容 OpenAI 格式的中转面板常用这个
# }

# # 4. 发送 POST 请求
# print("正在发送请求并上传视频...")
# response = requests.post(url, headers=headers, json=payload)
# print("请求已发送，等待响应...")
# start_time = time.time()
# # 5. 处理响应结果
# if response.status_code == 200:
#     result = response.json()
#     # 打印模型回复的文本内容
#     print(f"响应时间: {time.time() - start_time:.2f} 秒")
#     print("生成结果：")
#     print(result["candidates"][0]["content"]["parts"][0]["text"])
# else:
#     print(f"响应时间: {time.time() - start_time:.2f} 秒")
#     print(f"请求失败，状态码: {response.status_code}")
#     print("错误详情:", response.text)


def AnalyseVideo(video_path, problem_part):
    with open(video_path, "rb") as video_file:
        video_base64 = base64.b64encode(video_file.read()).decode("utf-8")

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": "video/mp4", "data": video_base64}},
                    {
                        "text": f"请用一句话总结这个击球动作中的问题，30字以内。并用另外一句话给出{problem_part}的改进建议，30字以内。两句话之间用|分隔开来。"
                    },
                ],
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    print("正在发送请求并上传视频...")
    retry_count = 3
    response = requests.post(URL, headers=headers, json=payload)
    while (
        response.status_code != 200 and retry_count > 0
    ):  # 如果请求过于频繁，等待一段时间后重试
        print(f"请求失败，状态码: {response.status_code}，正在重试...")
        time.sleep(5)  # 等待5秒后重试
        response = requests.post(URL, headers=headers, json=payload)
        retry_count -= 1

    if response.status_code == 200:
        result = response.json()
        print("生成结果：", result["candidates"][0]["content"]["parts"][0]["text"])
        problem_text, suggest_text = result["candidates"][0]["content"]["parts"][0][
            "text"
        ].split("|")
        # return result["candidates"][0]["content"]["parts"][0]["text"], ""
        return problem_text, suggest_text
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print("错误详情:", response.text)
        return "", ""
