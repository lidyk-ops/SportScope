import os
import time
import json
from flask import Flask, render_template, request, jsonify
from pydantic import BaseModel, Field
from google import genai

app = Flask(__name__)

# 本地保存上传视频的目录
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 可选：环境变量里的默认 key（如果前端没传，就用这个）
DEFAULT_API_KEY = os.environ.get("GEMINI_API_KEY")


# ========== Pydantic：定义进攻 / 防守 双方结构 ==========

class SideAnalysis(BaseModel):
    summary: str = Field(
        description="一句话描述这个单位这次战术在干什么，例如：外侧跑球，Cover 3 下的短传防守等"
    )
    play_type: str = Field(
        description="战术类型，例如: inside run, outside run, quick pass, deep shot, blitz, zone coverage 等"
    )
    details: str = Field(
        description="更详细的战术说明、关键球员、路线概念、防守策略等"
    )


class FullPlayAnalysis(BaseModel):
    offense: SideAnalysis = Field(
        description="进攻方战术分析"
    )
    defense: SideAnalysis = Field(
        description="防守方战术分析"
    )


# ========== Gemini 相关函数 ==========

def upload_and_process_video(local_path: str, client: genai.Client):
    """
    将视频上传到 Gemini File API 并等待处理完成。
    返回上传后的文件对象。
    """
    print(f"正在上传视频: {local_path} ...")
    video_file = client.files.upload(file=local_path)
    print(f"上传成功: {video_file.name}")

    # 等待后端处理完视频
    while video_file.state.name == "PROCESSING":
        print("视频处理中，请稍候...")
        time.sleep(2)
        video_file = client.files.get(name=video_file.name)

    if video_file.state.name != "ACTIVE":
        raise RuntimeError(f"视频处理失败，状态: {video_file.state.name}")

    print("视频处理完毕，准备分析。")
    return video_file


def call_gemini_with_video(video_path: str, api_key: str) -> dict:
    """
    使用给定的 api_key 调用 Gemini，分析视频中
    进攻方和防守方的战术，返回 dict：
    {
      "offense": {...},
      "defense": {...}
    }
    """
    try:
        client = genai.Client(api_key=api_key)

        # 1. 上传并等待视频就绪
        video_file = upload_and_process_video(video_path, client)

        # 2. Prompt：要求分析 OFFENSE + DEFENSE
        prompt = (
            "You are an experienced American football coach. "
            "Watch this clip carefully and analyze BOTH sides:\n"
            "- OFFENSE: formation, play type, key concepts (routes / run scheme), intent.\n"
            "- DEFENSE: front, coverage, blitz / pressure, how they try to stop the play.\n\n"
            "Return a JSON object with TWO fields: 'offense' and 'defense'. "
            "Each field must contain: 'summary', 'play_type', 'details'. "
            "Use clear, concise coaching language."
        )

        # 3. 调用模型（使用免费模型 gemini-2.0-flash）
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                video_file,   # 文件对象
                prompt,       # 文本 prompt
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": FullPlayAnalysis,
            },
        )

        # SDK 会自动解析成 Pydantic 实例
        if getattr(response, "parsed", None):
            return response.parsed.model_dump()
        else:
            # 兜底：如果没有 parsed，就从 text 里解析 JSON
            return json.loads(response.text)

    except Exception as e:
        print(f"Gemini API 调用错误: {e}")
        # 返回一个结构完整的错误，防止前端崩掉
        return {
            "offense": {
                "summary": f"Error: {str(e)}",
                "play_type": "error",
                "details": "N/A",
            },
            "defense": {
                "summary": f"Error: {str(e)}",
                "play_type": "error",
                "details": "N/A",
            },
        }


# ========== Flask 路由 ==========

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze_video():
    # 1. 拿到 API Key（优先用前端传的，没有再用环境变量）
    api_key = request.form.get("api_key") or DEFAULT_API_KEY
    if not api_key:
        return jsonify({"error": "Missing API key"}), 400

    # 2. 检查视频文件
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Empty file name"}), 400

    # 3. 保存到本地
    timestamp = int(time.time())
    safe_name = file.filename.replace(" ", "_")
    filename = f"video_{timestamp}_{safe_name}"
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    # 4. 调用 Gemini 分析
    result = call_gemini_with_video(save_path, api_key)

    # 如果你不想占空间，可以在这里删文件
    # try:
    #     os.remove(save_path)
    # except Exception:
    #     pass

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)