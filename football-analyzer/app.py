import os
import time
import json
import logging
from flask import Flask, render_template, request, jsonify
from pydantic import BaseModel, Field
from typing import List # 引入 List
from google import genai

app = Flask(__name__)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEFAULT_API_KEY = os.environ.get("GEMINI_API_KEY")

# ========== 1. Pydantic 模型升级：恢复深度分析字段 ==========

class SideAnalysis(BaseModel):
    summary: str = Field(description="一句话战术总结")
    play_type: str = Field(description="战术类型 (e.g., Inside Zone, Screen Pass, Cover 3)")
    formation: str = Field(description="阵型名称 (e.g., Shotgun Trips Right, I-Form, Nickel 4-2-5)")
    personnel: str = Field(description="人员配置估计 (e.g., 11 Personnel, 12 Personnel, 5 DBs)")
    key_players: List[str] = Field(description="关键球员号码或位置列表 (e.g., ['#15 QB', '#87 TE'])")
    details: str = Field(description="详细战术分析，支持 Markdown 格式 (列表、加粗等)")

class FullPlayAnalysis(BaseModel):
    offense: SideAnalysis = Field(description="进攻方分析")
    defense: SideAnalysis = Field(description="防守方分析")

# ========== Gemini 相关函数 ==========

def upload_and_process_video(local_path: str, client: genai.Client):
    """上传视频并轮询直到处理完成"""
    logger.info(f"Uploading video: {local_path}")
    video_file = client.files.upload(file=local_path)
    
    while video_file.state.name == "PROCESSING":
        time.sleep(2)
        video_file = client.files.get(name=video_file.name)

    if video_file.state.name != "ACTIVE":
        raise RuntimeError(f"Video processing failed: {video_file.state.name}")

    logger.info("Video processed successfully.")
    return video_file

def call_gemini_with_video(video_path: str, api_key: str) -> dict:
    video_file = None
    client = None
    try:
        client = genai.Client(api_key=api_key)
        video_file = upload_and_process_video(video_path, client)

        # Prompt 升级：要求所有新增的字段
        prompt = (
            "You are an expert American Football coach analyzing game film. "
            "Analyze the attached video clip for BOTH Offense and Defense.\n\n"
            "Identify the Formation, Personnel package, Key Players involved, "
            "and the specific Play Concept.\n"
            "For 'details', provide a breakdown of the action, reading keys, and outcome using "
            "Markdown formatting (bullet points, bold text) for readability."
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[video_file, prompt],
            config={
                "response_mime_type": "application/json",
                "response_schema": FullPlayAnalysis,
            },
        )

        if getattr(response, "parsed", None):
            return response.parsed.model_dump()
        else:
            return json.loads(response.text)

    except Exception as e:
        logger.error(f"Gemini API Error: {e}")
        return {"error": str(e)}
    
    finally:
        # 恢复文件清理逻辑
        if client and video_file:
            try:
                client.files.delete(name=video_file.name)
            except:
                pass
        
        if os.path.exists(video_path):
            os.remove(video_path)
            logger.info(f"Deleted local file: {video_path}")

# ========== Flask 路由 ==========

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze_video():
    api_key = request.form.get("api_key") or DEFAULT_API_KEY
    if not api_key:
        return jsonify({"error": "Missing API key"}), 400

    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    timestamp = int(time.time())
    safe_name = file.filename.replace(" ", "_")
    filename = f"vid_{timestamp}_{safe_name}"
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    result = call_gemini_with_video(save_path, api_key)

    if "error" in result:
        return jsonify(result), 500

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)