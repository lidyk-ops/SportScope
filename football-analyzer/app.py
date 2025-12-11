import os
import time
import json
import logging
from flask import Flask, render_template, request, jsonify
from pydantic import BaseModel, Field
from typing import List
from google import genai

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEFAULT_API_KEY = os.environ.get("GEMINI_API_KEY")

# ========== Pydantic 模型 (保持不变) ==========

class SideAnalysis(BaseModel):
    summary: str = Field(description="One-sentence tactical summary.")
    play_type: str = Field(description="Play type (e.g., Inside Zone, Screen Pass, Cover 3).")
    formation: str = Field(description="Formation name.")
    personnel: str = Field(description="Estimated personnel package.")
    key_players: List[str] = Field(description="List of key player numbers or positions.")
    coach_feedback: str = Field(description="Specific feedback on execution, style depends on the requested persona.")
    details: str = Field(description="A strict Markdown bulleted list (starting with '*') detailing the chronological events. NO paragraphs.")

class FullPlayAnalysis(BaseModel):
    offense: SideAnalysis = Field(description="Analysis for the Offense.")
    defense: SideAnalysis = Field(description="Analysis for the Defense.")

# ========== Gemini 函数 ==========

def upload_and_process_video(local_path: str, client: genai.Client):
    logger.info(f"Uploading video: {local_path}")
    video_file = client.files.upload(file=local_path)
    while video_file.state.name == "PROCESSING":
        time.sleep(2)
        video_file = client.files.get(name=video_file.name)
    if video_file.state.name != "ACTIVE":
        raise RuntimeError(f"Video processing failed: {video_file.state.name}")
    return video_file

def call_gemini_with_video(video_path: str, api_key: str, persona: str) -> dict:
    video_file = None
    client = None
    try:
        client = genai.Client(api_key=api_key)
        video_file = upload_and_process_video(video_path, client)

        # --- Persona 指令 ---
        persona_instruction = ""
        if persona == "belichick":
            persona_instruction = (
                "For 'coach_feedback', adopt the persona of BILL BELICHICK. "
                "Style: Grumpy, short sentences, brutally honest, obsessed with details. "
                "Focus on mistakes and situational awareness."
            )
        elif persona == "dungy":
            persona_instruction = (
                "For 'coach_feedback', adopt the persona of TONY DUNGY. "
                "Style: Calm, soft-spoken, mentor-like, focused on fundamentals. "
                "Be constructive but firm."
            )
        else: # regular
            persona_instruction = (
                "For 'coach_feedback', provide standard, professional coaching advice."
            )

        # --- 核心修改：强制 details 使用 Bullet Points ---
        prompt = (
            "You are an expert American Football coach analyzing game film. "
            "Analyze the attached video clip for BOTH Offense and Defense.\n\n"
            f"{persona_instruction}\n\n"
            "Identify the Formation, Personnel, Key Players, and Play Concept.\n"
            "CRITICAL FORMATTING RULE: The 'details' field MUST be a Markdown bulleted list. "
            "Use an asterisk (*) for every single point. DO NOT write in paragraphs or blocks of text.\n" # <--- 加强语气
            "Example format:\n"
            "* Pre-snap read: [Observation]\n"
            "* Snap to handoff: [Action]\n"
            "* Key block: [Action]\n"
            "* Result: [Outcome]"
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
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
        if client and video_file:
            try:
                client.files.delete(name=video_file.name)
            except:
                pass
        if os.path.exists(video_path):
            os.remove(video_path)

# ========== Flask 路由 ==========

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze_video():
    api_key = request.form.get("api_key") or DEFAULT_API_KEY
    if not api_key: return jsonify({"error": "Missing API key"}), 400
    if "video" not in request.files: return jsonify({"error": "No video file"}), 400
    
    persona = request.form.get("persona", "regular")

    file = request.files["video"]
    if file.filename == "": return jsonify({"error": "Empty filename"}), 400

    timestamp = int(time.time())
    safe_name = file.filename.replace(" ", "_")
    filename = f"vid_{timestamp}_{safe_name}"
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    result = call_gemini_with_video(save_path, api_key, persona)

    if "error" in result: return jsonify(result), 500
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)