import os
import tempfile
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, flash
from faster_whisper import WhisperModel
from google import genai

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")

# base = สมดุลระหว่างความแม่นยำกับการใช้ RAM (เหมาะกับ Railway free tier)
# ถ้าเจอปัญหาหน่วยความจำไม่พอ ให้เปลี่ยน env var WHISPER_MODEL_SIZE เป็น "tiny"
MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MAX_CONTENT_LENGTH_MB = int(os.environ.get("MAX_UPLOAD_MB", "200"))

app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_MB * 1024 * 1024

_whisper_model = None


def get_whisper_model():
    """โหลดโมเดลครั้งเดียวแล้วเก็บไว้ใช้ซ้ำ (โหลดครั้งแรกจะช้าเพราะต้องดาวน์โหลดโมเดล)"""
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    return _whisper_model


def transcribe_audio(audio_path: str) -> str:
    model = get_whisper_model()
    segments, info = model.transcribe(audio_path, beam_size=5)
    lines = [f"[{seg.start:6.1f}s] {seg.text.strip()}" for seg in segments]
    return "\n".join(lines)


def analyze_transcript(transcript: str) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""คุณเป็นโค้ชการประชุมมืออาชีพ นี่คือบทถอดเสียงการประชุม
(อาจมีทั้งภาษาไทยและอังกฤษปนกัน):

---
{transcript}
---

กรุณาวิเคราะห์และตอบกลับเป็นภาษาไทยในรูปแบบนี้:

## สรุปประเด็นสำคัญ
(สรุปสั้นๆ 3-5 ข้อ ว่าประชุมคุยเรื่องอะไร ตกลงอะไรกันบ้าง)

## Action Items
(สิ่งที่ต้องทำต่อ ถ้ามีการพูดถึง ถ้าไม่มีให้บอกว่าไม่มี)

## คะแนนการพูด (1-10)
- ความชัดเจน (Clarity):
- จังหวะการพูด (Pacing):
- คำฟุ่มเฟือย/Filler words (เช่น เอ่อ, อ่า, um):
- การรับฟังผู้อื่น (Listening):

## ข้อเสนอแนะเพื่อปรับปรุง
(2-3 ข้อที่ทำได้จริงในครั้งหน้า)
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    return response.text


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if not GEMINI_API_KEY:
        flash("เซิร์ฟเวอร์ยังไม่ได้ตั้งค่า GEMINI_API_KEY (ดูใน Railway > Variables)")
        return redirect(url_for("index"))

    audio_file = request.files.get("audio_file")
    if not audio_file or audio_file.filename == "":
        flash("กรุณาเลือกไฟล์เสียงหรือวิดีโอก่อน")
        return redirect(url_for("index"))

    suffix = Path(audio_file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        transcript = transcribe_audio(tmp_path)
        report = analyze_transcript(transcript)
    except Exception as exc:  # แสดง error ให้ผู้ใช้เห็นแทนที่จะขึ้นหน้า 500 เฉยๆ
        flash(f"เกิดข้อผิดพลาดระหว่างประมวลผล: {exc}")
        return redirect(url_for("index"))
    finally:
        os.remove(tmp_path)

    return render_template("report.html", report=report, transcript=transcript)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 6000))
    app.run(host="0.0.0.0", port=port)