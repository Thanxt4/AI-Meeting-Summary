import os
import tempfile
import time
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
        cpu_threads = os.cpu_count() or 2  # ใช้ core ทั้งหมดที่มีอยู่ ไม่ต้องอัปเกรดแพลน
        _whisper_model = WhisperModel(
            MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=cpu_threads
        )
    return _whisper_model


def transcribe_audio(audio_path: str) -> str:
    model = get_whisper_model()
    segments, info = model.transcribe(
        audio_path,
        beam_size=1,        # ลดจาก 5 -> 1: เร็วขึ้นมาก แลกความแม่นยำเล็กน้อย
        vad_filter=True,    # ข้ามช่วงเงียบ/ไม่มีคนพูดไปเลย ไม่เสียเวลาถอดเสียงส่วนที่ไม่มีอะไร
    )
    lines = [f"[{seg.start:6.1f}s] {seg.text.strip()}" for seg in segments]
    return "\n".join(lines)


def analyze_transcript(transcript: str) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""คุณเป็นผู้ช่วยสรุปเนื้อหาการประชุมมืออาชีพ นี่คือบทถอดเสียงการประชุม
(อาจมีทั้งภาษาไทยและอังกฤษปนกัน):

---
{transcript}
---

กรุณาสรุปและตอบกลับเป็นภาษาไทยในรูปแบบนี้ โดยเน้นเนื้อหาที่ประชุมคุยกัน ห้ามใส่ความเห็นหรือให้คะแนนการพูดของผู้เข้าร่วม:

## หัวข้อการประชุม
(สรุปสั้นๆ 1-2 บรรทัดว่าประชุมเรื่องอะไร)

## ประเด็นสำคัญที่พูดคุย
(สรุปเป็นข้อๆ ว่าแต่ละประเด็นคุยอะไรกันบ้าง สรุปให้ครบแต่กระชับ)

## มติ/ข้อตกลง
(สิ่งที่ที่ประชุมตกลงกันได้ ถ้าไม่มีให้บอกว่าไม่มี)

## Action Items
(สิ่งที่ต้องทำต่อ พร้อมผู้รับผิดชอบและกำหนดเวลาถ้ามีการพูดถึง ถ้าไม่มีให้บอกว่าไม่มี)

## ประเด็นที่ยังค้างคาหรือต้องติดตามต่อ
(ถ้ามี)
"""

    max_retries = 4
    delay_seconds = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
            )
            return response.text
        except Exception as exc:
            last_error = exc
            # ถ้าเป็น error ชั่วคราวจากฝั่ง Google (โอเวอร์โหลด) ให้รอแล้วลองใหม่
            if "503" in str(exc) or "UNAVAILABLE" in str(exc) or "overloaded" in str(exc).lower():
                print(f"      Gemini โอเวอร์โหลด (ลองครั้งที่ {attempt + 1}/{max_retries}), รอ {delay_seconds}s...")
                time.sleep(delay_seconds)
                delay_seconds *= 2  # เพิ่มเวลารอเป็นเท่าตัวทุกครั้ง (exponential backoff)
                continue
            raise  # error ประเภทอื่น (เช่น key ผิด) ไม่ต้อง retry ให้โยนออกไปเลย

    raise last_error

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
