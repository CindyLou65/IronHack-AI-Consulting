import re
from pathlib import Path
from openai import OpenAI
import shutil
import subprocess

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_CHARS_PER_CHUNK = 3500

def normalize_whitespace(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_text(text, max_chars=MAX_CHARS_PER_CHUNK):
    text = normalize_whitespace(text)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    def flush():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    sentence_split = re.compile(r"(?<=[.!?])\s+")
    for p in paragraphs:
        parts = [p] if len(p) <= max_chars else sentence_split.split(p)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            candidate = part if not current else current + "\n\n" + part
            if len(candidate) <= max_chars:
                current = candidate
            else:
                flush()
                if len(part) > max_chars:
                    for i in range(0, len(part), max_chars):
                        piece = part[i:i+max_chars].strip()
                        if piece:
                            chunks.append(piece)
                else:
                    current = part
    flush()
    return chunks

def tts_to_mp3(text, out_path: Path, tts_model="tts-1", tts_voice="alloy"):
    speech = client.audio.speech.create(
        model=tts_model,
        voice=tts_voice,
        input=text,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(speech.read())

def ffmpeg_available():
    return shutil.which("ffmpeg") is not None

def concat_mp3(mp3_files, out_file: Path):
    """
    Concatenate MP3 chunks using ffmpeg if available, otherwise fallback to pydub.
    """
    if ffmpeg_available():
        list_file = out_file.parent / "concat_list.txt"
        with open(list_file, "w", encoding="utf-8") as f:
            for p in mp3_files:
                f.write(f"file '{p.resolve().as_posix()}'\n")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(out_file),
        ], check=True)
    else:
        from pydub import AudioSegment
        combined = AudioSegment.empty()
        for p in mp3_files:
            combined += AudioSegment.from_file(p, format="mp3")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        combined.export(out_file, format="mp3")
