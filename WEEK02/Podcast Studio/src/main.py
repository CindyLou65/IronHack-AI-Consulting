import os
import gradio as gr
from pathlib import Path
from data_processor import fetch_articles
from llm_processor import generate_podcast_script
from tts_generator import chunk_text, tts_to_mp3, concat_mp3, MAX_CHARS_PER_CHUNK

OUTPUT_DIR = Path("podcast_output")
CHUNKS_DIR = OUTPUT_DIR / "chunks"
DEFAULT_URLS = [
    "https://www.nationalgeographic.com/history/article/fall-of-ancient-roman-empire",
    "https://www.bbc.co.uk/history/ancient/romans/fallofrome_article_01.shtml",
]

def run_pipeline(urls_text, minutes, tts_model, tts_voice, max_chars):
    urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
    articles = fetch_articles(urls)
    script = generate_podcast_script(articles, minutes)
    chunks = chunk_text(script, max_chars=max_chars)

    mp3_files = []
    for i, chunk in enumerate(chunks, start=1):
        mp3_path = CHUNKS_DIR / f"chunk_{i:02d}.mp3"
        tts_to_mp3(chunk, mp3_path, tts_model, tts_voice)
        mp3_files.append(mp3_path)

    final_audio_path = OUTPUT_DIR / "podcast_episode_final.mp3"
    concat_mp3(mp3_files, final_audio_path)
    return script, final_audio_path

# --- Gradio app ---
with gr.Blocks(title="Fall of Rome Podcast Generator") as demo:
    urls_text = gr.Textbox(label="Article URLs", value="\n".join(DEFAULT_URLS), lines=3)
    minutes = gr.Number(label="Target length (minutes)", value=10)
    tts_model = gr.Textbox(label="TTS model", value="tts-1")
    tts_voice = gr.Dropdown(label="Voice", choices=["alloy", "onyx", "sage"], value="onyx")
    max_chars = gr.Slider(label="Chunk size", minimum=1200, maximum=4500, step=100, value=MAX_CHARS_PER_CHUNK)

    script_out = gr.Textbox(label="Script", lines=7)
    audio_out = gr.Audio(label="Audio", type="filepath")

    generate_btn = gr.Button("Generate episode")
    generate_btn.click(fn=run_pipeline, inputs=[urls_text, minutes, tts_model, tts_voice, max_chars],
                       outputs=[script_out, audio_out])

demo.launch()
