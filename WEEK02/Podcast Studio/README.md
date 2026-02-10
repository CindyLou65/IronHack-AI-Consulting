# Podcast Studio

A Python project that generates narrated podcast episodes from online articles using AI.

## Features

- Fetch and process articles from URLs.
- Generate a podcast script using OpenAI's GPT models.
- Convert script to audio using text-to-speech.
- Chunking and concatenation of audio for long scripts.
- Web interface via Gradio for easy interaction.

## Folder Structure

podcast-studio/
├── src/
│ ├── data_processor.py # Fetch and process articles
│ ├── llm_processor.py # Generate podcast scripts using OpenAI
│ ├── tts_generator.py # Text-to-speech generation and audio handling
│ └── main.py # Gradio application
├── templates/ # Optional templates for Gradio
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .env # API keys (not committed)


## Setup

1. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows

2. Install dependencies:
pip install -r requirements.txt

3. Create a .env file in the root directory with your OpenAI API key:
OPENAI_API_KEY=your_api_key_here

## Usage

Run the Gradio app:
python src/main.py
Paste article URLs (one per line).

Set target podcast length (minutes).

Choose TTS model and voice.

Click "Generate episode" to produce both script and audio.

The final audio will be saved in podcast_output/podcast_episode_final.mp3.

## Notes

Make sure ffmpeg is installed if you want fast MP3 concatenation.

Large scripts may require chunking to fit TTS limits.

.env file is not committed for security.