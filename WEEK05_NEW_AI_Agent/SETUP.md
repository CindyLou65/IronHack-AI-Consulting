# Project Setup

## Python Dependencies
Install via: `pip install -r requirements.txt`

## System Dependencies

### ngrok (for local development tunneling)
- Install via Microsoft Store: search "ngrok"
- Or via Chocolatey (requires admin): `choco install ngrok`
- Add authtoken: `ngrok config add-authtoken your-authtoken-here`
- Start tunnel: `ngrok http 8000`
- Note: Free plan generates random URLs on each restart

## Running the Project

### Start FastAPI server (Terminal 1):
`uvicorn api:app --reload`

### Start ngrok tunnel (Terminal 2):
`ngrok http 8000`

### Run agent directly (no API):
`python main.py`

## Environment Variables
Copy `.env.example` to `.env` and fill in:
- OPENAI_API_KEY
- TAVILY_API_KEY
- PINECONE_API_KEY
- LANGCHAIN_API_KEY
- LANGCHAIN_PROJECT=AI-Briefing-Agent
- LANGCHAIN_TRACING_V2=true
- LANGCHAIN_ENDPOINT=https://eu.api.smith.langchain.com/ ⚠️ EU endpoint - do NOT use https://api.smith.langchain.com (US endpoint will give 403 error)
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID