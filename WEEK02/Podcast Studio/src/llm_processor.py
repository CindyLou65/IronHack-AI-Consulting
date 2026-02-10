import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_podcast_script(articles, minutes=10, model="gpt-5-nano"):
    sources_text = ""
    for i, a in enumerate(articles, start=1):
        sources_text += f"\nSOURCE {i} ({a['url']}):\n{a['text']}\n"

    prompt = f"""
You are a podcast scriptwriter and narrator.
Task: Using the sources below, write an original podcast script suitable for a {minutes}-minute episode.
Write clear, engaging narration for a general audience.
Do not copy sentences; rephrase ideas in your own words.

Sources:
{sources_text}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You write engaging podcast scripts."},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content.strip()
