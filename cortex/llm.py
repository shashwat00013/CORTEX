"""One function, two providers: Gemini API or local Ollama."""
import requests

from . import config


def generate(prompt, system=""):
    if config.LLM_PROVIDER == "ollama":
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        r = requests.post(
            f"{config.OLLAMA_URL}/api/chat",
            json={"model": config.OLLAMA_MODEL, "messages": msgs, "stream": False},
            timeout=300,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY missing. Add it to .env (see .env.example).")
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    resp = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system or None),
    )
    return resp.text
