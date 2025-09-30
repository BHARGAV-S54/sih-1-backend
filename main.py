from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os, requests
from langdetect import detect

app = FastAPI()

# Allow frontend calls (you can restrict to your GitHub Pages domain later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API keys from Render environment
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    lang: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    user_msg = req.message

    # Step 1: Detect language
    try:
        detected_lang = detect(user_msg)  # e.g. 'hi' for Hindi, 'te' for Telugu
    except Exception:
        detected_lang = "en"

    # Step 2: Generate reply in English using Sarvam
    try:
        sarvam_res = requests.post(
            "https://api.sarvam.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {SARVAM_API_KEY}"},
            json={
                "model": "sarvam-m-1",
                "messages": [{"role": "user", "content": user_msg}]
            },
            timeout=20
        )
        sarvam_res.raise_for_status()
        reply_en = sarvam_res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return ChatResponse(reply=f"Sarvam error: {str(e)}", lang="en")

    # Step 3: If input was not English, translate reply back
    if detected_lang != "en":
        try:
            hf_res = requests.post(
                "https://api-inference.huggingface.co/models/ai4bharat/indictrans2-en-indic-1B",
                headers={"Authorization": f"Bearer {HF_API_KEY}"},
                json={"inputs": reply_en},
                timeout=20
            )
            hf_res.raise_for_status()
            reply = hf_res.json()[0]["translation_text"]
            return ChatResponse(reply=reply, lang=detected_lang)
        except Exception as e:
            # Fallback: return English if translation fails
            return ChatResponse(reply=reply_en, lang="en")

    # If English, return directly
    return ChatResponse(reply=reply_en, lang="en")
