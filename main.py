from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os, requests

app = FastAPI()

# Allow frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later to your GitHub Pages domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    user_msg = req.message

    # 1. Try Sarvam AI
    try:
        sarvam_res = requests.post(
            "https://api.sarvam.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {SARVAM_API_KEY}"},
            json={
                "model": "sarvam-m-1",  # multilingual chat model
                "messages": [{"role": "user", "content": user_msg}]
            },
            timeout=15
        )
        sarvam_res.raise_for_status()
        reply = sarvam_res.json()["choices"][0]["message"]["content"]
        return ChatResponse(reply=reply)
    except Exception as e:
        print("Sarvam failed:", e)

    # 2. Fallback: Hugging Face AI4Bharat (IndicTrans2 example)
    try:
        hf_res = requests.post(
            "https://api-inference.huggingface.co/models/ai4bharat/indictrans2-en-indic-1B",
            headers={"Authorization": f"Bearer {HF_API_KEY}"},
            json={"inputs": user_msg},
            timeout=15
        )
        hf_res.raise_for_status()
        reply = hf_res.json()[0]["translation_text"]
        return ChatResponse(reply=reply)
    except Exception as e:
        return ChatResponse(reply=f"AI service error: {str(e)}")
