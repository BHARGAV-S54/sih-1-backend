from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Allow your frontend (GitHub Pages) to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace "*" with your GitHub Pages domain for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Request model
class ChatRequest(BaseModel):
    message: str

# Response model
class ChatResponse(BaseModel):
    reply: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Groq model
            messages=[{"role": "user", "content": req.message}],
        )
        reply = response.choices[0].message.content
        return ChatResponse(reply=reply)
    except Exception as e:
        return ChatResponse(reply=f"Error: {str(e)}")
