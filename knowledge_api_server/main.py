from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from knowledge_inference.service import InferenceService

app = FastAPI()

# ⭐ ADD CORS HERE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = InferenceService()
service.initialize()


class ChatRequest(BaseModel):
    query: str
    debug: bool = True


@app.post("/chat")
def chat(req: ChatRequest):
    result = service.answer(req.query, debug=req.debug)

    return {
        "answer": result.answer,
        "confidence": result.confidence,
        "evidence": result.evidence,
        "debug": result.debug,
    }