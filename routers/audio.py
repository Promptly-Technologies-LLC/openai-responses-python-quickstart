import logging
import os

from dotenv import load_dotenv
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import PlainTextResponse
from openai import AsyncOpenAI

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/audio", tags=["audio"])


@router.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)) -> PlainTextResponse:
    """Transcribe an uploaded audio file using OpenAI Whisper."""
    load_dotenv(override=True)
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    transcription = await client.audio.transcriptions.create(
        model="whisper-1",
        file=(audio.filename, await audio.read(), audio.content_type),
    )

    return PlainTextResponse(transcription.text)
