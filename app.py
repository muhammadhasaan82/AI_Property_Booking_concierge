# app.py â€” FastAPI backend for OpenAI Realtime
# Adds: transcript-to-terminal, detailed diagnostics (/diag), safer logging.

import os
import tempfile
import io
from pathlib import Path
from typing import List, Optional

import httpx
import whisper
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from google.cloud import texttospeech
from pydantic import BaseModel

# -------------------- env --------------------
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    print("[WARN] OPENAI_API_KEY not set. Create a .env or export the variable.")

# Google TTS credentials
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
if not GOOGLE_APPLICATION_CREDENTIALS:
    print("[WARN] GOOGLE_APPLICATION_CREDENTIALS not set. Google TTS will not work.")
else:
    print("[INFO] Google TTS credentials found. Google TTS is ready.")

# Initialize Whisper model (load once at startup)
print("[INFO] Loading Whisper model...", flush=True)
try:
    whisper_model = whisper.load_model("base")
    print("[INFO] Whisper model loaded successfully", flush=True)
except Exception as e:
    print(f"[ERROR] Failed to load Whisper model: {e}", flush=True)
    whisper_model = None

# -------------------- app --------------------
app = FastAPI(title="Realtime Browser Demo (Python Backend)")

@app.get("/health")
async def health():
    return {"ok": True}

# -------------------- realtime: mint client secret --------------------
@app.post("/session")
async def create_session():
    """
    Mint an ephemeral client token for the browser and normalize the shape.
    """
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Server missing OPENAI_API_KEY")

    print("[/session] minting ephemeral client_secret", flush=True)

    url = "https://api.openai.com/v1/realtime/client_secrets"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "realtime=v1",
    }

    # Basic session configuration - the API handles text/audio automatically
    payload = {
        "session": {
            "type": "realtime",
            "model": "gpt-realtime",
            "instructions": (
                "You are a helpful multilingual voice assistant. "
                "You can communicate in English, Arabic, Korean, and Urdu. "
                "Respond in the same language the user speaks, or in English if unsure. "
                "Be brief and clear in your responses."
            )
        }
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(url, headers=headers, json=payload)
    except httpx.HTTPError as e:
        print(f"[/session] httpx error: {e}", flush=True)
        return JSONResponse(status_code=502, content={"error": f"Upstream connect error: {e}"})

    if r.status_code != 200:
        print(f"[/session] OpenAI returned {r.status_code}: {r.text[:400]}", flush=True)
        return JSONResponse(status_code=r.status_code, content={"error": r.text})

    # Normalize both possible shapes
    try:
        data = r.json()
    except Exception as e:
        print(f"[/session] Bad JSON: {e}, raw={r.text[:400]}", flush=True)
        return JSONResponse(status_code=500, content={"error": f"Bad JSON from OpenAI: {e}", "raw": r.text})

    value, expires_at = None, None
    cs = data.get("client_secret")
    if isinstance(cs, dict):
        value = cs.get("value"); expires_at = cs.get("expires_at")
    elif isinstance(cs, str):
        value = cs; expires_at = data.get("expires_at")
    if not value:
        value = data.get("value"); expires_at = expires_at or data.get("expires_at")

    if not value:
        print("[/session] Unexpected OpenAI response:", data, flush=True)
        return JSONResponse(status_code=500, content={"error": "No client_secret.value from OpenAI", "raw": data})

    # Log a short preview of the ephemeral key and expiry so you know it was minted
    print(f"[/session] client_secret ek=...{value[-6:]} exp={expires_at}", flush=True)
    return {"client_secret": {"value": value, "expires_at": expires_at}}

# -------------------- transcript -> terminal --------------------
class TranscriptPayload(BaseModel):
    line: Optional[str] = None
    entries: Optional[List[str]] = None

@app.post("/transcript")
async def receive_transcript(payload: TranscriptPayload):
    """
    Browser posts transcript lines here. We print them to the terminal.
    Accepts {"line": "..."} or {"entries": ["...", "..."]}.
    """
    to_print: List[str] = []
    if payload.line:
        to_print.append(payload.line)
    if payload.entries:
        to_print.extend([e for e in payload.entries if e])

    for e in to_print:
        print(f"[TRANSCRIPT] {e}", flush=True)

    return {"ok": True, "received": len(to_print)}

# -------------------- Whisper STT --------------------
@app.post("/stt")
async def speech_to_text(audio_file: UploadFile = File(...)):
    """
    Convert audio file to text using Whisper.
    Accepts audio files in various formats (mp3, wav, m4a, etc.).
    """
    if not whisper_model:
        raise HTTPException(status_code=500, detail="Whisper model not loaded")
    
    # Validate file type
    if not audio_file.content_type or not audio_file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.filename.split('.')[-1]}") as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Transcribe using Whisper
        result = whisper_model.transcribe(temp_file_path)
        transcription = result["text"].strip()
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        print(f"[STT] Transcribed: {transcription[:100]}...", flush=True)
        
        return {
            "ok": True,
            "transcription": transcription,
            "language": result.get("language", "unknown")
        }
        
    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        
        print(f"[STT] Error: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

# -------------------- Google TTS --------------------
class TTSPayload(BaseModel):
    text: str
    language_code: str = "en-US"
    voice_name: Optional[str] = None
    gender: Optional[str] = "NEUTRAL"  # MALE, FEMALE, NEUTRAL

@app.post("/tts")
async def text_to_speech(payload: TTSPayload):
    """
    Convert text to speech using Google TTS.
    Returns audio data as a streaming response.
    """
    if not GOOGLE_APPLICATION_CREDENTIALS:
        raise HTTPException(status_code=500, detail="Google TTS credentials not configured")
    
    try:
        # Initialize the client
        client = texttospeech.TextToSpeechClient()
        
        # Set up the synthesis input
        synthesis_input = texttospeech.SynthesisInput(text=payload.text)
        
        # Build the voice request
        voice = texttospeech.VoiceSelectionParams(
            language_code=payload.language_code,
            name=payload.voice_name,
            ssml_gender=getattr(texttospeech.SsmlVoiceGender, payload.gender)
        )
        
        # Select the type of audio file you want returned
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        
        # Perform the text-to-speech request
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        print(f"[TTS] Synthesized: {payload.text[:50]}...", flush=True)
        
        # Return the audio content as a streaming response
        audio_content = io.BytesIO(response.audio_content)
        return StreamingResponse(
            io.BytesIO(response.audio_content),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"}
        )
        
    except Exception as e:
        print(f"[TTS] Error: {e}", flush=True)
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(e)}")

# -------------------- diagnostics --------------------
@app.get("/diag")
async def diag():
    """
    Quick one-shot diagnostic for API key validity and token minting.
    Call: http://127.0.0.1:8000/diag
    """
    key_set = bool(OPENAI_API_KEY)
    tail = OPENAI_API_KEY[-6:] if key_set else None

    if not key_set:
        return {
            "ok": False,
            "error": "OPENAI_API_KEY missing",
            "recommendation": "Create .env with OPENAI_API_KEY=sk-... (no quotes) next to app.py",
        }

    results = {"openai_key_tail": tail}

    # 1) List models (validates key)
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r1 = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"}
            )
        results["models_status"] = r1.status_code
        if r1.status_code != 200:
            results["models_error"] = r1.text[:400]
    except Exception as e:
        results["models_exception"] = str(e)

    # 2) Try minting an ephemeral key
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r2 = await client.post(
                "https://api.openai.com/v1/realtime/client_secrets",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                    "OpenAI-Beta": "realtime=v1",
                },
                json={"session": {"type": "realtime", "model": "gpt-realtime"}}
            )
        results["client_secrets_status"] = r2.status_code
        if r2.status_code == 200:
            ek = r2.json().get("client_secret", {}).get("value") or r2.json().get("value")
            results["client_secret_tail"] = ek[-6:] if ek else None
        else:
            results["client_secrets_error"] = r2.text[:400]
    except Exception as e:
        results["client_secrets_exception"] = str(e)

    # Add STT/TTS status
    results["whisper_loaded"] = whisper_model is not None
    results["google_tts_configured"] = bool(GOOGLE_APPLICATION_CREDENTIALS)
    
    # Interpret common cases
    advice = []
    if results.get("models_status") == 401:
        advice.append("401 on /models -- API key invalid or revoked. Update OPENAI_API_KEY.")
    if results.get("client_secrets_status") == 401:
        advice.append("401 on /realtime/client_secrets -- API key invalid or not permitted for Realtime.")
    if results.get("client_secrets_status") == 403:
        advice.append("403 on /realtime/client_secrets -- org/plan not allowed for Realtime or key scope issue.")
    if results.get("client_secrets_status") == 429:
        advice.append("429 -- rate limit or quota exceeded. Check usage and limits.")
    if not results["whisper_loaded"]:
        advice.append("Whisper model not loaded. Check installation and model download.")
    if not results["google_tts_configured"]:
        advice.append("Google TTS not configured. Set GOOGLE_APPLICATION_CREDENTIALS environment variable.")
    results["advice"] = advice

    print(f"[/diag] {results}", flush=True)
    return results

# -------------------- static + index --------------------
# Serve /public files at /static and the SPA index at /
app.mount("/static", StaticFiles(directory="public", html=True), name="public")

@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse("public/index.html")

# -------------------- dev entry --------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=True)