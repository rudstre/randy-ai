"""
Speech-to-text functionality using Google Cloud Speech.
"""
import logging

from google.cloud import speech
from ....config import DEFAULT_LANGUAGE_CODE

logger = logging.getLogger("speech_stt")


def recognize_google_sync(pcm16_bytes: bytes, 
                         sr_hz: int = 16000, 
                         language: str = DEFAULT_LANGUAGE_CODE) -> str:
    """
    Synchronous Google Cloud Speech-to-Text recognition.
    Returns transcribed text or empty string if no speech detected.
    """
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=pcm16_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sr_hz,
        language_code=language,
        enable_automatic_punctuation=True,
    )
    
    try:
        resp = client.recognize(config=config, audio=audio)
        texts = [r.alternatives[0].transcript for r in resp.results if r.alternatives]
        return " ".join(texts).strip()
    except Exception as e:
        logger.error("Speech recognition failed: %s", e)
        return ""
