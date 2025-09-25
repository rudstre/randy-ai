"""Speech-to-text and text-to-speech modules."""

from .tts import tts_say, initialize_speaker, cleanup_speaker
from .stt import recognize_google_sync

__all__ = ["tts_say", "initialize_speaker", "cleanup_speaker", "recognize_google_sync"]
