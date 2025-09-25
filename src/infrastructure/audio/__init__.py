"""
Audio processing, speech services, and voice identification for LastBlackBox.

This module contains all audio-related functionality organized into clear submodules:
- hardware: NB3 hardware drivers for audio I/O
- processing: Signal processing, capture, and feature extraction  
- speech: Text-to-speech and speech-to-text services
- voice_id: Speaker identification and voice analysis
"""

# Convenient imports from submodules
from .processing import extract_voice_features, AudioCapture
from .speech import tts_say, recognize_google_sync, initialize_speaker, cleanup_speaker

__all__ = [
    "extract_voice_features", 
    "AudioCapture",
    "tts_say", 
    "recognize_google_sync", 
    "initialize_speaker", 
    "cleanup_speaker"
]