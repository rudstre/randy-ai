"""Infrastructure components for the LastBlackBox system.

This module contains low-level technical components that provide
foundational capabilities for the interview system.
"""

# Audio infrastructure
from .audio import (
    extract_voice_features, AudioCapture,
    tts_say, recognize_google_sync, initialize_speaker, cleanup_speaker
)

# LLM infrastructure  
from .llm import VertexRestClient

__all__ = [
    # Audio processing
    "extract_voice_features", "AudioCapture",
    
    # Speech services
    "tts_say", "recognize_google_sync", "initialize_speaker", "cleanup_speaker",
    
    # LLM client
    "VertexRestClient"
]
