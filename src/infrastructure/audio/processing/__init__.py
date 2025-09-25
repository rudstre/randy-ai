"""Audio processing and capture modules."""

# Import processing functions immediately (no dependencies)
from .processing import (
    stereo_to_mono, 
    remove_dc, 
    resample_48k_to_16k, 
    normalize_audio, 
    write_wav
)

# Import features (may have dependencies)
from .features import extract_voice_features

# Lazy imports for capture (avoid importing pyaudio unless needed)
def _get_audio_capture():
    """Lazy import for AudioCapture to avoid pyaudio dependency at import time."""
    from .capture import AudioCapture
    return AudioCapture

# Export AudioCapture via __getattr__ for lazy loading
def __getattr__(name):
    if name == "AudioCapture":
        return _get_audio_capture()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "AudioCapture",
    "extract_voice_features", 
    "stereo_to_mono",
    "remove_dc",
    "resample_48k_to_16k", 
    "normalize_audio",
    "write_wav"
]
