"""Audio hardware drivers for NB3 (Neural Board 3) Raspberry Pi."""

# Conditional imports to handle missing pyaudio dependency gracefully
try:
    from .microphone import Microphone
    from .speaker import Speaker
    from . import utilities
    __all__ = ["Microphone", "Speaker", "utilities"]
except ImportError:
    # pyaudio not available - this is expected in non-RPi environments
    __all__ = []