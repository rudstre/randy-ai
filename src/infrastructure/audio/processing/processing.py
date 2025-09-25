"""
Basic audio processing functions including format conversions and normalization.
"""
import wave
import numpy as np
from scipy.signal import resample_poly

from ....config import TARGET_RMS


def stereo_to_mono(x: np.ndarray) -> np.ndarray:
    """Convert stereo audio to mono by averaging channels."""
    return np.mean(x, axis=1)


def remove_dc(x: np.ndarray) -> np.ndarray:
    """Remove DC offset from audio signal."""
    return x - np.mean(x)


def resample_48k_to_16k(mono48: np.ndarray) -> np.ndarray:
    """Resample 48kHz audio to 16kHz."""
    return resample_poly(mono48, up=1, down=3).astype(np.float32)


def normalize_audio(audio: np.ndarray, target_rms: float = TARGET_RMS) -> np.ndarray:
    """Normalize audio to target RMS level."""
    rms = float(np.sqrt(np.mean(audio**2)) + 1e-9)
    gain = min(20.0, target_rms / rms) if rms > 0 else 1.0
    return audio * gain


def write_wav(path: str, pcm16: np.ndarray, sr: int, channels: int = 1) -> None:
    """Write PCM16 audio data to WAV file."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
