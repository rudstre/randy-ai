"""Voice biometric identification and similarity analysis."""

# Voice biometric identification exports
from .biometrics import VoiceBiometricProfile
from .identification import ProgressiveVoiceIdentifier, IdentificationConfidence, SpeakerDisposition
from .similarity import SimilarityCalculator

__all__ = [
    "VoiceBiometricProfile",
    "ProgressiveVoiceIdentifier", 
    "IdentificationConfidence",
    "SpeakerDisposition",
    "SimilarityCalculator"
]
