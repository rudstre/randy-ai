"""Voice profile management and speaker identification."""

from .profiles import VoiceProfile, VoiceProfileManager
from .identification import ProgressiveVoiceIdentifier, IdentificationConfidence, SpeakerDisposition
from .similarity import SimilarityCalculator

__all__ = [
    "VoiceProfile",
    "VoiceProfileManager", 
    "ProgressiveVoiceIdentifier",
    "IdentificationConfidence",
    "SpeakerDisposition",
    "SimilarityCalculator"
]
