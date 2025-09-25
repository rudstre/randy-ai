"""
Randy Configuration System
==========================

This file contains ALL configuration for the Randy interview system.
- User settings at the top (things users might want to change)
- Internal constants at the bottom (technical defaults)
"""
import os
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# USER SETTINGS - Edit these to customize Randy's behavior
# =============================================================================

# REQUIRED: Set your Google Cloud project
GOOGLE_CLOUD_PROJECT = "your-project-id"  # Change this!
GOOGLE_APPLICATION_CREDENTIALS = None  # Optional: path to credentials JSON

# Interview settings
MAX_QUESTIONS = 3
PER_TURN_SECONDS = 30.0
WORKDIR = "./_convo"

# Speech settings
ENABLE_TTS = True
TTS_VOICE = "en-US-Neural2-G"
SPEAKER_VOLUME = 0.45
LANGUAGE_CODE = "en-US"

# Voice identification
ENABLE_VOICE_PROFILES = True
VOICE_PROFILES_DIR = "./_voice_profiles"
VOICE_AGGRESSIVENESS = 0.7

# Logging
LOG_FILE = "./_convo/interview.log"
LOG_LEVEL = "INFO"


# =============================================================================
# PERSONALITY SYSTEM
# =============================================================================

@dataclass
class PersonalityConfig:
    """Randy's personality configuration with all traits."""
    # Core traits
    directness: float = 0.8
    curiosity: float = 0.7
    skepticism: float = 0.8
    engagement: float = 0.6
    tolerance: float = 0.3
    
    # Communication style
    snark: float = 0.9
    verbosity: float = 0.5
    theatricality: float = 0.6
    humor: float = 0.8
    intimidation: float = 0.6
    
    # Social dynamics
    dominance: float = 0.6
    playfulness: float = 0.8
    empathy: float = 0.5
    intensity: float = 0.7
    
    # Behavioral traits
    randomness: float = 0.6
    weirdness: float = 0.8
    chaos: float = 0.8
    philosophical: float = 0.7
    wisdom: float = 0.8
    
    # Custom personality instructions
    custom_context: str = ""
    
    @classmethod
    def from_preset(cls, preset_name: str) -> 'PersonalityConfig':
        """Create personality from preset."""
        presets = {
            "troll": cls(
                chaos=0.9, randomness=0.9, weirdness=0.8, playfulness=0.8, 
                snark=0.7, tolerance=0.2, empathy=0.2
            ),
            "detective": cls(
                curiosity=0.9, skepticism=0.8, intensity=0.7, verbosity=0.8, 
                snark=0.4, directness=0.9, chaos=0.1
            ),
            "therapist": cls(
                empathy=0.9, tolerance=0.8, intensity=0.2, verbosity=0.7,
                directness=0.3, curiosity=0.6, intimidation=0.2
            ),
            "comedian": cls(
                humor=1.0, playfulness=0.8, theatricality=0.8, randomness=0.6, 
                snark=0.7, engagement=0.9, empathy=0.5
            ),
            "chaos_agent": cls(
                chaos=1.0, randomness=1.0, weirdness=0.8, theatricality=0.7,
                playfulness=0.9, tolerance=0.1, verbosity=0.9
            ),
            "intimidator": cls(
                intimidation=1.0, directness=1.0, dominance=1.0, intensity=1.0,
                tolerance=0.1, skepticism=0.9, empathy=0.1
            ),
            "philosopher": cls(
                philosophical=0.9, wisdom=0.8, verbosity=0.8, curiosity=0.8, 
                empathy=0.6, intensity=0.3, directness=0.4, tolerance=0.7
            ),
        }
        return presets.get(preset_name, cls())


# Personality presets (set to True to use a preset instead of custom values above)
USE_PRESET_PERSONALITY = False
PRESET_PERSONALITY = "default"  # Options: troll, detective, therapist, comedian, chaos_agent, intimidator, philosopher


# =============================================================================
# INTERNAL CONSTANTS - Don't change these unless you know what you're doing
# =============================================================================

# Audio processing
SAMPLE_RATE_CAPTURE = 48000
SAMPLE_RATE_TARGET = 16000
CHANNELS = 2
FRAME_MS = 30
MIC_GAIN = 10.0
TARGET_RMS = 0.06
MAX_SAMPLES_30S = 48000 * 30

# Voice Activity Detection
VAD_SILENCE_THRESHOLD = 0.01
VAD_SILENCE_DURATION = 2.0
VAD_MIN_SPEECH_DURATION = 1.0

# Hardware
SPEAKER_DEVICE = 1
SPEAKER_SAMPLE_RATE = 16000
SPEAKER_BUFFER_SIZE = 1600
ALSA_LIB_NAMES = ("libasound.so.2", "libasound.so")

# TTS technical
TTS_PITCH = 55
TTS_AMPLITUDE = 120
TTS_RATE_WPM = 180

# Voice identification technical
SPEAKER_SIMILARITY_THRESHOLD = 0.7
PROFILE_CONFIDENCE_THRESHOLD = 0.6
ADAPTIVE_TOLERANCE_ENABLED = True
ADAPTIVE_LENIENCY_FACTOR = 0.5
ADAPTIVE_CONFIDENCE_ADJUSTMENT = 0.3
DATA_CONFIDENCE_SCALE = 8.0

# LLM
VERTEX_LOCATION = "us-central1"
MODEL_NAME = "gemini-2.5-flash-lite"
LLM_TIMEOUT = 60
MAX_OUTPUT_TOKENS = 512

# OpenSMILE
OPENSMILE_CONFIG_CANDIDATES = [
    os.getenv("OPENSMILE_CONFIG"),
    "/usr/local/opensmile-config/egemaps/v02/eGeMAPSv02.conf",
]

CANONICAL_FEATURE_MAPPING = {
    "pitch_mean": "F0semitoneFrom27.5Hz_sma3nz_amean",
    "pitch_std": "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "pitch_range": "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
    "formant_1_freq": "F1frequencySma3nz_amean",
    "formant_2_freq": "F2frequencySma3nz_amean", 
    "formant_3_freq": "F3frequencySma3nz_amean",
    "formant_1_bandwidth": "F1bandwidthSma3nz_amean",
    "formant_2_bandwidth": "F2bandwidthSma3nz_amean",
    "mfcc_1": "mfcc1_sma3_amean",
    "mfcc_2": "mfcc2_sma3_amean",
    "mfcc_3": "mfcc3_sma3_amean",
    "mfcc_4": "mfcc4_sma3_amean",
    "mfcc_5": "mfcc5_sma3_amean",
    "mfcc_6": "mfcc6_sma3_amean",
    "mfcc_7": "mfcc7_sma3_amean",
    "mfcc_8": "mfcc8_sma3_amean",
    "mfcc_9": "mfcc9_sma3_amean",
    "mfcc_10": "mfcc10_sma3_amean",
    "mfcc_11": "mfcc11_sma3_amean",
    "mfcc_12": "mfcc12_sma3_amean",
    "jitter": "jitterLocal_sma3nz_amean",
    "shimmer": "shimmerLocaldB_sma3nz_amean",
    "hnr": "HNRdBACF_sma3nz_amean",
    "spectral_centroid": "spectralCentroidSma3_amean",
    "spectral_rolloff": "spectralRollOff25Sma3_amean", 
    "spectral_flux": "spectralFluxSma3_amean",
    "loudness": "loudness_sma3_amean",
    "rms_energy": "RMSenergySma3_amean",
    "zcr": "zcr_sma3_amean",
}


# =============================================================================
# MAIN CONFIG OBJECT
# =============================================================================

@dataclass
class Config:
    """Main configuration object."""
    google_cloud_project: str
    google_application_credentials: Optional[str] = None
    max_questions: int = MAX_QUESTIONS
    per_turn_seconds: float = PER_TURN_SECONDS
    workdir: str = WORKDIR
    enable_tts: bool = ENABLE_TTS
    tts_voice: str = TTS_VOICE
    tts_rate_wpm: int = TTS_RATE_WPM
    tts_pitch: int = TTS_PITCH
    tts_amplitude: int = TTS_AMPLITUDE
    speaker_volume: float = SPEAKER_VOLUME
    language_code: str = LANGUAGE_CODE
    enable_voice_profiles: bool = ENABLE_VOICE_PROFILES
    voice_profiles_dir: str = VOICE_PROFILES_DIR
    voice_aggressiveness: float = VOICE_AGGRESSIVENESS
    speaker_similarity_threshold: float = SPEAKER_SIMILARITY_THRESHOLD
    log_file: str = LOG_FILE
    log_level: str = LOG_LEVEL
    
    def get_personality_config(self) -> PersonalityConfig:
        """Get personality configuration."""
        if USE_PRESET_PERSONALITY and PRESET_PERSONALITY != "default":
            return PersonalityConfig.from_preset(PRESET_PERSONALITY)
        else:
            return PersonalityConfig()


def get_config() -> Config:
    """Load configuration."""
    project = os.getenv("GOOGLE_CLOUD_PROJECT") or GOOGLE_CLOUD_PROJECT
    credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or GOOGLE_APPLICATION_CREDENTIALS
    
    if project == "your-project-id":
        raise ValueError("Please set GOOGLE_CLOUD_PROJECT in config.py or as environment variable")
    
    return Config(
        google_cloud_project=project,
        google_application_credentials=credentials
    )


# For backward compatibility (old imports)
get_user_config = get_config
PersonalityTraits = PersonalityConfig

# Legacy constant names for backward compatibility
DEFAULT_MAX_QUESTIONS = MAX_QUESTIONS
DEFAULT_PER_TURN_SECONDS = PER_TURN_SECONDS  
DEFAULT_WORKDIR = WORKDIR
DEFAULT_ENABLE_TTS = ENABLE_TTS
DEFAULT_TTS_VOICE = TTS_VOICE
DEFAULT_TTS_RATE_WPM = TTS_RATE_WPM
DEFAULT_SPEAKER_VOLUME = SPEAKER_VOLUME
DEFAULT_LANGUAGE_CODE = LANGUAGE_CODE
DEFAULT_ENABLE_VOICE_PROFILES = ENABLE_VOICE_PROFILES
DEFAULT_VOICE_PROFILES_DIR = VOICE_PROFILES_DIR
DEFAULT_VOICE_AGGRESSIVENESS = VOICE_AGGRESSIVENESS
DEFAULT_LOG_FILE = LOG_FILE
DEFAULT_LOG_LEVEL = LOG_LEVEL

# Internal constants with legacy names
DEFAULT_SAMPLE_RATE_CAPTURE = SAMPLE_RATE_CAPTURE
DEFAULT_SAMPLE_RATE_TARGET = SAMPLE_RATE_TARGET
DEFAULT_CHANNELS = CHANNELS
DEFAULT_FRAME_MS = FRAME_MS
DEFAULT_MIC_GAIN = MIC_GAIN
DEFAULT_TARGET_RMS = TARGET_RMS
DEFAULT_MAX_SAMPLES_30S = MAX_SAMPLES_30S
DEFAULT_VAD_SILENCE_THRESHOLD = VAD_SILENCE_THRESHOLD
DEFAULT_VAD_SILENCE_DURATION = VAD_SILENCE_DURATION
DEFAULT_VAD_MIN_SPEECH_DURATION = VAD_MIN_SPEECH_DURATION
DEFAULT_SPEAKER_DEVICE = SPEAKER_DEVICE
DEFAULT_SPEAKER_SAMPLE_RATE = SPEAKER_SAMPLE_RATE
DEFAULT_SPEAKER_BUFFER_SIZE = SPEAKER_BUFFER_SIZE
DEFAULT_TTS_PITCH = TTS_PITCH
DEFAULT_TTS_AMPLITUDE = TTS_AMPLITUDE
DEFAULT_SPEAKER_SIMILARITY_THRESHOLD = SPEAKER_SIMILARITY_THRESHOLD
DEFAULT_PROFILE_CONFIDENCE_THRESHOLD = PROFILE_CONFIDENCE_THRESHOLD
DEFAULT_ADAPTIVE_TOLERANCE_ENABLED = ADAPTIVE_TOLERANCE_ENABLED
DEFAULT_ADAPTIVE_LENIENCY_FACTOR = ADAPTIVE_LENIENCY_FACTOR
DEFAULT_ADAPTIVE_CONFIDENCE_ADJUSTMENT = ADAPTIVE_CONFIDENCE_ADJUSTMENT
DEFAULT_DATA_CONFIDENCE_SCALE = DATA_CONFIDENCE_SCALE
DEFAULT_VERTEX_LOCATION = VERTEX_LOCATION
DEFAULT_MODEL_NAME = MODEL_NAME
DEFAULT_LLM_TIMEOUT = LLM_TIMEOUT
DEFAULT_MAX_OUTPUT_TOKENS = MAX_OUTPUT_TOKENS
