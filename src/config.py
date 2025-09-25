"""
Configuration constants and settings for the conversation integrator.
Enhanced with environment variable management and validation.
"""
import os
from dataclasses import dataclass
from typing import Optional

# OpenSMILE configuration
OPENSMILE_CONFIG_CANDIDATES = [
    os.getenv("OPENSMILE_CONFIG"),
    "/usr/local/opensmile-config/egemaps/v02/eGeMAPSv02.conf",
]

# Canonical mapping of exact column names for OpenSMILE features
# Optimized for speaker identification - includes ALL best features for maximum accuracy
CANONICAL_FEATURE_MAPPING = {
    # Tier 1: Highest discriminative power (Essential for speaker ID)
    "pitch_mean": "F0semitoneFrom27.5Hz_sma3nz_amean",
    "pitch_std": "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "pitch_range": "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
    
    # Formant frequencies (critical for speaker identification)
    "formant_1_freq": "F1frequencySma3nz_amean",
    "formant_2_freq": "F2frequencySma3nz_amean", 
    "formant_3_freq": "F3frequencySma3nz_amean",
    "formant_1_bandwidth": "F1bandwidthSma3nz_amean",
    "formant_2_bandwidth": "F2bandwidthSma3nz_amean",
    
    # MFCCs (most important spectral features)
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
    
    # Voice quality measures (high discriminative power)
    "jitter": "jitterLocal_sma3nz_amean",
    "shimmer": "shimmerLocaldB_sma3nz_amean",
    "hnr": "HNRdBACF_sma3nz_amean",
    
    # Spectral features (good discriminative power)
    "spectral_centroid": "spectralCentroidSma3_amean",
    "spectral_rolloff": "spectralRollOff25Sma3_amean", 
    "spectral_flux": "spectralFluxSma3_amean",
    
    # Energy features
    "loudness": "loudness_sma3_amean",
    "rms_energy": "RMSenergySma3_amean",
    "zcr": "zcr_sma3_amean",
}

# Audio processing defaults
DEFAULT_SAMPLE_RATE_CAPTURE = 48000
DEFAULT_SAMPLE_RATE_TARGET = 16000
DEFAULT_CHANNELS = 2
DEFAULT_FRAME_MS = 30
DEFAULT_MIC_GAIN = 10.0
DEFAULT_TARGET_RMS = 0.06
DEFAULT_MAX_SAMPLES_30S = 48000 * 30

# Speech and TTS defaults
DEFAULT_LANGUAGE_CODE = "en-US"
DEFAULT_TTS_RATE_WPM = 180  # Words per minute (80-450)
DEFAULT_TTS_VOICE = "en-US-Neural2-G"  # Google Cloud TTS voice name
DEFAULT_TTS_PITCH = 55       # Pitch (0-99, default 50) - not used with Google TTS
DEFAULT_TTS_AMPLITUDE = 120  # Volume (0-200, default 100)
DEFAULT_ENABLE_TTS = True    # Enable TTS by default (set to False for text-only default)
# Google Cloud TTS Voice Options (High Quality Neural Voices):
# 
# FEMALE VOICES:
# "en-US-Neural2-F" - Natural US female
# "en-US-Neural2-G" - Casual US female
# "en-US-Neural2-H" - Conversational US female
# "en-GB-Neural2-A" - British female
# "en-GB-Neural2-C" - Elegant British female
# "en-AU-Neural2-A" - Australian female
#
# MALE VOICES:
# "en-US-Neural2-A" - Deep US male
# "en-US-Neural2-C" - Warm US male  
# "en-US-Neural2-D" - Confident US male
# "en-GB-Neural2-B" - British male
# "en-GB-Neural2-D" - Sophisticated British male
# "en-AU-Neural2-B" - Australian male
#
# UNIQUE VOICES:
# "en-US-Neural2-I" - Child-like voice
# "en-US-Neural2-J" - News anchor style

# NB3 Speaker defaults
DEFAULT_SPEAKER_DEVICE = 1      # Audio output device for RPi
DEFAULT_SPEAKER_SAMPLE_RATE = 16000  # Match TTS output
DEFAULT_SPEAKER_BUFFER_SIZE = 1600   # Buffer size for audio playback
DEFAULT_SPEAKER_VOLUME = 0.45   # Speaker volume (0.0-1.0, default 0.25)

# LLM defaults
DEFAULT_VERTEX_LOCATION = "us-central1"
DEFAULT_MODEL_NAME = "gemini-2.5-flash-lite"
DEFAULT_LLM_TIMEOUT = 60
DEFAULT_MAX_OUTPUT_TOKENS = 512

# Interview defaults
DEFAULT_MAX_QUESTIONS = 3
DEFAULT_PER_TURN_SECONDS = 30.0  # Max time to wait
DEFAULT_WORKDIR = "./_convo"

# AI Interviewer personality parameters (0.0 = minimum, 1.0 = maximum)
DEFAULT_INTERVIEWER_DIRECTNESS = 0.8      # How blunt/straightforward (0=diplomatic, 1=brutally honest)
DEFAULT_INTERVIEWER_CURIOSITY = 0.7       # How probing/inquisitive (0=surface-level, 1=deep-diving)
DEFAULT_INTERVIEWER_SKEPTICISM = 0.8      # How jaded/suspicious (0=trusting, 1=cynical)
DEFAULT_INTERVIEWER_ENGAGEMENT = 0.8      # How energetic/interesting (0=boring, 1=captivating)
DEFAULT_INTERVIEWER_TOLERANCE = 0.2       # Patience with difficult people (0=quick to dismiss, 1=very patient)

# Additional custom personality context (appended to generated personality from parameters above)
DEFAULT_INTERVIEWER_PERSONALITY_CONTEXT = ""

# Voice Activity Detection settings
DEFAULT_VAD_SILENCE_THRESHOLD = 0.01  # RMS threshold for silence (increased from 0.01 for less sensitivity)
DEFAULT_VAD_SILENCE_DURATION = 2.0    # Seconds of silence to end recording
DEFAULT_VAD_MIN_SPEECH_DURATION = 1.0 # Minimum speech before silence detection starts

# Logging configuration
DEFAULT_LOG_FILE = "./_convo/interview.log"
DEFAULT_LOG_LEVEL = "INFO"

# Voice profile settings
DEFAULT_VOICE_PROFILES_DIR = "./_voice_profiles"
DEFAULT_SPEAKER_SIMILARITY_THRESHOLD = 0.7  # Minimum similarity for speaker identification
DEFAULT_ENABLE_VOICE_PROFILES = True  # Enable voice profile storage and identification
DEFAULT_PROFILE_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to update existing profile

# Voice identification aggressiveness settings
DEFAULT_VOICE_AGGRESSIVENESS = 0.7  # 0.0 = very conservative, 1.0 = very aggressive
# Examples:
# 0.0 = Maximum security, minimal false positives, may miss some hostile speakers
# 0.3 = Conservative, good for public-facing systems
# 0.5 = Balanced approach
# 0.7 = Moderately aggressive (current default)
# 1.0 = Maximum aggressiveness, catch all hostile speakers, higher false positives

# Adaptive threshold settings
DEFAULT_ADAPTIVE_TOLERANCE_ENABLED = True  # Enable adaptive tolerance based on data quantity
DEFAULT_ADAPTIVE_LENIENCY_FACTOR = 0.5  # How much more lenient to be with low data (0.0-1.0)
DEFAULT_ADAPTIVE_CONFIDENCE_ADJUSTMENT = 0.3  # Max threshold reduction for new speakers (0.0-0.5)
DEFAULT_DATA_CONFIDENCE_SCALE = 8.0  # Sample size scaling factor for confidence calculation

# ALSA/Audio library names for error suppression
ALSA_LIB_NAMES = ("libasound.so.2", "libasound.so")


@dataclass
class Config:
    """
    Enhanced configuration class with environment variable loading and validation.
    
    This class provides a structured way to manage configuration with proper
    environment variable loading, validation, and type safety.
    """
    
    # Google Cloud settings (from environment)
    google_cloud_project: str
    google_application_credentials: Optional[str] = None
    
    # OpenSMILE configuration
    opensmile_config: Optional[str] = None
    
    # Audio processing settings
    sample_rate_capture: int = DEFAULT_SAMPLE_RATE_CAPTURE
    sample_rate_target: int = DEFAULT_SAMPLE_RATE_TARGET
    channels: int = DEFAULT_CHANNELS
    frame_ms: int = DEFAULT_FRAME_MS
    mic_gain: float = DEFAULT_MIC_GAIN
    target_rms: float = DEFAULT_TARGET_RMS
    max_samples_30s: int = DEFAULT_MAX_SAMPLES_30S
    
    # Speech and TTS settings
    language_code: str = DEFAULT_LANGUAGE_CODE
    tts_rate_wpm: int = DEFAULT_TTS_RATE_WPM
    tts_voice: str = DEFAULT_TTS_VOICE
    tts_pitch: int = DEFAULT_TTS_PITCH
    tts_amplitude: int = DEFAULT_TTS_AMPLITUDE
    enable_tts: bool = DEFAULT_ENABLE_TTS    
    # Speaker settings
    speaker_device: int = DEFAULT_SPEAKER_DEVICE
    speaker_sample_rate: int = DEFAULT_SPEAKER_SAMPLE_RATE
    speaker_buffer_size: int = DEFAULT_SPEAKER_BUFFER_SIZE
    
    # LLM settings
    vertex_location: str = DEFAULT_VERTEX_LOCATION
    model_name: str = DEFAULT_MODEL_NAME
    llm_timeout: int = DEFAULT_LLM_TIMEOUT
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    
    # Interview settings
    max_questions: int = DEFAULT_MAX_QUESTIONS
    per_turn_seconds: float = DEFAULT_PER_TURN_SECONDS
    workdir: str = DEFAULT_WORKDIR
    
    # AI Interviewer personality parameters
    interviewer_directness: float = DEFAULT_INTERVIEWER_DIRECTNESS
    interviewer_curiosity: float = DEFAULT_INTERVIEWER_CURIOSITY  
    interviewer_skepticism: float = DEFAULT_INTERVIEWER_SKEPTICISM
    interviewer_engagement: float = DEFAULT_INTERVIEWER_ENGAGEMENT
    interviewer_tolerance: float = DEFAULT_INTERVIEWER_TOLERANCE
    interviewer_personality_context: str = DEFAULT_INTERVIEWER_PERSONALITY_CONTEXT
    
    # Voice Activity Detection settings
    vad_silence_threshold: float = DEFAULT_VAD_SILENCE_THRESHOLD
    vad_silence_duration: float = DEFAULT_VAD_SILENCE_DURATION
    vad_min_speech_duration: float = DEFAULT_VAD_MIN_SPEECH_DURATION
    
    # Logging settings
    log_file: str = DEFAULT_LOG_FILE
    log_level: str = DEFAULT_LOG_LEVEL
    
    # Voice profile settings
    voice_profiles_dir: str = DEFAULT_VOICE_PROFILES_DIR
    speaker_similarity_threshold: float = DEFAULT_SPEAKER_SIMILARITY_THRESHOLD
    enable_voice_profiles: bool = DEFAULT_ENABLE_VOICE_PROFILES
    profile_confidence_threshold: float = DEFAULT_PROFILE_CONFIDENCE_THRESHOLD
    voice_aggressiveness: float = DEFAULT_VOICE_AGGRESSIVENESS
    
    @classmethod
    def from_environment(cls, **overrides) -> 'Config':
        """
        Load configuration from environment variables with optional overrides.
        
        Args:
            **overrides: Any configuration values to override
            
        Returns:
            Config instance with values loaded from environment
            
        Raises:
            ValueError: If required environment variables are missing
        """
        # Required environment variables
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT environment variable must be set. "
                "Add 'export GOOGLE_CLOUD_PROJECT=your-project-id' to your ~/.bashrc"
            )
        
        # Optional environment variables
        config_data = {
            "google_cloud_project": project,
            "google_application_credentials": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            "opensmile_config": os.getenv("OPENSMILE_CONFIG"),
        }
        
        # Apply any overrides
        config_data.update(overrides)
        
        return cls(**config_data)
    
    def validate(self) -> None:
        """
        Validate configuration values.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate required fields
        if not self.google_cloud_project:
            raise ValueError("Google Cloud project ID is required")
        
        # Validate credentials file if provided
        if self.google_application_credentials:
            if not os.path.exists(self.google_application_credentials):
                raise ValueError(
                    f"Google Application Credentials file not found: "
                    f"{self.google_application_credentials}"
                )
        
        # Validate OpenSMILE config if provided
        if self.opensmile_config and not os.path.exists(self.opensmile_config):
            raise ValueError(f"OpenSMILE config file not found: {self.opensmile_config}")
        
        # Validate numeric ranges
        if not (0.0 <= self.voice_aggressiveness <= 1.0):
            raise ValueError("voice_aggressiveness must be between 0.0 and 1.0")
        
        if self.max_questions <= 0:
            raise ValueError("max_questions must be positive")
        
        if self.per_turn_seconds <= 0:
            raise ValueError("per_turn_seconds must be positive")
    
    def get_opensmile_config(self) -> str:
        """
        Get the OpenSMILE configuration file path.
        
        Returns:
            Path to OpenSMILE config file
            
        Raises:
            RuntimeError: If no valid config file is found
        """
        candidates = [
            self.opensmile_config,
            *OPENSMILE_CONFIG_CANDIDATES
        ]
        
        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                return candidate
        
        raise RuntimeError(
            "No valid OpenSMILE config file found. "
            "Install OpenSMILE or set OPENSMILE_CONFIG environment variable."
        )
    
    def create_workdir(self) -> str:
        """
        Create and return the working directory path.
        
        Returns:
            Absolute path to working directory
        """
        os.makedirs(self.workdir, exist_ok=True)
        return os.path.abspath(self.workdir)


# Global config instance (can be overridden)
_global_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Loads from environment on first call, then returns cached instance.
    
    Returns:
        Config instance
    """
    global _global_config
    if _global_config is None:
        _global_config = Config.from_environment()
        _global_config.validate()
    return _global_config


def set_config(config: Config) -> None:
    """
    Set the global configuration instance.
    
    Args:
        config: Config instance to use globally
    """
    global _global_config
    _global_config = config
