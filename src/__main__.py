#!/usr/bin/env python3
"""
Main entry point for the LastBlackBox interview system.
Allows running the package with: python -m lastblackbox
"""
import sys
from .config import get_config, DEFAULT_VOICE_AGGRESSIVENESS, DEFAULT_ENABLE_TTS, DEFAULT_SPEAKER_VOLUME
from . import InterviewOrchestrator


def main():
    """Command-line interface for the interview orchestrator."""
    
    # Load configuration from environment
    try:
        config = get_config()
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        sys.exit(1)

    # TTS configuration with explicit flags taking precedence
    explicit_tts = "--tts" in sys.argv or "--speech" in sys.argv
    explicit_text = "--text" in sys.argv or "--no-tts" in sys.argv
    
    if explicit_text:
        use_tts = False
    elif explicit_tts:
        use_tts = True
    else:
        use_tts = config.enable_tts  # Use config default    
    # Check for aggressiveness and volume parameters
    voice_aggressiveness = DEFAULT_VOICE_AGGRESSIVENESS
    speaker_volume = DEFAULT_SPEAKER_VOLUME
    for arg in sys.argv:
        if arg.startswith("--aggressiveness="):
            try:
                voice_aggressiveness = float(arg.split("=")[1])
                voice_aggressiveness = max(0.0, min(1.0, voice_aggressiveness))  # Clamp 0-1
            except (ValueError, IndexError):
                print("❌ Invalid aggressiveness value. Use --aggressiveness=0.0 to --aggressiveness=1.0")
                sys.exit(1)
        elif arg.startswith("--volume="):
            try:
                speaker_volume = float(arg.split("=")[1])
                speaker_volume = max(0.0, min(1.0, speaker_volume))  # Clamp 0-1
            except (ValueError, IndexError):
                print("❌ Invalid volume value. Use --volume=0.0 to --volume=1.0")
                sys.exit(1)
        elif arg in ["--conservative", "--safe"]:
            voice_aggressiveness = 0.3
        elif arg in ["--aggressive", "--security"]:
            voice_aggressiveness = 0.9
        elif arg in ["--balanced"]:
            voice_aggressiveness = 0.5
        elif arg in ["--quiet"]:
            speaker_volume = 0.1
        elif arg in ["--loud"]:
            speaker_volume = 0.8
    
    # Show configuration
    if use_tts:
        print("🔊 TTS Mode: Robot will speak questions aloud (default)")
        print("   (Use --text or --no-tts to disable speech)")
        print(f"🔉 Speaker Volume: {speaker_volume:.1f}")
        print("   (Use --volume=0.0-1.0, --quiet, or --loud)")
    else:
        print("📝 Text Mode: Questions will be displayed as text only")    
    # Show aggressiveness level
    if voice_aggressiveness <= 0.3:
        agg_desc = "Conservative (fewer false positives)"
    elif voice_aggressiveness <= 0.6:
        agg_desc = "Balanced"
    else:
        agg_desc = "Aggressive (catch hostile speakers faster)"
    
    print(f"🎯 Voice Aggressiveness: {voice_aggressiveness:.1f} ({agg_desc})")
    print("   (Use --conservative, --balanced, --aggressive, or --aggressiveness=0.0-1.0)")

    # Create orchestrator using configuration
    orchestrator = InterviewOrchestrator(
        project_id=config.google_cloud_project, 
        use_tts=use_tts, 
        credentials_json=config.google_application_credentials,
        voice_aggressiveness=voice_aggressiveness,
        # Use other config values
        location=config.vertex_location,
        model_name=config.model_name,
        max_questions=config.max_questions,
        workdir=config.workdir,
        language_code=config.language_code,
        enable_voice_profiles=config.enable_voice_profiles,
        voice_profiles_dir=config.voice_profiles_dir,
        speaker_similarity_threshold=config.speaker_similarity_threshold,
        tts_voice=config.tts_voice,
        tts_rate=config.tts_rate_wpm,
        tts_pitch=config.tts_pitch,
        tts_amplitude=config.tts_amplitude,
        speaker_volume=speaker_volume,
        interviewer_directness=config.interviewer_directness,
        interviewer_curiosity=config.interviewer_curiosity,
        interviewer_skepticism=config.interviewer_skepticism,
        interviewer_engagement=config.interviewer_engagement,
        interviewer_tolerance=config.interviewer_tolerance,
        interviewer_personality_context=config.interviewer_personality_context
    )
    
    # Run the interview
    result = orchestrator.run(per_turn_seconds=config.per_turn_seconds)
    
    # Results are already displayed by the run() method
    # Detailed information is in the log file


if __name__ == "__main__":
    main()
