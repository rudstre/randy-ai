"""
Text-to-speech functionality using Google Cloud TTS.
"""
import os
import subprocess
import tempfile
import time
import logging
from typing import Optional

from ....config import (
    TTS_PITCH, TTS_AMPLITUDE, SPEAKER_DEVICE,
    SPEAKER_SAMPLE_RATE, SPEAKER_BUFFER_SIZE
)
from ....config import TTS_RATE_WPM, TTS_VOICE, SPEAKER_VOLUME

try:
    from ....utils import with_suppressed_audio_warnings
except ImportError:
    # Fallback decorator if import_utils not available
    def with_suppressed_audio_warnings(func):
        return func

logger = logging.getLogger("speech_tts")

# Global speaker instance (will be initialized when needed)
_speaker_instance = None


@with_suppressed_audio_warnings
def initialize_speaker(output_device: int = SPEAKER_DEVICE, 
                      num_channels: int = 1, 
                      sample_rate: int = SPEAKER_SAMPLE_RATE,
                      buffer_size: int = SPEAKER_BUFFER_SIZE,
                      volume: float = SPEAKER_VOLUME) -> None:
    """Initialize the NB3 Speaker for TTS output."""
    global _speaker_instance
    
    if _speaker_instance is not None:
        return  # Already initialized
    
    try:
        from ..hardware.speaker import Speaker
        _speaker_instance = Speaker(
            device=output_device,
            num_channels=num_channels, 
            format='int16',
            sample_rate=sample_rate,
            buffer_size_samples=buffer_size
        )
        # Set the volume after initialization
        _speaker_instance.volume = volume
        _speaker_instance.start()
        logger.info(f"Initialized NB3 Speaker on device {output_device} with volume {volume:.2f}")
    except Exception as e:
        logger.warning(f"Failed to initialize NB3 Speaker: {e}")
        _speaker_instance = None


def set_speaker_volume(volume: float) -> bool:
    """
    Set the speaker volume dynamically.
    
    Args:
        volume: Volume level (0.0 to 1.0)
        
    Returns:
        True if volume was set successfully, False otherwise
    """
    global _speaker_instance
    
    # Clamp volume to valid range
    volume = max(0.0, min(1.0, volume))
    
    if _speaker_instance:
        _speaker_instance.volume = volume
        logger.info(f"Speaker volume set to {volume:.2f}")
        return True
    else:
        logger.warning("Cannot set volume: Speaker not initialized")
        return False


def get_speaker_volume() -> Optional[float]:
    """
    Get the current speaker volume.
    
    Returns:
        Current volume level (0.0 to 1.0) or None if speaker not initialized
    """
    global _speaker_instance
    
    if _speaker_instance:
        return _speaker_instance.volume
    else:
        return None


def cleanup_speaker() -> None:
    """Clean up the speaker instance."""
    global _speaker_instance
    if _speaker_instance:
        try:
            _speaker_instance.stop()
        except Exception as e:
            logger.warning(f"Error stopping speaker: {e}")
        _speaker_instance = None


def pause_speaker() -> bool:
    """
    Temporarily pause the speaker for half-duplex I2S microphone recording.
    Returns True if speaker was running and is now paused.
    """
    global _speaker_instance
    
    if _speaker_instance is not None:
        try:
            _speaker_instance.stop()
            logger.debug("Paused NB3 Speaker for microphone recording")
            return True
        except Exception as e:
            logger.warning(f"Failed to pause speaker: {e}")
    
    return False


def resume_speaker() -> None:
    """
    Resume the speaker after microphone recording (for half-duplex I2S).
    Only resumes if the speaker was previously initialized.
    """
    global _speaker_instance
    
    if _speaker_instance is not None:
        try:
            _speaker_instance.start()
            logger.debug("Resumed NB3 Speaker after microphone recording")
        except Exception as e:
            logger.warning(f"Failed to resume speaker: {e}")
            # If resume fails, try to reinitialize
            cleanup_speaker()
            initialize_speaker()


@with_suppressed_audio_warnings
def tts_say(text: str, rate_wpm: int = TTS_RATE_WPM, voice: str = TTS_VOICE, 
           pitch: int = 50, amplitude: int = 100, use_speaker: bool = True) -> None:
    """
    High-quality Google Cloud Text-to-Speech through NB3 Speaker.
    """
    global _speaker_instance
    
    if not text.strip():
        return
    
    try:
        from google.cloud import texttospeech
        
        # Initialize Google TTS client
        client = texttospeech.TextToSpeechClient()
        
        # Configure the text input
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Build the voice request - use the specified voice
        voice_params = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name=voice  # Use the voice parameter from function arguments
        )
        
        # Configure audio output
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000
        )
        
        # Generate speech
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice_params, audio_config=audio_config
        )
        
        if use_speaker and _speaker_instance is None:
            initialize_speaker()
        
        if use_speaker and _speaker_instance:
            # Save to temporary WAV file and play through NB3 Speaker
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                wav_path = tmp_file.name
                tmp_file.write(response.audio_content)
            
            try:
                _speaker_instance.play_wav(wav_path)
                while _speaker_instance.is_playing():
                    time.sleep(0.1)
            finally:
                try:
                    os.unlink(wav_path)
                except:
                    pass
        else:
            # Save and play directly (fallback)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                wav_path = tmp_file.name
                tmp_file.write(response.audio_content)
            
            try:
                # Try macOS audio player first
                subprocess.run(["afplay", wav_path], check=True, capture_output=True)
            except FileNotFoundError:
                try:
                    # Fall back to Linux audio player
                    subprocess.run(["aplay", wav_path], check=True, capture_output=True)
                except (FileNotFoundError, subprocess.CalledProcessError):
                    print(f"🤖 {text}")  # Final fallback
            except subprocess.CalledProcessError:
                print(f"🤖 {text}")  # Final fallback
            finally:
                try:
                    os.unlink(wav_path)
                except:
                    pass
                    
    except Exception as e:
        logger.error(f"Google TTS failed: {e}")
        print(f"🤖 {text}")  # Simple text fallback


def test_voice_options():
    """Test different Google Cloud TTS voices with sample phrases."""
    # Sample voices to test (most popular ones)
    test_voices = [
        ("en-US-Neural2-F", "Natural US Female (default)"),
        ("en-US-Neural2-A", "Deep US Male"),
        ("en-US-Neural2-C", "Warm US Male"),
        ("en-GB-Neural2-A", "British Female"),
        ("en-GB-Neural2-B", "British Male"),
        ("en-AU-Neural2-A", "Australian Female"),
        ("en-US-Neural2-J", "News Anchor Style"),
    ]
    
    test_phrase = "Hello! I am your interviewing robot. How do I sound with this voice?"
    
    print("🎤 Testing Google Cloud TTS Voices...")
    print("=" * 50)
    
    for voice_id, description in test_voices:
        print(f"\n🔊 Testing: {description}")
        print(f"   Voice ID: {voice_id}")
        print(f"   Phrase: {test_phrase}")
        
        try:
            # Use the voice directly in tts_say call
            from google.cloud import texttospeech
            
            client = texttospeech.TextToSpeechClient()
            input_text = texttospeech.SynthesisInput(text=test_phrase)
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name=voice_id
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000
            )
            
            response = client.synthesize_speech(
                input=input_text, voice=voice, audio_config=audio_config
            )
            
            # Save and play through speaker
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(response.audio_content)
                temp_file.flush()
                
                if _speaker_instance:
                    _speaker_instance.play_wav(temp_file.name)
                else:
                    # Fallback to aplay
                    subprocess.run(["aplay", "-q", temp_file.name], 
                                 capture_output=True, check=False)
                
                os.unlink(temp_file.name)
            
            time.sleep(1)  # Brief pause between voices
            
        except Exception as e:
            print(f"   ❌ Error with {voice_id}: {e}")
        
        input("   Press Enter for next voice...")
    
    print("✅ Voice testing complete!")
    print("\nTo use a voice permanently:")
    print("1. Edit config.py")
    print("2. Change DEFAULT_TTS_VOICE to your preferred voice ID")
