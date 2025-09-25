"""
Audio capture functionality with Voice Activity Detection.
"""
import os
import re
import subprocess
import time
import logging
from typing import Dict, Optional, Any

import numpy as np

from ....config import (
    CHANNELS, SAMPLE_RATE_CAPTURE, FRAME_MS,
    MAX_SAMPLES_30S, MIC_GAIN, SAMPLE_RATE_TARGET,
    TARGET_RMS, VAD_SILENCE_THRESHOLD, VAD_SILENCE_DURATION,
    VAD_MIN_SPEECH_DURATION, ALSA_LIB_NAMES
)
from .processing import stereo_to_mono, remove_dc, resample_48k_to_16k, normalize_audio, write_wav
from ....utils import import_quietly, with_suppressed_audio_warnings

logger = logging.getLogger("audio_capture")


# NB3 Microphone will be imported lazily when needed
# This avoids pyaudio dependency issues when not on RPi


@with_suppressed_audio_warnings 
def get_best_microphone_config():
    """
    Automatically detect the best microphone device and parameters.
    Returns (device_index, channels, sample_rate) tuple.
    """
    try:
        # Use NB3 utilities to find I2S device by name
        try:
            from ..hardware import utilities
            device_index = utilities.get_input_device_by_name("MAX98357A")
            if device_index != -1:
                logger.info(f"Found MAX98357A device at index {device_index}")
            else:
                # Try other I2S device names
                for name in ["bcm2835-i2s", "i2s", "sph0645"]:
                    device_index = utilities.get_input_device_by_name(name)
                    if device_index != -1:
                        logger.info(f"Found I2S device '{name}' at index {device_index}")
                        break
        except Exception as e:
            logger.warning(f"NB3 device detection failed: {e}")
            device_index = -1
        
        # If NB3 detection failed, use PyAudio directly
        if device_index == -1:
            import pyaudio
            pa = pyaudio.PyAudio()
            
            for i in range(pa.get_device_count()):
                try:
                    info = pa.get_device_info_by_index(i)
                    max_input_channels = info.get('maxInputChannels', 0)
                    device_name = info.get('name', '')
                    if isinstance(max_input_channels, (int, float)) and max_input_channels > 0 and isinstance(device_name, str):
                        device_name_lower = device_name.lower()
                        if any(keyword in device_name_lower for keyword in ['max98357a', 'sph0645', 'i2s', 'bcm2835-i2s']):
                            device_index = i
                            logger.info(f"Found I2S device at PyAudio index {i}: {info['name']}")
                            break
                except Exception:
                    continue
            pa.terminate()
        
        # If still no device found, use default
        if device_index == -1:
            logger.info("No I2S device found, using default device 1")
            device_index = 1
        
        # Now get the device's supported parameters
        import pyaudio
        pa = pyaudio.PyAudio()
        try:
            info = pa.get_device_info_by_index(device_index)
            logger.info(f"Device {device_index} info: {info['name']}")
            logger.info(f"Max input channels: {info['maxInputChannels']}")
            logger.info(f"Default sample rate: {info['defaultSampleRate']}")
            
            # Use device's actual capabilities, but override with known working values for I2S
            max_input_channels = info.get('maxInputChannels', 2)
            channels = min(2, int(max_input_channels))  # Use device's reported max channels
            
            # For I2S devices, use known working sample rate (48000) instead of reported default
            device_name = info.get('name', '')
            if isinstance(device_name, str) and any(keyword in device_name.lower() for keyword in ['max98357a', 'sph0645', 'i2s', 'bcm2835-i2s']):
                sample_rate = 48000  # Known working rate for I2S devices
                logger.info(f"I2S device detected, using 48000 Hz (known working rate)")
            else:
                default_sample_rate = info.get('defaultSampleRate', 44100)
                sample_rate = int(default_sample_rate)  # Use device's preferred rate for other devices
            
            logger.info(f"Using {channels} channel(s) at {sample_rate} Hz")
            
            pa.terminate()
            return device_index, channels, sample_rate
            
        except Exception as e:
            logger.warning(f"Failed to get device info: {e}")
            pa.terminate()
            # Safe defaults
            return device_index, 2, 48000
            
    except Exception as e:
        logger.error(f"Microphone configuration detection failed: {e}")
        # Ultimate fallback
        return 1, 2, 48000


class AudioCapture:
    """Handles audio capture with the NB3 Microphone class and Voice Activity Detection."""
    
    def __init__(self, 
                 input_device: Optional[int] = None,
                 num_channels: int = CHANNELS,
                 sr_capture: int = SAMPLE_RATE_CAPTURE,
                 frame_ms: int = FRAME_MS,
                 max_samples: int = MAX_SAMPLES_30S,
                 mic_gain: float = MIC_GAIN,
                 sr_target: int = SAMPLE_RATE_TARGET,
                 target_rms: float = TARGET_RMS,
                 silence_threshold: float = VAD_SILENCE_THRESHOLD,
                 silence_duration: float = VAD_SILENCE_DURATION,
                 min_speech_duration: float = VAD_MIN_SPEECH_DURATION):
        
        # Auto-detect best microphone configuration if not specified
        if input_device is None:
            detected_device, detected_channels, detected_rate = get_best_microphone_config()
            self.input_device = detected_device
            # Override defaults with detected values
            if num_channels == CHANNELS:  # Only override if using default
                self.num_channels = detected_channels
            else:
                self.num_channels = num_channels
            if sr_capture == SAMPLE_RATE_CAPTURE:  # Only override if using default
                self.sr_capture = detected_rate
            else:
                self.sr_capture = sr_capture
        else:
            self.input_device = input_device
            self.num_channels = num_channels
            self.sr_capture = sr_capture
        # Use buffer size that matches working test files (sample_rate / 10) for I2S devices
        if self.input_device == 1 and self.sr_capture == 48000:
            self.frame_size = int(self.sr_capture / 10)  # 4800 samples for I2S at 48kHz (matches working test_input.py)
            logger.info(f"Using I2S buffer size: {self.frame_size} samples")
        else:
            self.frame_size = int(self.sr_capture * frame_ms / 1000)  # Original calculation for other devices
        self.max_samples = max_samples
        self.mic_gain = mic_gain
        self.sr_target = sr_target
        self.target_rms = target_rms
        
        # Voice Activity Detection settings
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration

    @with_suppressed_audio_warnings
    def capture_turn(self, max_seconds: float, output_dir: str, turn_idx: int) -> Dict[str, Any]:
        """
        Capture audio with Voice Activity Detection - stops when user stops talking.
        
        Args:
            max_seconds: Maximum time to wait (timeout)
            output_dir: Directory to save audio files
            turn_idx: Turn number for file naming
            
        Returns:
            Dict with audio file paths and processed data
        """
        logger.info(f"Attempting to open microphone with:")
        logger.info(f"  Device: {self.input_device}")
        logger.info(f"  Channels: {self.num_channels}")
        logger.info(f"  Sample Rate: {self.sr_capture}")
        logger.info(f"  Frame Size: {self.frame_size}")
        logger.info(f"  Format: int32")
        
        # Half-duplex I2S support: pause speaker before recording
        from ..speech.tts import pause_speaker, resume_speaker
        speaker_was_paused = pause_speaker()
        if speaker_was_paused:
            logger.info("Paused TTS speaker for half-duplex I2S recording")
        
        mic = None
        # Try to open microphone with retry for transient issues
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt + 1}/{max_retries}")
                    time.sleep(0.5)  # Brief delay between retries
                
                # Lazy import of NB3 Microphone to avoid pyaudio dependency issues
                try:
                    from ..hardware.microphone import Microphone
                except Exception as e:
                    raise RuntimeError(f"Failed to import NB3 microphone: {e}")
                
                mic = Microphone(self.input_device, self.num_channels, 'int32',
                               self.sr_capture, self.frame_size, self.max_samples)
                mic.gain = self.mic_gain
                mic.start()
                logger.info("Microphone opened successfully")
                break  # Success, exit retry loop
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed to open microphone: {e}")
                if attempt == max_retries - 1:  # Last attempt
                    # Resume speaker before re-raising error
                    if speaker_was_paused:
                        resume_speaker()
                        logger.info("Resumed TTS speaker after microphone failed to open")
                    raise e  # Re-raise the error
                # Continue to next retry attempt
        
        # Voice Activity Detection variables
        start_time = time.time()
        last_speech_time = start_time
        speech_started = False
        silence_start_time = None
        
        # Real-time audio analysis buffer
        check_interval = 0.1  # Check every 100ms
        
        print("   🔊 Speak now...")
        
        # Ensure microphone was successfully opened
        if mic is None:
            raise RuntimeError("Failed to open microphone after all retry attempts")
            
        try:
            while True:
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                # Timeout check
                if elapsed_time > max_seconds:
                    print(f"   ⏰ Timeout after {max_seconds:.1f}s")
                    break
                
                # Get current audio level
                current_samples = mic.valid_samples
                if current_samples > 0:
                    # Get recent audio data for analysis
                    recent_samples = min(int(self.sr_capture * 0.5), current_samples)  # Last 0.5 seconds
                    recent_data = mic.sound[current_samples - recent_samples:current_samples, :]
                    
                    if recent_data.size > 0:
                        # Calculate RMS of recent audio
                        mono_recent = stereo_to_mono(recent_data) if self.num_channels > 1 else recent_data.flatten()
                        rms_level = np.sqrt(np.mean(mono_recent**2))
                        
                        # Voice activity detection
                        is_speaking = rms_level > self.silence_threshold
                        
                        if is_speaking:
                            last_speech_time = current_time
                            if not speech_started:
                                speech_started = True
                                print(f"   🗣️  Speech detected! (level: {rms_level:.4f})")
                            silence_start_time = None
                        else:
                            # Check for end of speech
                            if speech_started:
                                if silence_start_time is None:
                                    silence_start_time = current_time
                                    print(f"   🤫 Silence started (level: {rms_level:.4f})")
                                else:
                                    silence_duration = current_time - silence_start_time
                                    if silence_duration >= self.silence_duration:
                                        speech_duration = last_speech_time - start_time
                                        if speech_duration >= self.min_speech_duration:
                                            print(f"   ✅ Speech ended (spoke for {speech_duration:.1f}s)")
                                            break
                                        else:
                                            # Reset if speech was too short
                                            print(f"   ⚠️  Speech too short ({speech_duration:.1f}s), continuing...")
                                            speech_started = False
                                            silence_start_time = None
                
                time.sleep(check_interval)
                
        finally:
            if mic:
                mic.stop()
            # Resume speaker after recording (if it was paused for half-duplex I2S)
            if speaker_was_paused:
                resume_speaker()
                logger.info("Resumed TTS speaker after recording")

        # Get final audio data
        data = mic.sound[:mic.valid_samples, :]
        if data.size == 0:
            print("   ⚠️  No audio captured")
            raise RuntimeError("No audio captured")

        total_duration = (mic.valid_samples / self.sr_capture)
        print(f"   📊 Captured {total_duration:.1f}s of audio")

        # Process audio
        mono = stereo_to_mono(data) if self.num_channels > 1 else data.flatten()
        mono = remove_dc(mono)
        y16k = resample_48k_to_16k(mono)
        y16k = normalize_audio(y16k, self.target_rms)
        
        # Convert to PCM16
        pcm16_16k = np.clip(y16k * 32767, -32768, 32767).astype(np.int16)
        raw48_pcm16 = np.clip(data * 32767, -32768, 32767).astype(np.int16)

        # Write files with unique timestamp to prevent collision across conversations
        timestamp = int(time.time() * 1000)  # millisecond timestamp
        raw48_path = os.path.join(output_dir, f"turn{turn_idx:02d}_{timestamp}_raw48.wav")
        stt16_path = os.path.join(output_dir, f"turn{turn_idx:02d}_{timestamp}_stt16.wav")
        
        write_wav(raw48_path, raw48_pcm16, self.sr_capture, channels=self.num_channels)
        write_wav(stt16_path, pcm16_16k, self.sr_target, channels=1)

        return {
            "raw48_path": raw48_path,
            "stt16_path": stt16_path,
            "stt16_bytes": pcm16_16k.tobytes()
        }
