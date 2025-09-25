"""
Service classes for the interview system.
"""
import os
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from .models import Turn, InterviewResult
from .schemas import InterviewState, PersonalityTraits, InterviewContext
from ..infrastructure.audio import extract_voice_features, tts_say, recognize_google_sync, initialize_speaker, cleanup_speaker
from ..infrastructure.data import PersonManager
from ..infrastructure.audio.voice_id import ProgressiveVoiceIdentifier
from ..config import CANONICAL_FEATURE_MAPPING

logger = logging.getLogger("services")


class AudioInterviewService:
    """Handles audio capture and processing for interviews."""
    
    def __init__(self, audio_capture, language_code: str = "en-US"):
        self.audio_capture = audio_capture
        self.language_code = language_code
    
    def capture_and_process_turn(self, 
                                question: str,
                                turn_idx: int, 
                                conversation_dir: str, 
                                per_turn_seconds: float) -> Turn:
        """
        Capture audio, process speech recognition, and extract features.
        
        Args:
            question: The question that was asked
            turn_idx: Turn index number
            conversation_dir: Directory to save audio files
            per_turn_seconds: Maximum seconds to record
            
        Returns:
            Turn object with all processed data
            
        Raises:
            RuntimeError: If audio processing fails
        """
        print("🎧 Listening (will stop when you finish speaking)...")
        try:
            # Capture audio
            audio_data = self.audio_capture.capture_turn(per_turn_seconds, conversation_dir, turn_idx)
            logger.info(f"Audio captured: {audio_data['stt16_path']}")
            
            # Speech recognition
            print("🔍 Processing speech...")
            transcript = recognize_google_sync(
                audio_data["stt16_bytes"], 
                sr_hz=self.audio_capture.sr_target, 
                language=self.language_code
            )
            logger.info(f"Speech recognition result: {transcript or '(empty)'}")
            
            # Voice feature extraction
            print("🎵 Analyzing voice features...")
            features = extract_voice_features(audio_data["stt16_path"])
            logger.info(f"Voice features extracted: {features}")

            print(f"💬 \"{transcript or '(no speech detected)'}\"")

            # Create turn record
            turn = Turn(
                idx=turn_idx,
                question=question,
                wav_path=audio_data["stt16_path"],
                transcript=transcript,
                features=features
            )
            
            return turn
            
        except Exception as e:
            logger.error("Failed to process turn %d: %s", turn_idx, e)
            print("❌ Processing failed - check log for details")
            
            # Create minimal turn record for error case
            return Turn(
                idx=turn_idx,
                question=question,
                wav_path="",
                transcript="",
                features={k: 0.0 for k in CANONICAL_FEATURE_MAPPING.keys()}
            )


class PersonIdentificationService:
    """Handles person identification and voice profiling."""
    
    def __init__(self, 
                 person_manager: Optional[PersonManager],
                 progressive_identifier: Optional[ProgressiveVoiceIdentifier]):
        self.person_manager = person_manager
        self.progressive_identifier = progressive_identifier
    
    def identify_person_from_turn(self, turn: Turn, turn_idx: int) -> Dict[str, Any]:
        """
        Identify person from a turn and return identification results.
        
        Args:
            turn: Turn object with voice features
            turn_idx: Turn index number
            
        Returns:
            Dict with identification results
        """
        if not self.progressive_identifier:
            return {
                "person_id": None,
                "confidence": 0.0,
                "should_welcome": False,
                "should_terminate": False,
                "reasoning": "Voice identification disabled"
            }
        
        id_result = self.progressive_identifier.identify_from_turn(turn, turn_idx)
        
        print(f"🎯 {id_result.reasoning}")
        
        return {
            "person_id": id_result.speaker_id,
            "confidence": id_result.confidence,
            "should_welcome": id_result.should_welcome,
            "should_terminate": id_result.should_terminate,
            "reasoning": id_result.reasoning,
            "disposition": id_result.disposition
        }
    
    def welcome_returning_person(self, person_id: str, use_tts: bool = True) -> bool:
        """
        Welcome a returning person.
        
        Args:
            person_id: ID of the person to welcome
            use_tts: Whether to use text-to-speech
            
        Returns:
            True if welcome was performed
        """
        if not self.person_manager:
            return False
            
        person_info = self.person_manager.get_person_info(person_id)
        person_name = person_info.get('name', 'friend') if person_info else 'friend'
        welcome_msg = f"Welcome back, {person_name}! It's nice to see you again."
        
        print(f"👋 {welcome_msg}")
        if use_tts:
            tts_say(welcome_msg)
        
        return True
    
    def process_final_speaker_profile(self, 
                                    turns: List[Turn], 
                                    state: InterviewState,
                                    final_payload: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str], bool, float]:
        """
        Process speaker profile at end of interview.
        
        Returns:
            Tuple of (speaker_id, speaker_name, is_returning_speaker, confidence)
        """
        if not self.person_manager or not turns:
            return None, None, False, 0.0
        
        # Use progressive identification results if available
        if state.identified_speaker:
            speaker_id = state.identified_speaker
            is_returning_speaker = True
            final_speaker_confidence = state.speaker_confidence
            
            # Get speaker info
            speaker_info = self.person_manager.get_speaker_info(speaker_id)
            speaker_name = speaker_info.get("name", "Unknown") if speaker_info else "Unknown"
            
            # Update existing profile
            self._update_existing_profile(turns, speaker_id, final_speaker_confidence, final_payload, state)
            
            if not state.was_welcomed:  # Only print if we didn't already welcome them
                print(f"👤 Identified returning speaker: {speaker_name} (confidence: {final_speaker_confidence:.2f})")
            
            return speaker_id, speaker_name, is_returning_speaker, final_speaker_confidence
        
        else:
            # Create speaker profile
            speaker_id = self._create_new_speaker_profile(turns, final_payload, state)
            
            if speaker_id:
                speaker_info = self.person_manager.get_speaker_info(speaker_id)
                speaker_name = speaker_info.get("name", "Unknown") if speaker_info else "Unknown"
                
                print(f"👤 Speaker detected, created profile: {speaker_name}")
                logger.info(f"Created voice profile: {speaker_id} ({speaker_name})")
                
                return speaker_id, speaker_name, False, 1.0
        
        return None, None, False, 0.0
    
    def _update_existing_profile(self, turns: List[Turn], speaker_id: str, 
                               confidence: float, final_payload: Optional[Dict[str, Any]], 
                               state: InterviewState):
        """Update existing speaker profile with conversation data."""
        if not self.person_manager:
            return
            
        conversation_features = self.person_manager.aggregate_conversation_features(turns)
        if conversation_features and final_payload:
            individual_turn_features = [turn.features for turn in turns if hasattr(turn, 'features') and turn.features]
            
            conversation_summary = f"Interview on {datetime.now().strftime('%Y-%m-%d %H:%M')} - {final_payload.get('opinion_word', 'neutral')} opinion"
            # Create conversation record for the update
            from ..infrastructure.data.conversations import ConversationRecord, ConversationTurn
            
            # Convert turns to ConversationTurn objects
            conversation_turns = []
            for i, turn in enumerate(turns):
                if hasattr(turn, 'features') and turn.features:
                    conv_turn = ConversationTurn(
                        turn_idx=i,
                        question=getattr(turn, 'question', ''),
                        transcript=getattr(turn, 'transcript', ''),
                        timestamp=datetime.now().isoformat(),
                        voice_features=turn.features,
                        duration_seconds=getattr(turn, 'duration_seconds', None)
                    )
                    conversation_turns.append(conv_turn)
            
            # Create conversation record
            conversation_record = ConversationRecord(
                conversation_id=f"conv_{int(datetime.now().timestamp())}",
                date_time=datetime.now().isoformat(),
                duration_minutes=None,
                turns=conversation_turns,
                final_opinion_word=final_payload.get('opinion_word', 'neutral'),
                final_score_overall=final_payload.get('score_overall', 0.0),
                aggregated_voice_features=conversation_features
            )
            
            self.person_manager.update_person_profile(
                speaker_id, conversation_features, conversation_record
            )
            
            logger.info(f"Updated profile for returning speaker: {speaker_id} with confidence {confidence:.3f}")
    
    def _create_new_speaker_profile(self, turns: List[Turn], 
                                  final_payload: Optional[Dict[str, Any]], 
                                  state: InterviewState) -> Optional[str]:
        """Create speaker profile."""
        if not self.person_manager:
            return None
            
        conversation_features = self.person_manager.aggregate_conversation_features(turns)
        if not conversation_features:
            return None
        
        # Prepare conversation metadata
        conversation_summary = ""
        opinion_word = ""
        overall_score = 0.0
        
        if final_payload:
            conversation_summary = f"First meeting on {datetime.now().strftime('%Y-%m-%d %H:%M')} - {final_payload.get('opinion_word', 'neutral')} opinion"
            opinion_word = final_payload.get('opinion_word', 'neutral')
            overall_score = final_payload.get('score_overall', 0.0)
        
        # Extract individual turn features
        individual_turn_features = [turn.features for turn in turns if hasattr(turn, 'features') and turn.features]
        
        # Create conversation record for new profile
        from ..infrastructure.data.conversations import ConversationRecord, ConversationTurn
        
        # Convert turns to ConversationTurn objects
        conversation_turns = []
        for i, turn in enumerate(turns):
            if hasattr(turn, 'features') and turn.features:
                conv_turn = ConversationTurn(
                    turn_idx=i,
                    question=getattr(turn, 'question', ''),
                    transcript=getattr(turn, 'transcript', ''),
                    timestamp=datetime.now().isoformat(),
                    voice_features=turn.features,
                    duration_seconds=getattr(turn, 'duration_seconds', None)
                )
                conversation_turns.append(conv_turn)
        
        # Create conversation record
        conversation_record = ConversationRecord(
            conversation_id=f"conv_{int(datetime.now().timestamp())}",
            date_time=datetime.now().isoformat(),
            duration_minutes=None,
            turns=conversation_turns,
            final_opinion_word=opinion_word,
            final_score_overall=overall_score,
            aggregated_voice_features=conversation_features
        )
        
        # Create profile
        speaker_id = self.person_manager.create_new_person_profile(
            conversation_features,
            conversation_record=conversation_record,
            person_name=state.extracted_speaker_name
        )
        
        return speaker_id
    
    def start_new_conversation(self):
        """Initialize progressive identification for new conversation."""
        if self.progressive_identifier and self.person_manager:
            self.progressive_identifier.start_new_conversation()
            known_speakers = self.person_manager.list_all_persons()
            if known_speakers:
                print(f"📚 I know {len(known_speakers)} people from previous conversations")


class TTSService:
    """Handles text-to-speech functionality."""
    
    def __init__(self, 
                 use_tts: bool = True,
                 voice: str = "en-US-Neural2-F",
                 rate_wpm: int = 180,
                 pitch: int = 55,
                 amplitude: int = 120,
                 speaker_volume: Optional[float] = None):
        self.use_tts = use_tts
        self.voice = voice
        self.rate_wpm = rate_wpm
        self.pitch = pitch
        self.amplitude = amplitude
        
        # Import config default if not provided
        if speaker_volume is None:
            from ..config import DEFAULT_SPEAKER_VOLUME
            speaker_volume = DEFAULT_SPEAKER_VOLUME
        self.speaker_volume = speaker_volume
        
        if self.use_tts:
            initialize_speaker(volume=self.speaker_volume)
    
    def speak_or_print(self, message: str, prefix: str = "🤖"):
        """Speak message via TTS or print if TTS disabled."""
        if self.use_tts:
            tts_say(message, 
                   rate_wpm=self.rate_wpm,
                   voice=self.voice,
                   pitch=self.pitch,
                   amplitude=self.amplitude)
        else:
            print(f"{prefix} {message}")
    
    def set_volume(self, volume: float) -> bool:
        """
        Set speaker volume.
        
        Args:
            volume: Volume level (0.0 to 1.0)
            
        Returns:
            True if successful, False otherwise
        """
        if self.use_tts:
            from ..infrastructure.audio.speech.tts import set_speaker_volume
            success = set_speaker_volume(volume)
            if success:
                self.speaker_volume = volume
            return success
        else:
            # Just store the value for when TTS is enabled
            self.speaker_volume = max(0.0, min(1.0, volume))
            return True
    
    def get_volume(self) -> float:
        """
        Get current speaker volume.
        
        Returns:
            Current volume level (0.0 to 1.0)
        """
        if self.use_tts:
            from ..infrastructure.audio.speech.tts import get_speaker_volume
            current_volume = get_speaker_volume()
            if current_volume is not None:
                return current_volume
        return self.speaker_volume
    
    def cleanup(self):
        """Cleanup TTS resources."""
        if self.use_tts:
            cleanup_speaker()


class ConversationManager:
    """Manages conversation flow and metadata."""
    
    def __init__(self, workdir: str):
        self.workdir = workdir
    
    def create_conversation_workspace(self) -> Tuple[str, str]:
        """
        Create unique conversation directory.
        
        Returns:
            Tuple of (conversation_id, conversation_dir)
        """
        conversation_id = f"conv_{int(time.time())}"
        conversation_dir = os.path.join(self.workdir, conversation_id)
        os.makedirs(conversation_dir, exist_ok=True)
        return conversation_id, conversation_dir
    
    def aggregate_conversation_features(self, turns: List[Turn]) -> Dict[str, float]:
        """Aggregate voice features across all turns in conversation."""
        if not turns:
            return {}
        
        # Extract all valid feature dictionaries
        valid_features = [turn.features for turn in turns if turn.features]
        if not valid_features:
            return {}
        
        # Get all feature keys from first valid feature set
        feature_keys = valid_features[0].keys()
        aggregated = {}
        
        # Calculate mean for each feature
        for key in feature_keys:
            values = [features.get(key, 0.0) for features in valid_features if key in features]
            if values:
                aggregated[key] = sum(values) / len(values)
            else:
                aggregated[key] = 0.0
        
        return aggregated
    
    def build_interview_context(self, 
                               turns: List[Turn], 
                               remaining_questions: int,
                               state: InterviewState,
                               personality_traits: PersonalityTraits) -> InterviewContext:
        """Build interview context for decision making."""
        acoustic_features = self.aggregate_conversation_features(turns)
        
        return InterviewContext(
            turns=turns,
            remaining_questions=remaining_questions,
            state=state,
            personality_traits=personality_traits,
            acoustic_features_aggregate=acoustic_features
        )
