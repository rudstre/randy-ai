"""
Refactored interview orchestrator using service-based architecture.
"""
import os
import logging
import time
from typing import Optional, List, Dict, Any
from datetime import datetime

from .models import Turn, InterviewResult
from .schemas import InterviewState, PersonalityTraits, InterviewContext
from .services import (
    AudioInterviewService, 
    PersonIdentificationService, 
    TTSService, 
    ConversationManager
)
from .decision_engine import InterviewDecisionEngine, PromptEngine
from .events import (
    InterviewEventBus, EventLogger, InterviewMetrics,
    InterviewStartedEvent, TurnCompletedEvent, SpeakerIdentifiedEvent,
    SpeakerWelcomedEvent, HostileBehaviorDetectedEvent, 
    InterviewTerminatedEvent, InterviewCompletedEvent,
    DecisionMadeEvent, ErrorOccurredEvent
)
# AudioCapture imported lazily to avoid dependency issues
from ..infrastructure.llm import VertexRestClient
from ..infrastructure.data import PersonManager
from ..infrastructure.audio.voice_id import ProgressiveVoiceIdentifier
from ..utils import setup_logging
from ..config import (
    DEFAULT_VERTEX_LOCATION, DEFAULT_MODEL_NAME, DEFAULT_CHANNELS,
    DEFAULT_SAMPLE_RATE_CAPTURE, DEFAULT_FRAME_MS, DEFAULT_MAX_SAMPLES_30S,
    DEFAULT_MIC_GAIN, DEFAULT_SAMPLE_RATE_TARGET, DEFAULT_TARGET_RMS,
    DEFAULT_MAX_QUESTIONS, DEFAULT_WORKDIR, DEFAULT_LANGUAGE_CODE,
    DEFAULT_ENABLE_VOICE_PROFILES, DEFAULT_VOICE_PROFILES_DIR,
    DEFAULT_SPEAKER_SIMILARITY_THRESHOLD, DEFAULT_TTS_VOICE,
    DEFAULT_TTS_RATE_WPM, DEFAULT_TTS_PITCH, DEFAULT_TTS_AMPLITUDE, DEFAULT_SPEAKER_VOLUME,
    DEFAULT_VOICE_AGGRESSIVENESS, DEFAULT_PER_TURN_SECONDS
)
from ..config import PersonalityConfig

logger = logging.getLogger("orchestrator")


class InterviewOrchestrator:
    """
    AI interview orchestrator using service-based architecture.
    
    This system breaks down interview functionality into focused services
    and uses event-driven communication between components for better
    maintainability, testability, and extensibility.
    """
    
    def __init__(self,
                 project_id: Optional[str] = None,
                 location: str = DEFAULT_VERTEX_LOCATION,
                 model_name: str = DEFAULT_MODEL_NAME,
                 input_device: Optional[int] = None,
                 num_channels: int = DEFAULT_CHANNELS,
                 sr_capture: int = DEFAULT_SAMPLE_RATE_CAPTURE,
                 frame_ms: int = DEFAULT_FRAME_MS,
                 max_samples: int = DEFAULT_MAX_SAMPLES_30S,
                 mic_gain: float = DEFAULT_MIC_GAIN,
                 sr_target: int = DEFAULT_SAMPLE_RATE_TARGET,
                 target_rms: float = DEFAULT_TARGET_RMS,
                 max_questions: int = DEFAULT_MAX_QUESTIONS,
                 workdir: str = DEFAULT_WORKDIR,
                 language_code: str = DEFAULT_LANGUAGE_CODE,
                 use_tts: bool = True,
                 credentials_json: Optional[str] = None,
                 enable_voice_profiles: bool = DEFAULT_ENABLE_VOICE_PROFILES,
                 voice_profiles_dir: str = DEFAULT_VOICE_PROFILES_DIR,
                 speaker_similarity_threshold: float = DEFAULT_SPEAKER_SIMILARITY_THRESHOLD,
                 tts_voice: str = DEFAULT_TTS_VOICE,
                 tts_rate: int = DEFAULT_TTS_RATE_WPM,
                 tts_pitch: int = DEFAULT_TTS_PITCH,
                 tts_amplitude: int = DEFAULT_TTS_AMPLITUDE,
                 speaker_volume: Optional[float] = None,
                 voice_aggressiveness: float = DEFAULT_VOICE_AGGRESSIVENESS,
                 personality_config: Optional['PersonalityConfig'] = None):
        
        # Store core configuration
        self.max_questions = max_questions
        self.log_file = os.path.join(workdir, "interview.log")
        
        # Setup logging
        setup_logging(self.log_file)
        
        # Create personality configuration
        if personality_config is None:
            personality_config = PersonalityConfig()  # Use defaults
        self.personality_traits = personality_config
        
        # Initialize event system
        self.event_bus = InterviewEventBus()
        self.event_logger = EventLogger()
        self.metrics = InterviewMetrics()
        
        # Subscribe to events
        self.event_bus.subscribe_all(self.event_logger.handle_event)
        self.event_bus.subscribe_all(self.metrics.handle_event)
        
        # Initialize conversation manager
        self.conversation_manager = ConversationManager(workdir)
        
        # Initialize audio service (lazy import to avoid pyaudio dependency)
        try:
            from ..infrastructure.audio.processing import AudioCapture
            audio_capture = AudioCapture(
                input_device=input_device,
                num_channels=num_channels,
                sr_capture=sr_capture,
                frame_ms=frame_ms,
                max_samples=max_samples,
                mic_gain=mic_gain,
                sr_target=sr_target,
                target_rms=target_rms
            )
            self.audio_service = AudioInterviewService(audio_capture, language_code)
        except ImportError as e:
            logger.warning(f"AudioCapture not available: {e}")
            self.audio_service = None
        
        # Initialize voice services
        person_manager = None
        progressive_identifier = None
        
        if enable_voice_profiles:
            person_manager = PersonManager(voice_profiles_dir)
            # PersonManager now provides compatibility interface for ProgressiveVoiceIdentifier
            progressive_identifier = ProgressiveVoiceIdentifier(
                person_manager,
                aggressiveness=voice_aggressiveness
            )
        
        self.person_service = PersonIdentificationService(
            person_manager, 
            progressive_identifier
        )
        
        # Initialize TTS service
        if speaker_volume is None:
            speaker_volume = DEFAULT_SPEAKER_VOLUME
        self.tts_service = TTSService(
            use_tts=use_tts,
            voice=tts_voice,
            rate_wpm=tts_rate,
            pitch=tts_pitch,
            amplitude=tts_amplitude,
            speaker_volume=speaker_volume
        )
        
        # Initialize LLM and decision services
        if not project_id:
            raise ValueError("project_id is required for LLM functionality")
        
        self.llm_client = VertexRestClient(
            project=project_id,
            location=location,
            model=model_name,
            credentials_json=credentials_json
        )
        
        self.prompt_engine = PromptEngine(self.personality_traits)
        self.decision_engine = InterviewDecisionEngine(self.llm_client, self.prompt_engine)
    
    def run(self,
            initial_question: Optional[str] = None,
            per_turn_seconds: float = DEFAULT_PER_TURN_SECONDS) -> InterviewResult:
        """
        Run the complete interview process.
        
        Args:
            initial_question: First question to ask
            per_turn_seconds: Maximum seconds per turn
            
        Returns:
            InterviewResult with complete interview data
        """
        # Create conversation workspace
        conversation_id, conversation_dir = self.conversation_manager.create_conversation_workspace()
        
        # Initialize interview state
        state = InterviewState(conversation_id=conversation_id)
        turns: List[Turn] = []
        remaining_questions = self.max_questions
        current_question = initial_question or self.prompt_engine.generate_opening_question()
        final_payload: Optional[Dict[str, Any]] = None
        
        # Emit interview started event
        self.event_bus.emit(InterviewStartedEvent(
            conversation_id or "unknown", time.time(), self.max_questions
        ))
        
        print(f"\n🎙️  Starting interview - {self.max_questions} questions")
        print(f"📝 Detailed logs: {self.log_file}")
        print("=" * 50)
        
        # Start speaker identification
        self.person_service.start_new_conversation()
        
        try:
            # Main interview loop
            while True:
                turn_idx = len(turns) + 1
                
                # Ask question
                self.tts_service.speak_or_print(current_question)
                logger.info(f"Question {turn_idx}/{self.max_questions}: {current_question}")
                
                # Process turn
                turn = self._process_turn(
                    current_question, turn_idx, conversation_dir, 
                    per_turn_seconds, state
                )
                turns.append(turn)
                state.increment_turn()
                
                # Emit turn completed event
                self.event_bus.emit(TurnCompletedEvent(
                    conversation_id or "unknown", time.time(), turn_idx, 
                    turn.transcript, turn.features
                ))
                
                # Check for early termination
                if state.termination_reason:
                    final_payload = self._handle_early_termination(state, turns)
                    break
                
                # Decrement remaining questions
                remaining_questions -= 1
                
                # Make decision about continuation
                decision_result = self._make_continuation_decision(
                    turns, remaining_questions, state
                )
                
                if decision_result["action"] == "ask" and remaining_questions > 0:
                    current_question = decision_result["question"]
                    if decision_result.get("extracted_name") and not state.extracted_speaker_name:
                        state.extracted_speaker_name = decision_result["extracted_name"]
                    continue
                else:
                    # Finalize interview
                    final_payload = decision_result
                    if decision_result.get("extracted_name") and not state.extracted_speaker_name:
                        state.extracted_speaker_name = decision_result["extracted_name"]
                    break
            
            # Process final results
            result = self._build_final_result(turns, state, final_payload, conversation_id)
            
            # Emit completion event
            conv_id = conversation_id or "unknown"
            if state.termination_reason:
                self.event_bus.emit(InterviewTerminatedEvent(
                    conv_id, time.time(), state.termination_reason,
                    len(turns), result.termination_message
                ))
            else:
                self.event_bus.emit(InterviewCompletedEvent(
                    conv_id, time.time(), len(turns),
                    result.final_opinion_word, result.final_score_overall,
                    result.speaker_id
                ))
            
            # Display results
            self._display_results(result)
            
            return result
            
        except Exception as e:
            # Emit error event
            self.event_bus.emit(ErrorOccurredEvent(
                conversation_id or "unknown", time.time(), type(e).__name__,
                str(e), "orchestrator"
            ))
            logger.error("Interview failed with error: %s", e)
            raise
        
        finally:
            # Cleanup
            self.tts_service.cleanup()
            self.conversation_manager.cleanup_conversation_workspace()
    
    def _process_turn(self, question: str, turn_idx: int, conversation_dir: str,
                     per_turn_seconds: float, state: InterviewState) -> Turn:
        """Process a single conversation turn."""
        try:
            # Capture and process audio
            if self.audio_service is None:
                raise RuntimeError("Audio service not available - missing dependencies")
            
            turn = self.audio_service.capture_and_process_turn(
                question, turn_idx, conversation_dir, per_turn_seconds
            )
            
            # Identify speaker
            identification_result = self.person_service.identify_person_from_turn(turn, turn_idx)
            
            # Handle speaker identification results
            self._handle_speaker_identification(identification_result, state)
            
            return turn
            
        except Exception as e:
            logger.error("Failed to process turn %d: %s", turn_idx, e)
            self.event_bus.emit(ErrorOccurredEvent(
                state.conversation_id or "unknown", time.time(), type(e).__name__,
                str(e), "turn_processing"
            ))
            raise
    
    def _handle_speaker_identification(self, identification_result: Dict[str, Any], 
                                     state: InterviewState):
        """Handle speaker identification results and events."""
        person_id = identification_result.get("person_id")
        confidence = identification_result.get("confidence", 0.0)
        should_welcome = identification_result.get("should_welcome", False)
        should_terminate = identification_result.get("should_terminate", False)
        
        # Update state
        if person_id:
            state.update_speaker_identification(person_id, confidence)
            
            # Emit speaker identified event
            self.event_bus.emit(SpeakerIdentifiedEvent(
                state.conversation_id or "unknown", time.time(), person_id,
                confidence, should_welcome
            ))
        
        # Handle welcoming
        if should_welcome and not state.was_welcomed and person_id:
            if self.person_service.welcome_returning_person(person_id, self.tts_service.use_tts):
                state.welcome_speaker()
                
                # Emit welcomed event
                self.event_bus.emit(SpeakerWelcomedEvent(
                    state.conversation_id or "unknown", time.time(), person_id, f"Speaker {person_id}"
                ))
        
        # Handle termination
        if should_terminate:
            disposition = identification_result.get("disposition", "unknown")
            reasoning = identification_result.get("reasoning", "Hostile behavior detected")
            
            state.set_termination(f"Hostile speaker detected: {person_id}")
            
            # Emit hostile behavior event
            self.event_bus.emit(HostileBehaviorDetectedEvent(
                state.conversation_id or "unknown", time.time(), person_id or "unknown",
                reasoning, disposition
            ))
            
            print(f"\n❌ {state.termination_reason}")
    
    def _make_continuation_decision(self, turns: List[Turn], remaining_questions: int, 
                                  state: InterviewState) -> Dict[str, Any]:
        """Make decision about whether to continue or finalize interview."""
        print("🤔 Analyzing response...")
        print(f"   Questions remaining: {remaining_questions}")
        
        # Build context
        context = self.conversation_manager.build_interview_context(
            turns, remaining_questions, state, self.personality_traits
        )
        
        # Get decision from LLM
        decision = self.decision_engine.decide_next_action(context)
        
        print(f"   LLM decided: {decision.action}")
        logger.info(f"Questions remaining: {remaining_questions}, LLM decision: {decision}")
        
        # Emit decision event
        self.event_bus.emit(DecisionMadeEvent(
            state.conversation_id or "unknown", time.time(), decision.action,
            decision.question, remaining_questions
        ))
        
        # Convert decision to dict format for compatibility
        result = {
            "action": decision.action,
            "question": decision.question,
            "extracted_name": decision.extracted_name,
            "opinion_word": decision.opinion_word,
            "score_overall": decision.score_overall,
            "score_text_only": decision.score_text_only,
            "rationale": decision.rationale,
            "termination_message": decision.termination_message
        }
        
        return result
    
    def _handle_early_termination(self, state: InterviewState, turns: List[Turn]) -> Dict[str, Any]:
        """Handle early termination due to hostile behavior."""
        if state.identified_speaker:
            # Generate custom termination message
            custom_termination = self.decision_engine.generate_hostile_termination_message(
                state.identified_speaker, turns, self.person_service.person_manager
            )
        else:
            custom_termination = "I need to end our conversation here. Take care."
        
        # Speak termination message
        self.tts_service.speak_or_print(custom_termination)
        
        return {
            "opinion_word": "dismissive",
            "score_overall": -0.9,
            "score_text_only": -0.5,
            "rationale": "Interview terminated early due to previous negative interaction history.",
            "termination_message": custom_termination,
            "_message_spoken": True  # Flag to prevent duplicate speaking
        }
    
    def _build_final_result(self, turns: List[Turn], state: InterviewState,
                           final_payload: Optional[Dict[str, Any]],
                           conversation_id: str) -> InterviewResult:
        """Build final interview result."""
        # Process speaker profile
        speaker_id, speaker_name, is_returning_speaker, final_speaker_confidence = (
            self.person_service.process_final_speaker_profile(turns, state, final_payload)
        )
        
        # Speak final termination message (only if not already spoken)
        if (final_payload and final_payload.get("termination_message") and 
            not final_payload.get("_message_spoken", False)):
            self.tts_service.speak_or_print(final_payload["termination_message"])
        
        # Build result
        result = InterviewResult(
            turns=turns,
            final_opinion_word=(final_payload.get("opinion_word") or "neutral").strip().lower() if final_payload else "neutral",
            final_opinion_rationale=final_payload.get("rationale", "") if final_payload else "",
            final_score_text_only=float(final_payload.get("score_text_only", 0.0)) if final_payload else 0.0,
            final_score_overall=float(final_payload.get("score_overall", 0.0)) if final_payload else 0.0,
            termination_message=final_payload.get("termination_message", "Thanks for the conversation.") if final_payload else "Thanks for the conversation.",
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            is_returning_speaker=is_returning_speaker,
            speaker_confidence=final_speaker_confidence,
            termination_reason=state.termination_reason,
            was_welcomed=state.was_welcomed
        )
        
        logger.info(f"Interview completed - Final result: {result}")
        return result
    
    def _display_results(self, result: InterviewResult):
        """Display final interview results."""
        print("\n" + "=" * 50)
        if result.termination_reason:
            print("🚫 INTERVIEW TERMINATED")
            print("=" * 50)
            print(f"🛑 Reason: {result.termination_reason}")
        else:
            print("🎯 INTERVIEW COMPLETE")
            print("=" * 50)
            print(f"📊 Final Assessment: {result.final_opinion_word.title()}")
            print(f"📝 Reasoning: {result.final_opinion_rationale}")
            print(f"🔢 Overall Score: {result.final_score_overall:+.2f}")
            print(f"💭 Text Score: {result.final_score_text_only:+.2f}")
        
        # Show termination message
        if result.termination_message and result.termination_message.strip():
            print(f"🤖 AI's Final Words: \"{result.termination_message}\"")
        
        # Show voice profile information
        if result.speaker_id:
            if result.is_returning_speaker:
                welcome_status = " (welcomed back)" if result.was_welcomed else ""
                print(f"👤 Speaker: {result.speaker_name} (returning, confidence: {result.speaker_confidence:.2f}){welcome_status}")
            else:
                print(f"👤 Speaker: New person (profile created: {result.speaker_id})")
        
        print(f"📁 Full details logged to: {self.log_file}")
        
        # Show metrics
        metrics = self.metrics.get_metrics()
        print(f"📈 Session metrics: {metrics}")
    
    def get_metrics(self) -> Dict[str, int]:
        """Get current session metrics."""
        return self.metrics.get_metrics()
    
    def reset_metrics(self):
        """Reset session metrics."""
        self.metrics.reset()
