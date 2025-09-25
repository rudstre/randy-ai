"""
Testing infrastructure with mock services for the interview system.
"""
import os
import json
import tempfile
from typing import Dict, Any, List, Optional
from unittest.mock import Mock
from dataclasses import dataclass

from .models import Turn, InterviewResult
from .schemas import InterviewDecision, PersonalityTraits, InterviewState
from .services import AudioInterviewService, SpeakerIdentificationService, TTSService
from .decision_engine import InterviewDecisionEngine, PromptEngine
# AudioCapture imported conditionally in methods to avoid dependency issues


@dataclass
class MockAudioData:
    """Mock audio data for testing."""
    transcript: str
    features: Dict[str, float]
    wav_path: str = ""
    
    def __post_init__(self):
        if not self.wav_path:
            self.wav_path = f"/tmp/mock_audio_{id(self)}.wav"


class MockAudioService(AudioInterviewService):
    """Mock audio service for testing."""
    
    def __init__(self, mock_responses: List[MockAudioData]):
        # Don't call super().__init__ to avoid creating real AudioCapture
        self.mock_responses = mock_responses
        self.current_response_idx = 0
        self.language_code = "en-US"
    
    def capture_and_process_turn(self, 
                                question: str,
                                turn_idx: int, 
                                conversation_dir: str, 
                                per_turn_seconds: float) -> Turn:
        """Return mock turn data."""
        if self.current_response_idx < len(self.mock_responses):
            mock_data = self.mock_responses[self.current_response_idx]
            self.current_response_idx += 1
            
            return Turn(
                idx=turn_idx,
                question=question,
                wav_path=mock_data.wav_path,
                transcript=mock_data.transcript,
                features=mock_data.features
            )
        else:
            # Return empty turn if no more mock responses
            return Turn(
                idx=turn_idx,
                question=question,
                wav_path="",
                transcript="",
                features={}
            )


class MockSpeakerIdentificationService(SpeakerIdentificationService):
    """Mock speaker identification service for testing."""
    
    def __init__(self, mock_identifications: List[Dict[str, Any]]):
        # Don't call super().__init__ to avoid requiring real services
        self.mock_identifications = mock_identifications
        self.current_identification_idx = 0
        self.welcomed_speakers = set()
    
    def identify_speaker_from_turn(self, turn: Turn, turn_idx: int) -> Dict[str, Any]:
        """Return mock identification result."""
        if self.current_identification_idx < len(self.mock_identifications):
            result = self.mock_identifications[self.current_identification_idx]
            self.current_identification_idx += 1
            return result
        else:
            return {
                "speaker_id": None,
                "confidence": 0.0,
                "should_welcome": False,
                "should_terminate": False,
                "reasoning": "No more mock identifications",
                "disposition": None
            }
    
    def welcome_returning_speaker(self, speaker_id: str, use_tts: bool = True) -> bool:
        """Mock welcome behavior."""
        if speaker_id not in self.welcomed_speakers:
            print(f"👋 Welcome back, Speaker {speaker_id}!")
            self.welcomed_speakers.add(speaker_id)
            return True
        return False
    
    def process_final_speaker_profile(self, *args, **kwargs):
        """Mock profile processing."""
        return "mock_speaker_123", "Mock Speaker", False, 0.8
    
    def start_new_conversation(self):
        """Mock conversation start."""
        print("📚 Mock: Starting new conversation")


class MockTTSService(TTSService):
    """Mock TTS service for testing."""
    
    def __init__(self, use_tts: bool = False):
        # Don't call super().__init__ to avoid initializing real TTS
        self.use_tts = use_tts
        self.spoken_messages = []
    
    def speak_or_print(self, message: str, prefix: str = "🤖"):
        """Mock speak/print behavior."""
        self.spoken_messages.append(message)
        print(f"[MOCK TTS] {prefix} {message}")
    
    def cleanup(self):
        """Mock cleanup."""
        pass


class MockLLMClient:
    """Mock LLM client for testing."""
    
    def __init__(self, mock_responses: List[str]):
        self.mock_responses = mock_responses
        self.current_response_idx = 0
        self.request_history = []
    
    def generate_content(self, prompt: str, temperature: float = 0.0, **kwargs) -> str:
        """Return mock LLM response."""
        self.request_history.append({
            "prompt": prompt,
            "temperature": temperature,
            "kwargs": kwargs
        })
        
        if self.current_response_idx < len(self.mock_responses):
            response = self.mock_responses[self.current_response_idx]
            self.current_response_idx += 1
            return response
        else:
            # Default fallback response
            return '{"action": "final", "opinion_word": "neutral", "score_overall": 0.0, "score_text_only": 0.0, "rationale": "Mock fallback", "termination_message": "Mock goodbye"}'


class MockDecisionEngine(InterviewDecisionEngine):
    """Mock decision engine for testing."""
    
    def __init__(self, mock_decisions: List[InterviewDecision]):
        # Don't call super().__init__ to avoid requiring real LLM client
        self.mock_decisions = mock_decisions
        self.current_decision_idx = 0
    
    def decide_next_action(self, context) -> InterviewDecision:
        """Return mock decision."""
        if self.current_decision_idx < len(self.mock_decisions):
            decision = self.mock_decisions[self.current_decision_idx]
            self.current_decision_idx += 1
            return decision
        else:
            # Default fallback
            return InterviewDecision(
                action="final",
                opinion_word="neutral",
                score_overall=0.0,
                score_text_only=0.0,
                rationale="Mock fallback decision",
                termination_message="Mock goodbye"
            )
    
    def generate_hostile_termination_message(self, *args, **kwargs) -> str:
        """Mock hostile termination message."""
        return "I'm ending this conversation due to hostile behavior. [MOCK]"


def create_mock_interview_setup() -> Dict[str, Any]:
    """Create a complete mock interview setup for testing."""
    
    # Mock audio responses
    mock_audio_responses = [
        MockAudioData(
            transcript="Hi, I'm John and I'm excited to be here.",
            features={"pitch_mean": 120.5, "mfcc_1": 0.3, "loudness": 0.8}
        ),
        MockAudioData(
            transcript="I work in software development and love solving problems.",
            features={"pitch_mean": 118.2, "mfcc_1": 0.31, "loudness": 0.75}
        ),
        MockAudioData(
            transcript="Thank you for the conversation, it was interesting.",
            features={"pitch_mean": 121.1, "mfcc_1": 0.29, "loudness": 0.82}
        )
    ]
    
    # Mock speaker identifications
    mock_identifications = [
        {
            "speaker_id": None,
            "confidence": 0.0,
            "should_welcome": False,
            "should_terminate": False,
            "reasoning": "New speaker detected",
            "disposition": None
        },
        {
            "speaker_id": "speaker_123",
            "confidence": 0.85,
            "should_welcome": False,
            "should_terminate": False,
            "reasoning": "Speaker identified with good confidence",
            "disposition": "friendly"
        },
        {
            "speaker_id": "speaker_123",
            "confidence": 0.9,
            "should_welcome": False,
            "should_terminate": False,
            "reasoning": "Confirmation of speaker identity",
            "disposition": "friendly"
        }
    ]
    
    # Mock LLM responses  
    mock_llm_responses = [
        '{"action": "ask", "question": "What kind of software development do you enjoy most?", "extracted_name": "John"}',
        '{"action": "ask", "question": "What challenges you most in your work?"}',
        '{"action": "final", "opinion_word": "positive", "score_overall": 0.7, "score_text_only": 0.6, "rationale": "Engaged and thoughtful responses", "termination_message": "Great talking with you, John! Keep up the excellent work.", "extracted_name": "John"}'
    ]
    
    # Mock decisions
    mock_decisions = [
        InterviewDecision(
            action="ask",
            question="What kind of software development do you enjoy most?",
            extracted_name="John"
        ),
        InterviewDecision(
            action="ask", 
            question="What challenges you most in your work?"
        ),
        InterviewDecision(
            action="final",
            opinion_word="positive",
            score_overall=0.7,
            score_text_only=0.6,
            rationale="Engaged and thoughtful responses",
            termination_message="Great talking with you, John! Keep up the excellent work.",
            extracted_name="John"
        )
    ]
    
    # Create services
    audio_service = MockAudioService(mock_audio_responses)
    speaker_service = MockSpeakerIdentificationService(mock_identifications)
    tts_service = MockTTSService(use_tts=False)
    llm_client = MockLLMClient(mock_llm_responses)
    decision_engine = MockDecisionEngine(mock_decisions)
    
    # Create personality traits
    personality_traits = PersonalityTraits(
        directness=0.7,
        curiosity=0.8,
        skepticism=0.6,
        engagement=0.8,
        tolerance=0.5
    )
    
    # Create temporary directory for conversation
    temp_dir = tempfile.mkdtemp()
    
    return {
        "audio_service": audio_service,
        "speaker_service": speaker_service, 
        "tts_service": tts_service,
        "llm_client": llm_client,
        "decision_engine": decision_engine,
        "personality_traits": personality_traits,
        "temp_dir": temp_dir
    }


def create_test_conversation_data() -> List[Turn]:
    """Create test conversation data."""
    return [
        Turn(
            idx=1,
            question="Hi! Please introduce yourself.",
            wav_path="/tmp/test_turn1.wav",
            transcript="Hello, I'm Alice and I work in marketing.",
            features={"pitch_mean": 180.5, "mfcc_1": 0.25, "loudness": 0.7}
        ),
        Turn(
            idx=2, 
            question="What do you enjoy most about marketing?",
            wav_path="/tmp/test_turn2.wav",
            transcript="I love the creative aspects and connecting with people.",
            features={"pitch_mean": 175.2, "mfcc_1": 0.27, "loudness": 0.73}
        ),
        Turn(
            idx=3,
            question="What are your biggest challenges?",
            wav_path="/tmp/test_turn3.wav", 
            transcript="Keeping up with all the new digital platforms and tools.",
            features={"pitch_mean": 178.8, "mfcc_1": 0.24, "loudness": 0.68}
        )
    ]


class TestInterviewResult:
    """Helper for validating interview results."""
    
    @staticmethod
    def validate_result(result: InterviewResult) -> List[str]:
        """
        Validate interview result and return list of issues found.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []
        
        if not result.turns:
            issues.append("No turns recorded")
        
        if not result.final_opinion_word:
            issues.append("Missing final opinion word")
        
        if result.final_score_overall is None:
            issues.append("Missing overall score")
        
        if not result.termination_message:
            issues.append("Missing termination message")
        
        if result.final_score_overall is not None:
            if not (-1.0 <= result.final_score_overall <= 1.0):
                issues.append(f"Overall score out of range: {result.final_score_overall}")
        
        if result.final_score_text_only is not None:
            if not (-1.0 <= result.final_score_text_only <= 1.0):
                issues.append(f"Text score out of range: {result.final_score_text_only}")
        
        # Validate turns
        for i, turn in enumerate(result.turns):
            if not turn.transcript and not turn.features:
                issues.append(f"Turn {i+1} has no transcript or features")
        
        return issues
    
    @staticmethod
    def assert_valid_result(result: InterviewResult) -> None:
        """Assert that interview result is valid, raising AssertionError if not."""
        issues = TestInterviewResult.validate_result(result)
        if issues:
            raise AssertionError(f"Invalid interview result: {'; '.join(issues)}")


def cleanup_test_files(temp_dir: str) -> None:
    """Clean up test files and directories."""
    import shutil
    try:
        shutil.rmtree(temp_dir)
    except OSError:
        pass  # Directory may not exist or be deletable
