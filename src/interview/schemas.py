"""
Structured data models and schemas for the interview system.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Literal, TYPE_CHECKING
from enum import Enum
import json

if TYPE_CHECKING:
    from .models import Turn

try:
    from pydantic import BaseModel, Field, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None  # type: ignore
    Field = None
    validator = None


class InterviewAction(str, Enum):
    """Valid interview actions."""
    ASK = "ask"
    FINAL = "final"


@dataclass
class InterviewState:
    """Manages interview state throughout the conversation."""
    identified_speaker: Optional[str] = None
    speaker_confidence: float = 0.0
    termination_reason: Optional[str] = None
    was_welcomed: bool = False
    extracted_speaker_name: Optional[str] = None
    conversation_id: Optional[str] = None
    turn_count: int = 0
    
    def update_speaker_identification(self, speaker_id: str, confidence: float):
        """Update speaker identification information."""
        self.identified_speaker = speaker_id
        self.speaker_confidence = confidence
    
    def set_termination(self, reason: str):
        """Set termination reason."""
        self.termination_reason = reason
    
    def welcome_speaker(self):
        """Mark speaker as welcomed."""
        self.was_welcomed = True
    
    def increment_turn(self):
        """Increment turn counter.""" 
        self.turn_count += 1


@dataclass
class InterviewContext:
    """Context object containing all information needed for decision making."""
    turns: List['Turn']
    remaining_questions: int
    state: InterviewState
    personality_traits: 'PersonalityTraits'
    acoustic_features_aggregate: Dict[str, float]
    
    @property
    def turn_count(self) -> int:
        """Get the current turn count."""
        return len(self.turns)
    
    def get_recent_turns(self, count: int = 2) -> List['Turn']:
        """Get the most recent turns."""
        return self.turns[-count:] if self.turns else []
    
    def get_full_transcript(self) -> str:
        """Get complete conversation transcript."""
        return " ".join([turn.transcript for turn in self.turns if turn.transcript])
    
    def should_finalize(self) -> bool:
        """Check if interview should be finalized."""
        return self.remaining_questions <= 0 or self.state.termination_reason is not None


# Import the flexible personality system
from ..config import PersonalityConfig as PersonalityTraits


# Define common structures using dataclasses (always available)
@dataclass
class InterviewDecision:
    """Decision structure for interview flow."""
    action: str
    question: Optional[str] = None
    extracted_name: Optional[str] = None
    opinion_word: Optional[str] = None
    score_overall: Optional[float] = None
    score_text_only: Optional[float] = None
    rationale: Optional[str] = None
    termination_message: Optional[str] = None
    
    def validate(self) -> bool:
        """Basic validation."""
        if self.action == "ask" and not self.question:
            return False
        if self.action == "final" and not self.opinion_word:
            return False
        if self.score_overall is not None and not (-1.0 <= self.score_overall <= 1.0):
            return False
        if self.score_text_only is not None and not (-1.0 <= self.score_text_only <= 1.0):
            return False
        return True

@dataclass
class HostileTerminationRequest:
    """Request for generating hostile speaker termination message."""
    speaker_name: str
    conversation_count: int
    recent_opinions: List[str]
    recent_scores: List[float]
    current_transcript: str
    personality_context: str


def parse_llm_decision(raw_response: str) -> InterviewDecision:
    """
    Parse LLM response into structured decision with robust error handling.
    
    Args:
        raw_response: Raw JSON string from LLM
        
    Returns:
        InterviewDecision object
        
    Raises:
        ValueError: If response cannot be parsed into valid decision
    """
    try:
        # First try direct JSON parsing
        data = json.loads(raw_response)
    except json.JSONDecodeError:
        # Try extracting JSON from text
        start = raw_response.find("{")
        end = raw_response.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(raw_response[start:end+1])
            except json.JSONDecodeError:
                raise ValueError(f"Could not extract valid JSON from LLM response: {raw_response}")
        else:
            raise ValueError(f"No JSON found in LLM response: {raw_response}")
    
    # Create decision object
    try:
        # Filter only known fields for dataclass
        valid_fields = {k: v for k, v in data.items() 
                       if k in InterviewDecision.__dataclass_fields__}
        decision = InterviewDecision(**valid_fields)
        
        if not decision.validate():
            raise ValueError(f"Decision validation failed: {decision}")
        return decision
    except Exception as e:
        raise ValueError(f"Invalid decision structure: {e}")
