"""
Conversation and interview data structures.
Handles turn-by-turn conversation records and metadata.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    turn_idx: int
    question: str  # What the AI asked
    transcript: str  # What the person said
    timestamp: str  # When this turn happened
    voice_features: Dict[str, float]  # Voice features for this specific turn
    duration_seconds: Optional[float] = None  # How long they spoke

@dataclass
class ConversationRecord:
    """Complete record of a single conversation session."""
    # Basic metadata
    conversation_id: str
    date_time: str  # ISO format timestamp
    duration_minutes: Optional[float] = None
    
    # Conversation flow
    turns: List[ConversationTurn] = field(default_factory=list)
    initial_question: str = ""  # The opening question
    
    # AI's assessment
    final_opinion_word: str = ""  # positive, negative, neutral, etc.
    final_score_overall: float = 0.0  # -1.0 to 1.0
    final_score_text_only: float = 0.0  # Score based just on text
    ai_rationale: str = ""  # AI's reasoning for the opinion
    termination_message: str = ""  # What the AI said at the end
    termination_reason: Optional[str] = None  # If ended early, why?
    
    # AI's personality context for this conversation
    ai_personality_traits: Dict[str, float] = field(default_factory=dict)  # directness, curiosity, etc.
    
    # Identification details
    identification_confidence: float = 0.0  # How confident we were this was them
    was_welcomed_back: bool = False  # Did we recognize and welcome them?
    
    # Technical details
    aggregated_voice_features: Dict[str, float] = field(default_factory=dict)  # Conversation-level voice summary
