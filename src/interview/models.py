"""
Data models for the interview system.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class Turn:
    """Represents a single conversation turn."""
    idx: int
    question: Optional[str]
    wav_path: str
    transcript: str
    features: Dict[str, float]


@dataclass
class InterviewResult:
    """Final interview results and analysis."""
    turns: List[Turn] = field(default_factory=list)
    final_opinion_word: str = "neutral"
    final_opinion_rationale: str = ""
    final_score_text_only: float = 0.0
    final_score_overall: float = 0.0
    termination_message: str = ""
    # Voice profile information
    speaker_id: Optional[str] = None
    speaker_name: Optional[str] = None
    is_returning_speaker: bool = False
    speaker_confidence: float = 0.0
    # Progressive identification information
    termination_reason: Optional[str] = None
    was_welcomed: bool = False
