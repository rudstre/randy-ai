"""
AI analysis and behavioral assessment for interview subjects.
Handles personality analysis, behavioral pattern detection, and assessment tracking.
"""
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from ..infrastructure.data.conversations import ConversationRecord

logger = logging.getLogger("interview_analysis")

@dataclass
class PersonalityAssessment:
    """AI's assessment of a person's personality and behavior patterns."""
    
    # Overall assessment
    personality_summary: str = ""  # AI's overall take on their personality
    
    # Behavioral patterns
    behavioral_patterns: List[str] = field(default_factory=list)  # Things the AI has noticed
    communication_style: str = ""  # How they communicate
    engagement_level: str = ""  # How engaged they are in conversations
    
    # Sentiment indicators  
    hostility_indicators: List[str] = field(default_factory=list)  # Specific hostile behaviors
    positive_indicators: List[str] = field(default_factory=list)  # Positive behaviors
    
    # Trends over time
    sentiment_trend: str = ""  # improving, declining, stable
    consistency_score: float = 0.0  # How consistent their behavior is
    
    # Assessment metadata
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    confidence_level: float = 0.0  # How confident the AI is in this assessment
    conversation_count: int = 0  # Number of conversations this is based on

class InterviewAnalyzer:
    """Analyzes conversation patterns and maintains personality assessments."""
    
    def __init__(self):
        self.assessments: Dict[str, PersonalityAssessment] = {}
    
    def analyze_conversation(self, person_id: str, conversation: ConversationRecord) -> None:
        """Analyze a single conversation and update person's assessment."""
        
        if person_id not in self.assessments:
            self.assessments[person_id] = PersonalityAssessment()
        
        assessment = self.assessments[person_id]
        assessment.conversation_count += 1
        assessment.last_updated = datetime.now().isoformat()
        
        # Analyze conversation content
        self._analyze_communication_patterns(conversation, assessment)
        self._analyze_sentiment_indicators(conversation, assessment)
        self._update_behavioral_patterns(conversation, assessment)
        self._calculate_consistency(person_id, conversation, assessment)
        
        logger.info(f"Updated personality assessment for {person_id}")
    
    def _analyze_communication_patterns(self, conversation: ConversationRecord, assessment: PersonalityAssessment) -> None:
        """Analyze how the person communicates."""
        
        # Analyze response lengths
        response_lengths = [len(turn.transcript) for turn in conversation.turns if turn.transcript]
        avg_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0
        
        # Analyze engagement
        if avg_length > 100:
            if "detailed responses" not in assessment.behavioral_patterns:
                assessment.behavioral_patterns.append("detailed responses")
        elif avg_length < 30:
            if "brief responses" not in assessment.behavioral_patterns:
                assessment.behavioral_patterns.append("brief responses")
        
        # Analyze question engagement
        engaging_words = ["interesting", "think", "feel", "because", "actually", "really"]
        dismissive_words = ["whatever", "sure", "fine", "okay", "yeah"]
        
        all_text = " ".join([turn.transcript.lower() for turn in conversation.turns if turn.transcript])
        
        engaging_count = sum(1 for word in engaging_words if word in all_text)
        dismissive_count = sum(1 for word in dismissive_words if word in all_text)
        
        if engaging_count > dismissive_count and engaging_count > 2:
            assessment.engagement_level = "high"
            if "thoughtful engagement" not in assessment.positive_indicators:
                assessment.positive_indicators.append("thoughtful engagement")
        elif dismissive_count > engaging_count:
            assessment.engagement_level = "low"
            if "dismissive responses" not in assessment.hostility_indicators:
                assessment.hostility_indicators.append("dismissive responses")
        else:
            assessment.engagement_level = "moderate"
    
    def _analyze_sentiment_indicators(self, conversation: ConversationRecord, assessment: PersonalityAssessment) -> None:
        """Detect positive and negative sentiment indicators."""
        
        # Based on final AI assessment
        if conversation.final_opinion_word in ["negative", "hostile", "dismissive", "cold"]:
            reason = f"AI rated conversation as {conversation.final_opinion_word}"
            if reason not in assessment.hostility_indicators:
                assessment.hostility_indicators.append(reason)
        
        elif conversation.final_opinion_word in ["positive", "warm", "friendly", "engaging"]:
            reason = f"AI rated conversation as {conversation.final_opinion_word}"
            if reason not in assessment.positive_indicators:
                assessment.positive_indicators.append(reason)
        
        # Early termination analysis
        if conversation.termination_reason:
            termination_indicator = f"Conversation terminated: {conversation.termination_reason}"
            if termination_indicator not in assessment.hostility_indicators:
                assessment.hostility_indicators.append(termination_indicator)
        
        # Score-based analysis
        if conversation.final_score_overall < -0.5:
            if "consistently negative interactions" not in assessment.hostility_indicators:
                assessment.hostility_indicators.append("consistently negative interactions")
        elif conversation.final_score_overall > 0.5:
            if "positive interaction pattern" not in assessment.positive_indicators:
                assessment.positive_indicators.append("positive interaction pattern")
    
    def _update_behavioral_patterns(self, conversation: ConversationRecord, assessment: PersonalityAssessment) -> None:
        """Update observed behavioral patterns."""
        
        # Analyze AI's rationale for additional insights
        rationale_lower = conversation.ai_rationale.lower()
        
        # Pattern detection based on AI rationale
        if "thoughtful" in rationale_lower:
            if "thoughtful responses" not in assessment.behavioral_patterns:
                assessment.behavioral_patterns.append("thoughtful responses")
        
        if "consistent" in rationale_lower:
            if "consistent behavior" not in assessment.behavioral_patterns:
                assessment.behavioral_patterns.append("consistent behavior")
        
        if "authentic" in rationale_lower:
            if "authentic presentation" not in assessment.positive_indicators:
                assessment.positive_indicators.append("authentic presentation")
        
        if "evasive" in rationale_lower:
            if "evasive responses" not in assessment.hostility_indicators:
                assessment.hostility_indicators.append("evasive responses")
        
        # Turn count analysis
        if len(conversation.turns) < 2:
            if "very brief interactions" not in assessment.behavioral_patterns:
                assessment.behavioral_patterns.append("very brief interactions")
        elif len(conversation.turns) > 5:
            if "extended conversations" not in assessment.behavioral_patterns:
                assessment.behavioral_patterns.append("extended conversations")
    
    def _calculate_consistency(self, person_id: str, conversation: ConversationRecord, assessment: PersonalityAssessment) -> None:
        """Calculate consistency score based on conversation history."""
        
        # This would need access to previous conversations to calculate properly
        # For now, use a simple heuristic based on current conversation
        
        if assessment.conversation_count == 1:
            assessment.consistency_score = 1.0  # First conversation, fully consistent
        else:
            # Simple consistency based on sentiment stability
            # In a full implementation, this would analyze patterns across conversations
            current_sentiment = 1.0 if conversation.final_score_overall > 0 else -1.0 if conversation.final_score_overall < 0 else 0.0
            
            # For now, just update based on current vs expected sentiment
            # This is a placeholder - real implementation would track sentiment history
            assessment.consistency_score = 0.8  # Placeholder value
    
    def get_assessment(self, person_id: str) -> Optional[PersonalityAssessment]:
        """Get the current personality assessment for a person."""
        return self.assessments.get(person_id)
    
    def update_personality_summary(self, person_id: str, conversations: List[ConversationRecord]) -> None:
        """Generate an overall personality summary based on all conversations."""
        
        if person_id not in self.assessments:
            return
        
        assessment = self.assessments[person_id]
        
        # Generate summary based on patterns and indicators
        summary_parts = []
        
        # Overall sentiment
        positive_count = len(assessment.positive_indicators)
        negative_count = len(assessment.hostility_indicators)
        
        if positive_count > negative_count * 2:
            summary_parts.append("Generally positive and cooperative individual")
        elif negative_count > positive_count * 2:
            summary_parts.append("Shows concerning behavioral patterns")
        else:
            summary_parts.append("Mixed behavioral presentation")
        
        # Engagement level
        if assessment.engagement_level == "high":
            summary_parts.append("demonstrates strong engagement in conversations")
        elif assessment.engagement_level == "low":
            summary_parts.append("shows limited engagement or interest")
        
        # Communication style
        if "detailed responses" in assessment.behavioral_patterns:
            summary_parts.append("provides thoughtful, detailed responses")
        elif "brief responses" in assessment.behavioral_patterns:
            summary_parts.append("tends to give brief, minimal responses")
        
        # Consistency
        if assessment.consistency_score > 0.8:
            summary_parts.append("maintains consistent behavior across interactions")
        elif assessment.consistency_score < 0.5:
            summary_parts.append("shows variable or unpredictable behavior")
        
        assessment.personality_summary = ". ".join(summary_parts).capitalize() + "."
        
        # Update confidence based on data quantity and consistency
        assessment.confidence_level = min(1.0, (assessment.conversation_count / 5.0) * assessment.consistency_score)
    
    def analyze_trends(self, person_id: str, conversations: List[ConversationRecord]) -> None:
        """Analyze trends in behavior over time."""
        
        if person_id not in self.assessments or len(conversations) < 2:
            return
        
        assessment = self.assessments[person_id]
        
        # Analyze sentiment trend
        scores = [c.final_score_overall for c in conversations]
        
        if len(scores) >= 3:
            recent_scores = scores[-3:]
            earlier_scores = scores[:-3] if len(scores) > 3 else scores[:-1]
            
            recent_avg = sum(recent_scores) / len(recent_scores)
            earlier_avg = sum(earlier_scores) / len(earlier_scores)
            
            if recent_avg > earlier_avg + 0.2:
                assessment.sentiment_trend = "improving"
            elif recent_avg < earlier_avg - 0.2:
                assessment.sentiment_trend = "declining"
            else:
                assessment.sentiment_trend = "stable"
        else:
            assessment.sentiment_trend = "insufficient_data"
    
    def get_risk_assessment(self, person_id: str) -> Dict[str, Any]:
        """Get a risk assessment for a person based on their behavioral patterns."""
        
        assessment = self.get_assessment(person_id)
        if not assessment:
            return {"risk_level": "unknown", "reason": "no_assessment_data"}
        
        risk_factors = len(assessment.hostility_indicators)
        positive_factors = len(assessment.positive_indicators)
        
        # Calculate risk level
        if risk_factors >= 3 and positive_factors == 0:
            risk_level = "high"
        elif risk_factors >= 2:
            risk_level = "moderate"
        elif risk_factors == 1 and positive_factors == 0:
            risk_level = "low"
        else:
            risk_level = "minimal"
        
        return {
            "risk_level": risk_level,
            "risk_factors": assessment.hostility_indicators,
            "positive_factors": assessment.positive_indicators,
            "confidence": assessment.confidence_level,
            "conversation_count": assessment.conversation_count,
            "sentiment_trend": assessment.sentiment_trend
        }
