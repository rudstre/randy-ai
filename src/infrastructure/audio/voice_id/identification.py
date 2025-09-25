"""
Progressive voice identification module for real-time speaker recognition with early termination.
Optimized for conversation-based identification where early detection of hostile speakers is critical.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger("progressive_voice_id")

class IdentificationConfidence(Enum):
    VERY_HIGH = "very_high"      # >0.85 - Act immediately
    HIGH = "high"                # 0.7-0.85 - Strong evidence
    MEDIUM = "medium"            # 0.5-0.7 - Uncertain, continue
    LOW = "low"                  # <0.5 - Likely new person

class SpeakerDisposition(Enum):
    HOSTILE = "hostile"          # Terminate immediately if identified
    NEUTRAL = "neutral"          # Normal conversation
    FRIENDLY = "friendly"        # Welcome back

@dataclass
class ProgressiveIdentificationResult:
    speaker_id: Optional[str]
    confidence: float
    confidence_level: IdentificationConfidence
    disposition: Optional[SpeakerDisposition]
    should_terminate: bool
    should_welcome: bool
    reasoning: str
    cumulative_evidence: Dict[str, Any]

class ProgressiveVoiceIdentifier:
    """
    Identifies speakers progressively, with early termination for hostile speakers.
    Optimized for single-turn identification but improves with more data.
    """
    
    def __init__(self, voice_profile_manager, 
                 aggressiveness: float = 0.7):
        """
        Initialize progressive voice identifier.
        
        Args:
            voice_profile_manager: VoiceProfileManager instance
            aggressiveness: How aggressive the identification should be (0.0-1.0)
                          0.0 = very conservative (minimal false positives)
                          1.0 = very aggressive (catch all hostile speakers)
        """
        self.voice_manager = voice_profile_manager
        self.aggressiveness = max(0.0, min(1.0, aggressiveness))  # Clamp to 0-1
        
        # Calculate dynamic thresholds based on aggressiveness
        self.confidence_thresholds = self._calculate_confidence_thresholds()
        self.hostile_threshold = self._calculate_hostile_threshold()
        self.welcome_threshold = self._calculate_welcome_threshold()
        
        # Progressive evidence tracking for current conversation
        self.current_session_evidence = {}
        self.turn_similarities = {}  # speaker_id -> list of similarities
        self.session_active = False
    
    def _calculate_confidence_thresholds(self) -> Dict[IdentificationConfidence, float]:
        """Calculate confidence thresholds based on aggressiveness level."""
        # Conservative thresholds (aggressiveness = 0.0)
        conservative = {
            IdentificationConfidence.VERY_HIGH: 0.90,
            IdentificationConfidence.HIGH: 0.80,
            IdentificationConfidence.MEDIUM: 0.65,
            IdentificationConfidence.LOW: 0.0
        }
        
        # Aggressive thresholds (aggressiveness = 1.0)
        aggressive = {
            IdentificationConfidence.VERY_HIGH: 0.65,
            IdentificationConfidence.HIGH: 0.45,
            IdentificationConfidence.MEDIUM: 0.30,
            IdentificationConfidence.LOW: 0.0
        }
        
        # Interpolate between conservative and aggressive based on aggressiveness
        thresholds = {}
        for level in IdentificationConfidence:
            conservative_val = conservative[level]
            aggressive_val = aggressive[level]
            interpolated = conservative_val + self.aggressiveness * (aggressive_val - conservative_val)
            thresholds[level] = interpolated
        
        return thresholds
    
    def _get_adaptive_confidence_thresholds(self, speaker_id: Optional[str] = None) -> Dict[IdentificationConfidence, float]:
        """Get confidence thresholds adapted to speaker's data quality."""
        base_thresholds = self.confidence_thresholds.copy()
        
        # Check if adaptive tolerance is enabled
        from ....config import DEFAULT_ADAPTIVE_TOLERANCE_ENABLED
        if not DEFAULT_ADAPTIVE_TOLERANCE_ENABLED:
            return base_thresholds
        
        if not speaker_id or speaker_id not in self.voice_manager.profiles:
            # For unknown speakers or no speaker, use base thresholds
            return base_thresholds
        
        # Get data confidence for this speaker
        profile = self.voice_manager.profiles[speaker_id]
        data_confidence = profile.get_data_confidence()
        
        # When data confidence is low, make thresholds more lenient
        # Formula: threshold * (1.0 - adjustment_factor * (1.0 - data_confidence))
        from ....config import DEFAULT_ADAPTIVE_CONFIDENCE_ADJUSTMENT
        adjustment_factor = DEFAULT_ADAPTIVE_CONFIDENCE_ADJUSTMENT  # Up to 30% reduction in thresholds for unknown speakers
        
        adaptive_thresholds = {}
        for level, threshold in base_thresholds.items():
            if level != IdentificationConfidence.LOW:  # Don't adjust LOW threshold (always 0.0)
                reduction = adjustment_factor * (1.0 - data_confidence)
                adaptive_thresholds[level] = max(0.0, threshold * (1.0 - reduction))
            else:
                adaptive_thresholds[level] = threshold
        
        logger.debug(f"Adaptive thresholds for {speaker_id} (data_confidence={data_confidence:.3f}): "
                    f"VERY_HIGH={adaptive_thresholds[IdentificationConfidence.VERY_HIGH]:.3f}, "
                    f"HIGH={adaptive_thresholds[IdentificationConfidence.HIGH]:.3f}, "
                    f"MEDIUM={adaptive_thresholds[IdentificationConfidence.MEDIUM]:.3f}")
        
        return adaptive_thresholds
    
    def _calculate_hostile_threshold(self) -> float:
        """Calculate hostile detection threshold based on aggressiveness."""
        # Conservative: 0.85, Aggressive: 0.50
        return 0.85 - (self.aggressiveness * 0.35)
    
    def _calculate_welcome_threshold(self) -> float:
        """Calculate welcome threshold based on aggressiveness."""
        # Conservative: 0.80, Aggressive: 0.50
        return 0.80 - (self.aggressiveness * 0.30)
    
    def get_penalty_multipliers(self) -> Dict[str, float]:
        """Get dynamic penalty multipliers based on aggressiveness."""
        # More aggressive = smaller penalties (less harsh)
        base_penalties = {
            'single_turn': 0.90,      # Conservative penalty for single turn
            'low_similarity': 0.50,   # Conservative penalty for low similarity
            'unrealistic': 0.70,      # Conservative penalty for unrealistic features
            'boost_factor': 1.15      # Conservative boost factor
        }
        
        aggressive_penalties = {
            'single_turn': 0.98,      # Minimal penalty for single turn
            'low_similarity': 0.80,   # Minimal penalty for low similarity
            'unrealistic': 0.90,      # Minimal penalty for unrealistic features
            'boost_factor': 1.35      # Strong boost factor
        }
        
        # Interpolate penalties
        penalties = {}
        for key in base_penalties:
            conservative_val = base_penalties[key]
            aggressive_val = aggressive_penalties[key]
            penalties[key] = conservative_val + self.aggressiveness * (aggressive_val - conservative_val)
        
        return penalties
    
    def start_new_conversation(self):
        """Reset progressive tracking for new conversation."""
        self.current_session_evidence = {}
        self.turn_similarities = {}
        self.session_active = True
        logger.info("Started conversation - reset progressive tracking")
    
    def identify_from_turn(self, turn, turn_number: int = 1) -> ProgressiveIdentificationResult:
        """
        Identify speaker from a single turn, with progressive improvement.
        
        Args:
            turn: Turn object with features
            turn_number: Which turn this is (1-based)
            
        Returns:
            ProgressiveIdentificationResult with action recommendation
        """
        if not hasattr(turn, 'features') or not turn.features:
            return ProgressiveIdentificationResult(
                speaker_id=None,
                confidence=0.0,
                confidence_level=IdentificationConfidence.LOW,
                disposition=None,
                should_terminate=False,
                should_welcome=False,
                reasoning="No voice features available",
                cumulative_evidence={}
            )
        
        # Get similarity scores against all known speakers
        speaker_similarities = self._compute_turn_similarities(turn.features)
        
        # Update progressive evidence
        self._update_progressive_evidence(speaker_similarities, turn_number)
        
        # Determine best match and confidence
        best_speaker, raw_confidence = self._get_best_match(speaker_similarities)
        
        # Apply progressive confidence boost
        adjusted_confidence = self._apply_progressive_boost(best_speaker, raw_confidence, turn_number)
        
        # Determine confidence level (use adaptive thresholds based on speaker data)
        confidence_level = self._classify_confidence(adjusted_confidence, best_speaker)
        
        # Get speaker disposition if identified
        disposition = self._get_speaker_disposition(best_speaker) if best_speaker else None
        
        # Decision logic for termination and welcoming
        should_terminate = self._should_terminate_conversation(
            best_speaker, adjusted_confidence, disposition, turn_number
        )
        
        should_welcome = self._should_welcome_speaker(
            best_speaker, adjusted_confidence, disposition, turn_number
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            best_speaker, adjusted_confidence, confidence_level, 
            disposition, turn_number, should_terminate, should_welcome
        )
        
        result = ProgressiveIdentificationResult(
            speaker_id=best_speaker,
            confidence=adjusted_confidence,
            confidence_level=confidence_level,
            disposition=disposition,
            should_terminate=should_terminate,
            should_welcome=should_welcome,
            reasoning=reasoning,
            cumulative_evidence=self._get_cumulative_evidence()
        )
        
        logger.info(f"Turn {turn_number} identification: {reasoning}")
        return result
    
    def _compute_turn_similarities(self, features: Dict[str, float]) -> Dict[str, float]:
        """Compute similarity scores against all known speakers."""
        similarities = {}
        
        for speaker_id, profile in self.voice_manager.profiles.items():
            similarity = profile.calculate_similarity(features, self.voice_manager.similarity_calculator)
            similarities[speaker_id] = similarity
        
        return similarities
    
    def _update_progressive_evidence(self, similarities: Dict[str, float], turn_number: int):
        """Update progressive evidence tracking."""
        for speaker_id, similarity in similarities.items():
            if speaker_id not in self.turn_similarities:
                self.turn_similarities[speaker_id] = []
            self.turn_similarities[speaker_id].append(similarity)
        
        # Update session evidence
        self.current_session_evidence[f'turn_{turn_number}'] = similarities.copy()
    
    def _get_best_match(self, similarities: Dict[str, float]) -> Tuple[Optional[str], float]:
        """Get the best matching speaker and raw confidence."""
        if not similarities:
            return None, 0.0
        
        best_speaker = max(similarities.keys(), key=lambda k: similarities[k])
        best_similarity = similarities[best_speaker]
        
        return best_speaker, best_similarity
    
    def _apply_progressive_boost(self, speaker_id: Optional[str], 
                               raw_confidence: float, turn_number: int) -> float:
        """
        Simple progressive confidence: build up confidence within an interview.
        No penalties - just boost confidence as we get more consistent evidence.
        """
        if not speaker_id:
            return raw_confidence
        
        # Get all similarities for this speaker in current conversation
        similarities = self.turn_similarities.get(speaker_id, [])
        
        if len(similarities) < 2:
            # First turn - no penalty, just return raw confidence
            return raw_confidence
        
        # Multiple turns - calculate consistency bonus
        mean_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
        
        # Consistency bonus: reward stable, high similarities
        if mean_similarity > 0.5:  # Only boost if we have decent similarity
            # Boost factor increases with more turns and higher consistency
            consistency_bonus = min(0.3, (len(similarities) - 1) * 0.05)  # Up to 30% boost
            stability_bonus = max(0.0, float((0.2 - std_similarity) * 0.5))     # Reward low variance
            
            total_boost = 1.0 + consistency_bonus + stability_bonus
            boosted_confidence = min(1.0, raw_confidence * total_boost)
            
            logger.debug(f"Progressive boost for {speaker_id}: "
                       f"{raw_confidence:.3f} → {boosted_confidence:.3f} "
                       f"(turns: {len(similarities)}, consistency: {1.0/(1.0+std_similarity):.3f})")
            
            return boosted_confidence
        
        # Low similarity - no boost
        return raw_confidence
    
    def _classify_confidence(self, confidence: float, speaker_id: Optional[str] = None) -> IdentificationConfidence:
        """Classify confidence level with adaptive thresholds."""
        # Get adaptive thresholds based on profile data confidence
        thresholds = self._get_adaptive_confidence_thresholds(speaker_id)
        
        for level in [IdentificationConfidence.VERY_HIGH, 
                     IdentificationConfidence.HIGH,
                     IdentificationConfidence.MEDIUM,
                     IdentificationConfidence.LOW]:
            if confidence >= thresholds[level]:
                return level
        return IdentificationConfidence.LOW
    
    def _get_speaker_disposition(self, speaker_id: Optional[str]) -> Optional[SpeakerDisposition]:
        """Get speaker's disposition based on history."""
        if not speaker_id or speaker_id not in self.voice_manager.profiles:
            return None
        
        speaker_info = self.voice_manager.get_speaker_info(speaker_id)
        
        if not speaker_info:
            return SpeakerDisposition.NEUTRAL
        
        # Check for hostile pattern
        if speaker_info.get('is_hostile', False):
            return SpeakerDisposition.HOSTILE
        
        # Check for friendly pattern
        avg_score = speaker_info.get('avg_recent_score', 0.0)
        recent_scores = speaker_info.get('recent_scores', [])
        
        if avg_score > 0.3 and len(recent_scores) >= 1:
            return SpeakerDisposition.FRIENDLY
        elif avg_score < -0.3:
            return SpeakerDisposition.HOSTILE
        else:
            return SpeakerDisposition.NEUTRAL
    
    def _should_terminate_conversation(self, speaker_id: Optional[str], 
                                     confidence: float,
                                     disposition: Optional[SpeakerDisposition],
                                     turn_number: int) -> bool:
        """
        Simple termination logic: If we're confident this is a hostile speaker, terminate.
        """
        if not speaker_id or disposition != SpeakerDisposition.HOSTILE:
            return False
        
        # Simple rule: If we're confident (HIGH or above) that this is a hostile speaker, terminate
        adaptive_thresholds = self._get_adaptive_confidence_thresholds(speaker_id)
        if confidence >= adaptive_thresholds[IdentificationConfidence.HIGH]:
            logger.info(f"Turn {turn_number} termination: high confidence ({confidence:.3f}) hostile speaker {speaker_id}")
            return True
        
        return False
    
    def _should_welcome_speaker(self, speaker_id: Optional[str],
                              confidence: float,
                              disposition: Optional[SpeakerDisposition],
                              turn_number: int) -> bool:
        """
        Decide if speaker should be welcomed back (for first turn only).
        """
        if (turn_number == 1 and speaker_id and 
            disposition == SpeakerDisposition.FRIENDLY):
            # Use adaptive welcome threshold
            adaptive_thresholds = self._get_adaptive_confidence_thresholds(speaker_id)
            adaptive_welcome_threshold = min(self.welcome_threshold, adaptive_thresholds[IdentificationConfidence.MEDIUM])
            if confidence >= adaptive_welcome_threshold:
                return True
        return False
    
    def _generate_reasoning(self, speaker_id: Optional[str], confidence: float,
                          confidence_level: IdentificationConfidence,
                          disposition: Optional[SpeakerDisposition],
                          turn_number: int, should_terminate: bool, should_welcome: bool) -> str:
        """Generate human-readable reasoning for the decision."""
        if not speaker_id:
            return f"Turn {turn_number}: No speaker match found (confidence: {confidence:.2f})"
        
        base = f"Turn {turn_number}: Identified as {speaker_id} with {confidence_level.value} confidence ({confidence:.2f})"
        
        if disposition:
            base += f", disposition: {disposition.value}"
        
        if should_terminate:
            base += " → TERMINATING (hostile speaker detected)"
        elif should_welcome:
            base += " → WELCOMING (friendly returning speaker)"
        elif disposition == SpeakerDisposition.HOSTILE:
            base += " → MONITORING (potential hostile speaker)"
        
        return base
    
    def _get_cumulative_evidence(self) -> Dict[str, Any]:
        """Get summary of cumulative evidence collected so far."""
        evidence = {
            'turn_count': len(self.current_session_evidence),
            'top_candidates': {}
        }
        
        # Calculate running averages for top candidates
        for speaker_id, similarities in self.turn_similarities.items():
            if similarities:
                evidence['top_candidates'][speaker_id] = {
                    'avg_similarity': float(np.mean(similarities)),
                    'consistency': float(1.0 / (1.0 + np.std(similarities) * 2.0)),
                    'turn_count': len(similarities),
                    'latest_similarity': float(similarities[-1])
                }
        
        return evidence
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of the current identification session."""
        if not self.session_active:
            return {}
        
        # Find most likely speaker
        best_candidate = None
        best_avg_similarity = 0.0
        
        for speaker_id, similarities in self.turn_similarities.items():
            if similarities:
                avg_sim = np.mean(similarities)
                if avg_sim > best_avg_similarity:
                    best_avg_similarity = avg_sim
                    best_candidate = speaker_id
        
        return {
            'session_active': self.session_active,
            'turns_processed': len(self.current_session_evidence),
            'speakers_considered': len(self.turn_similarities),
            'best_candidate': best_candidate,
            'best_avg_similarity': best_avg_similarity,
            'evidence': self.current_session_evidence
        }
