"""
Voice profile management system for speaker identification and tracking.
Stores acoustic features and metadata for people the system has conversed with.
"""
import os
import json
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np

# Feature weights moved to similarity.py

logger = logging.getLogger("voice_profiles")

@dataclass
class VoiceProfile:
    """Represents a stored voice profile for a speaker."""
    
    # Identification
    speaker_id: str  # Unique identifier for the speaker
    speaker_name: Optional[str] = None  # Human-readable name (if provided)
    
    # Acoustic features (means across all conversations)
    features_mean: Dict[str, float] = field(default_factory=dict)
    features_std: Dict[str, float] = field(default_factory=dict)  # Standard deviation
    features_min: Dict[str, float] = field(default_factory=dict)  # Min values seen
    features_max: Dict[str, float] = field(default_factory=dict)  # Max values seen
    
    # Historical data
    conversation_count: int = 0
    total_turns: int = 0
    first_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    last_seen: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Raw feature history (for detailed analysis) - stores conversation-level aggregates
    feature_history: List[Dict[str, float]] = field(default_factory=list)
    
    # All individual turn features for running statistics
    all_turn_features: List[Dict[str, float]] = field(default_factory=list)
    
    # Conversation metadata
    conversation_summaries: List[str] = field(default_factory=list)  # Brief summaries
    confidence_scores: List[float] = field(default_factory=list)  # ID confidence per conversation
    opinion_history: List[str] = field(default_factory=list)  # List of opinion words from each conversation
    score_history: List[float] = field(default_factory=list)  # List of overall scores from each conversation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            else:
                return obj
        
        result = convert_numpy_types(data)
        return result if isinstance(result, dict) else {}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VoiceProfile':
        """Create VoiceProfile from dictionary."""
        return cls(**data)
    
    def update_features(self, new_features: Dict[str, float], confidence: float = 1.0, turn_count: int = 1, individual_turns: Optional[List[Dict[str, float]]] = None) -> None:
        """Update profile with new voice features from a conversation."""
        # Store conversation-level aggregate
        self.feature_history.append(new_features.copy())
        self.confidence_scores.append(confidence)
        self.total_turns += turn_count
        self.last_seen = datetime.now().isoformat()
        
        # Store all individual turn features for running statistics
        if individual_turns:
            self.all_turn_features.extend(individual_turns)
        
        # Calculate running statistics from ALL turns
        if self.all_turn_features:
            # Initialize if first time
            if not self.features_mean:
                self.features_mean = {}
                self.features_std = {}
                self.features_min = {}
                self.features_max = {}
            
            # Calculate statistics from all individual turns
            for feature_name in new_features.keys():
                all_turn_values = [turn.get(feature_name, 0.0) for turn in self.all_turn_features if feature_name in turn]
                
                if all_turn_values:
                    self.features_mean[feature_name] = float(np.mean(all_turn_values))
                    self.features_min[feature_name] = float(np.min(all_turn_values))
                    self.features_max[feature_name] = float(np.max(all_turn_values))
                    
                    # Calculate std from all turns (much more meaningful!)
                    if len(all_turn_values) > 1:
                        self.features_std[feature_name] = float(np.std(all_turn_values))
                    else:
                        self.features_std[feature_name] = 0.0
        else:
            # Fallback to conversation-level features if no individual turns available
            if not self.features_mean:
                self.features_mean = new_features.copy()
                self.features_std = {k: 0.0 for k in new_features.keys()}
                self.features_min = new_features.copy()
                self.features_max = new_features.copy()
    
    def get_feature_vector(self) -> np.ndarray:
        """Get normalized feature vector for comparison."""
        # Use consistent feature order based on sorted keys
        features = []
        if self.features_mean:
            for feature_name in sorted(self.features_mean.keys()):
                features.append(self.features_mean.get(feature_name, 0.0))
        return np.array(features, dtype=np.float32)
    
    def get_data_confidence(self, scale_factor: Optional[float] = None) -> float:
        """
        Calculate confidence in our statistical data based on sample size and stability.
        
        Args:
            scale_factor: Optional override for confidence scaling (defaults to config value)
        
        Returns:
            Float between 0.0 and 1.0 representing confidence in the profile data.
            Higher values indicate more reliable statistics.
        """
        if not self.all_turn_features:
            return 0.0
        
        sample_size = len(self.all_turn_features)
        
        # Use provided scale factor or default from config
        if scale_factor is None:
            from ....config import DEFAULT_DATA_CONFIDENCE_SCALE
            scale_factor = DEFAULT_DATA_CONFIDENCE_SCALE
        
        # Base confidence from sample size (asymptotic approach to 1.0)
        # Using the formula: 1 - exp(-sample_size / scale_factor)
        # Default scale_factor=8.0: 1 turn = ~18%, 3 turns = ~45%, 8 turns = ~80%, 15 turns = ~95%
        base_confidence = 1.0 - np.exp(-sample_size / scale_factor)
        
        # Stability bonus: reward consistent features across turns
        stability_score = 1.0
        if sample_size > 1 and self.features_std:
            # Calculate coefficient of variation for key features
            cv_scores = []
            for feature_name in ['pitch_mean', 'mfcc_1', 'mfcc_2', 'loudness']:
                if (feature_name in self.features_mean and 
                    feature_name in self.features_std and 
                    self.features_mean[feature_name] != 0):
                    cv = self.features_std[feature_name] / abs(self.features_mean[feature_name])
                    # Convert CV to stability score (lower CV = higher stability)
                    cv_scores.append(max(0.0, 1.0 - min(cv, 1.0)))
            
            if cv_scores:
                stability_score = np.mean(cv_scores)
        
        # Combine base confidence with stability
        final_confidence = base_confidence * (0.7 + 0.3 * stability_score)
        
        return min(1.0, max(0.0, final_confidence))
    
    def get_adaptive_tolerance_modifier(self, leniency_factor: Optional[float] = None) -> float:
        """
        Get tolerance modifier for adaptive thresholds.
        
        Args:
            leniency_factor: Optional override for leniency (defaults to config value)
        
        Returns:
            Multiplier for similarity thresholds. Values > 1.0 make thresholds more lenient,
            values < 1.0 make them stricter.
        """
        data_confidence = self.get_data_confidence()
        
        # Use provided leniency factor or default from config
        if leniency_factor is None:
            from ....config import DEFAULT_ADAPTIVE_LENIENCY_FACTOR
            leniency_factor = DEFAULT_ADAPTIVE_LENIENCY_FACTOR
        
        # When we have low confidence in our data, be more lenient
        # When we have high confidence, use normal thresholds
        # Formula: 1.0 + (1.0 - data_confidence) * leniency_factor
        tolerance_modifier = 1.0 + (1.0 - data_confidence) * leniency_factor
        
        return tolerance_modifier

    def calculate_similarity(self, other_features: Dict[str, float], similarity_calculator=None) -> float:
        """Calculate similarity using external similarity calculator with adaptive tolerance."""
        if not self.features_mean:
            return 0.0
        
        # Use provided calculator or create default one
        if similarity_calculator is None:
            from .similarity import SimilarityCalculator
            similarity_calculator = SimilarityCalculator(sigma_tolerance=5.0)
        
        # Check if adaptive tolerance is enabled
        from ....config import DEFAULT_ADAPTIVE_TOLERANCE_ENABLED
        if DEFAULT_ADAPTIVE_TOLERANCE_ENABLED:
            # Get adaptive tolerance modifier based on data confidence
            tolerance_modifier = self.get_adaptive_tolerance_modifier()
        else:
            tolerance_modifier = 1.0  # No adaptation
        
        return similarity_calculator.calculate_similarity(
            self.features_mean, 
            self.features_std, 
            other_features,
            adaptive_tolerance_modifier=tolerance_modifier
        )
    


class VoiceProfileManager:
    """Manages storage and retrieval of voice profiles."""
    
    def __init__(self, profiles_dir: str = "./_voice_profiles", sigma_tolerance: float = 5.0, outlier_threshold: float = 3.0):
        self.profiles_dir = profiles_dir
        self.profiles: Dict[str, VoiceProfile] = {}
        
        # Create similarity calculator with configurable tolerance and outlier detection
        from .similarity import SimilarityCalculator
        self.similarity_calculator = SimilarityCalculator(
            sigma_tolerance=sigma_tolerance,
            outlier_threshold=outlier_threshold
        )
        
        # Create directory
        os.makedirs(self.profiles_dir, exist_ok=True)
        
        # Load existing profiles
        self.load_profiles()
    
    def _get_profile_path(self, speaker_id: str) -> str:
        """Get file path for a speaker's profile."""
        return os.path.join(self.profiles_dir, f"{speaker_id}.json")
    
    def save_profile(self, profile: VoiceProfile) -> None:
        """Save a voice profile to disk."""
        profile_path = self._get_profile_path(profile.speaker_id)
        
        try:
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(profile.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Saved voice profile for speaker {profile.speaker_id}")
        except Exception as e:
            logger.error(f"Failed to save profile for {profile.speaker_id}: {e}")
    
    def load_profile(self, speaker_id: str) -> Optional[VoiceProfile]:
        """Load a voice profile from disk."""
        profile_path = self._get_profile_path(speaker_id)
        
        if not os.path.exists(profile_path):
            return None
        
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            profile = VoiceProfile.from_dict(data)
            logger.info(f"Loaded voice profile for speaker {speaker_id}")
            return profile
        except Exception as e:
            logger.error(f"Failed to load profile for {speaker_id}: {e}")
            return None
    
    def load_profiles(self) -> None:
        """Load all voice profiles from disk."""
        if not os.path.exists(self.profiles_dir):
            return
        
        for filename in os.listdir(self.profiles_dir):
            if filename.endswith('.json'):
                speaker_id = filename[:-5]  # Remove .json extension
                profile = self.load_profile(speaker_id)
                if profile:
                    self.profiles[speaker_id] = profile
        
        logger.info(f"Loaded {len(self.profiles)} voice profiles")
    
    def identify_speaker(self, features: Dict[str, float], 
                        similarity_threshold: float = 0.7) -> Tuple[Optional[str], float]:
        """
        Identify speaker based on voice features.
        
        Args:
            features: Voice features to match
            similarity_threshold: Minimum similarity score to consider a match
            
        Returns:
            Tuple of (speaker_id, confidence_score) or (None, 0.0) if no match
        """
        if not self.profiles:
            return None, 0.0
        
        best_match = None
        best_score = 0.0
        
        for speaker_id, profile in self.profiles.items():
            similarity = profile.calculate_similarity(features, self.similarity_calculator)
            
            if similarity > best_score:
                best_score = similarity
                best_match = speaker_id
        
        if best_score >= similarity_threshold:
            logger.info(f"Identified speaker {best_match} with confidence {best_score:.3f}")
            return best_match, best_score
        else:
            logger.info(f"No speaker match found (best score: {best_score:.3f})")
            return None, best_score
    
    def create_new_profile(self, features: Dict[str, float], 
                          speaker_name: Optional[str] = None,
                          initial_assessment: Optional[str] = None,
                          conversation_summary: str = "",
                          opinion_word: str = "",
                          overall_score: float = 0.0,
                          turn_count: int = 1,
                          individual_turns: Optional[List[Dict[str, float]]] = None) -> str:
        """
        Create a new voice profile for an unknown speaker.
        
        Args:
            features: Initial voice features
            speaker_name: Optional human-readable name
            conversation_summary: Summary of the first conversation
            opinion_word: Opinion detected in first conversation
            overall_score: Overall score from first conversation
            
        Returns:
            speaker_id: Generated unique identifier
        """
        # Generate unique speaker ID and name
        timestamp = str(int(time.time()))
        speaker_id = f"speaker_{timestamp}"
        
        # Ensure uniqueness
        counter = 1
        while speaker_id in self.profiles:
            speaker_id = f"speaker_{timestamp}_{counter}"
            counter += 1
        
        # Generate a human-readable name if none provided
        if not speaker_name:
            speaker_count = len(self.profiles) + 1
            speaker_name = f"Person #{speaker_count}"
        
        # Create profile
        profile = VoiceProfile(
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            conversation_count=1
        )
        
        # Add initial features
        profile.update_features(features, confidence=1.0, turn_count=turn_count, individual_turns=individual_turns)
        
        # Add conversation metadata if provided
        if conversation_summary:
            profile.conversation_summaries.append(conversation_summary)
        
        if opinion_word:
            profile.opinion_history.append(opinion_word)
        
        profile.score_history.append(overall_score)
        
        # Store and save
        self.profiles[speaker_id] = profile
        self.save_profile(profile)
        
        logger.info(f"Created new voice profile: {speaker_id}")
        return speaker_id
    
    def update_profile(self, speaker_id: str, features: Dict[str, float], 
                      confidence: float = 1.0, conversation_summary: str = "",
                      opinion_word: str = "", overall_score: float = 0.0,
                      turn_count: int = 1, individual_turns: Optional[List[Dict[str, float]]] = None) -> None:
        """Update an existing voice profile with new conversation data."""
        if speaker_id not in self.profiles:
            logger.warning(f"Cannot update unknown speaker: {speaker_id}")
            return
        
        profile = self.profiles[speaker_id]
        profile.update_features(features, confidence, turn_count, individual_turns)
        profile.conversation_count += 1
        
        if conversation_summary:
            profile.conversation_summaries.append(conversation_summary)
        
        if opinion_word:
            profile.opinion_history.append(opinion_word)
        
        profile.score_history.append(overall_score)
        
        # Save updated profile
        self.save_profile(profile)
        logger.info(f"Updated voice profile for {speaker_id}")
    
    def get_speaker_info(self, speaker_id: str) -> Optional[Dict[str, Any]]:
        """Get readable information about a speaker."""
        if speaker_id not in self.profiles:
            return None
        
        profile = self.profiles[speaker_id]
        
        # Simple analysis of recent interactions
        recent_opinions = profile.opinion_history[-3:] if profile.opinion_history else []
        recent_scores = profile.score_history[-3:] if profile.score_history else []
        
        # Determine if generally hostile based on recent interactions
        negative_opinions = ["negative", "cold", "hostile", "rude", "dismissive"]
        recent_negative_count = sum(1 for opinion in recent_opinions if opinion in negative_opinions)
        avg_recent_score = np.mean(recent_scores) if recent_scores else 0.0
        
        # Simple hostility check: any negative opinions OR consistently low scores
        is_hostile = (recent_negative_count >= 1 or 
                     (avg_recent_score < -0.5 and len(recent_scores) >= 1))
        
        return {
            "speaker_id": speaker_id,
            "name": profile.speaker_name or "Unknown",
            "conversations": profile.conversation_count,
            "total_turns": profile.total_turns,
            "first_seen": profile.first_seen,
            "last_seen": profile.last_seen,
            "avg_confidence": np.mean(profile.confidence_scores) if profile.confidence_scores else 0.0,
            "recent_opinions": recent_opinions,
            "recent_scores": recent_scores,
            "avg_recent_score": avg_recent_score,
            "is_hostile": is_hostile,
            "negative_interaction_count": recent_negative_count
        }
    
    
    def list_all_speakers(self) -> List[Dict[str, Any]]:
        """Get information about all known speakers."""
        return [info for speaker_id in self.profiles.keys() 
                if (info := self.get_speaker_info(speaker_id)) is not None]
    
    def delete_profile(self, speaker_id: str) -> bool:
        """Delete a voice profile."""
        if speaker_id not in self.profiles:
            return False
        
        # Remove from memory
        del self.profiles[speaker_id]
        
        # Remove file
        profile_path = self._get_profile_path(speaker_id)
        try:
            os.remove(profile_path)
            logger.info(f"Deleted voice profile for {speaker_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete profile file for {speaker_id}: {e}")
            return False
    
    def aggregate_conversation_features(self, turns: List) -> Dict[str, float]:
        """
        Aggregate voice features across all turns in a conversation.
        
        Args:
            turns: List of Turn objects with .features attribute
            
        Returns:
            Aggregated features dictionary
        """
        if not turns:
            return {}
        
        # Collect all available feature names from all turns
        all_feature_names = set()
        for turn in turns:
            if hasattr(turn, 'features') and turn.features:
                all_feature_names.update(turn.features.keys())
        
        # Aggregate each feature across turns
        all_features = {}
        for feature_name in all_feature_names:
            values = []
            for turn in turns:
                if hasattr(turn, 'features') and feature_name in turn.features:
                    values.append(turn.features[feature_name])
            
            if values:
                # Use median as it's more robust to outliers
                all_features[feature_name] = float(np.median(values))
            else:
                all_features[feature_name] = 0.0
        
        logger.info(f"Aggregated features from {len(turns)} turns: {all_features}")
        return all_features
