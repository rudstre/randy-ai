"""
Voice biometric profiles for person identification.
Handles acoustic features and statistical analysis for voice recognition.
"""
import os
import json
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime

import numpy as np

logger = logging.getLogger("voice_biometrics")

@dataclass
class VoiceBiometricProfile:
    """Voice biometric data for a person - focused on identification only."""
    
    # === IDENTITY ===
    person_id: str  # Unique identifier for the person
    person_name: Optional[str] = None  # Human-readable name (if provided)
    extracted_names: List[str] = field(default_factory=list)  # Names they've given over time
    
    # === VOICE BIOMETRICS (for identification) ===
    # Statistical features across ALL conversations
    features_mean: Dict[str, float] = field(default_factory=dict)
    features_std: Dict[str, float] = field(default_factory=dict)  # Standard deviation
    features_min: Dict[str, float] = field(default_factory=dict)  # Min values seen
    features_max: Dict[str, float] = field(default_factory=dict)  # Max values seen
    
    # Raw feature history (for detailed analysis) - stores conversation-level aggregates
    feature_history: List[Dict[str, float]] = field(default_factory=list)
    
    # All individual turn features for running statistics
    all_turn_features: List[Dict[str, float]] = field(default_factory=list)
    
    # === BIOMETRIC METADATA ===
    sample_count: int = 0  # Total number of voice samples
    conversation_count: int = 0
    first_recorded: str = field(default_factory=lambda: datetime.now().isoformat())
    last_recorded: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # === LEGACY COMPATIBILITY (for gradual migration) ===
    confidence_scores: List[float] = field(default_factory=list)  # ID confidence per conversation
    
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
    def from_dict(cls, data: Dict[str, Any]) -> 'VoiceBiometricProfile':
        """Create VoiceBiometricProfile from dictionary."""
        return cls(**data)
    
    def update_features(self, new_features: Dict[str, float], confidence: float = 1.0, 
                       sample_count: int = 1, individual_samples: Optional[List[Dict[str, float]]] = None) -> None:
        """Update profile with new voice features from a conversation."""
        # Store conversation-level aggregate
        self.feature_history.append(new_features.copy())
        self.confidence_scores.append(confidence)
        self.sample_count += sample_count
        self.last_recorded = datetime.now().isoformat()
        
        # Store all individual samples for running statistics
        if individual_samples:
            self.all_turn_features.extend(individual_samples)
        
        # Calculate running statistics from ALL samples
        if self.all_turn_features:
            # Initialize if first time
            if not self.features_mean:
                self.features_mean = {}
                self.features_std = {}
                self.features_min = {}
                self.features_max = {}
            
            # Calculate statistics from all individual samples
            # Get all feature names that exist in any sample
            all_feature_names = set()
            for sample in self.all_turn_features:
                all_feature_names.update(sample.keys())
            
            for feature_name in all_feature_names:
                all_sample_values = [sample.get(feature_name, 0.0) for sample in self.all_turn_features if feature_name in sample]
                
                if all_sample_values:
                    self.features_mean[feature_name] = float(np.mean(all_sample_values))
                    self.features_min[feature_name] = float(np.min(all_sample_values))
                    self.features_max[feature_name] = float(np.max(all_sample_values))
                    
                    # Calculate std from all samples (much more meaningful!)
                    if len(all_sample_values) > 1:
                        self.features_std[feature_name] = float(np.std(all_sample_values))
                    else:
                        self.features_std[feature_name] = 0.0
        else:
            # Fallback to conversation-level features if no individual samples available
            if not self.features_mean:
                self.features_mean = new_features.copy()
                self.features_std = {k: 0.0 for k in new_features.keys()}
                self.features_min = new_features.copy()
                self.features_max = new_features.copy()
    
    def recompute_statistics(self) -> bool:
        """
        Recompute statistical features from existing all_turn_features data.
        Useful for fixing profiles that have empty features_mean due to bugs.
        
        Returns:
            True if statistics were recomputed, False if no data available
        """
        if not self.all_turn_features:
            return False
        
        # Clear existing statistics
        self.features_mean = {}
        self.features_std = {}
        self.features_min = {}
        self.features_max = {}
        
        # Get all feature names that exist in any sample
        all_feature_names = set()
        for sample in self.all_turn_features:
            all_feature_names.update(sample.keys())
        
        # Calculate statistics for each feature
        for feature_name in all_feature_names:
            all_sample_values = [sample.get(feature_name, 0.0) for sample in self.all_turn_features if feature_name in sample]
            
            if all_sample_values:
                self.features_mean[feature_name] = float(np.mean(all_sample_values))
                self.features_min[feature_name] = float(np.min(all_sample_values))
                self.features_max[feature_name] = float(np.max(all_sample_values))
                
                # Calculate std from all samples
                if len(all_sample_values) > 1:
                    self.features_std[feature_name] = float(np.std(all_sample_values))
                else:
                    self.features_std[feature_name] = 0.0
        
        logger.info(f"Recomputed statistics for {len(self.features_mean)} features from {len(self.all_turn_features)} samples")
        return True
    
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
        # Default scale_factor=8.0: 1 sample = ~18%, 3 samples = ~45%, 8 samples = ~80%, 15 samples = ~95%
        base_confidence = 1.0 - np.exp(-sample_size / scale_factor)
        
        # Stability bonus: reward consistent features across samples
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
            similarity_calculator = SimilarityCalculator()
        
        # Check if adaptive tolerance is enabled
        from ....config import DEFAULT_ADAPTIVE_TOLERANCE_ENABLED
        if DEFAULT_ADAPTIVE_TOLERANCE_ENABLED:
            # Get adaptive tolerance modifier based on data confidence
            tolerance_modifier = self.get_adaptive_tolerance_modifier()
        else:
            tolerance_modifier = 1.0  # No adaptation
        
        # Get the similarity result
        result = similarity_calculator.calculate_similarity(
            self.features_mean, 
            self.features_std, 
            other_features,
            adaptive_tolerance_modifier=tolerance_modifier
        )
        
        # Return just the similarity score for backward compatibility
        # The detailed result is available if needed later
        return result.similarity
