"""
Voice similarity calculation module optimized for RPi4.
Uses mathematically sound approaches with uncertainty quantification.
"""
import logging
from typing import Dict, Tuple, Optional, List
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger("voice_similarity")

# Research-based feature importance with optimized groupings for RPi4
FEATURE_GROUPS = {
    # Core spectral features (most discriminative, always computed)
    "core_spectral": {
        "features": ["mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4"],
        "weight": 1.0,
        "outlier_threshold": 2.5  # Stricter for core features
    },
    
    # Prosodic features (pitch-related, highly personal)
    "prosodic": {
        "features": ["pitch_mean", "pitch_std", "pitch_range"],
        "weight": 0.9,
        "outlier_threshold": 3.0  # More lenient for emotional variation
    },
    
    # Formant features (vocal tract shape, very discriminative)
    "formants": {
        "features": ["formant_1_freq", "formant_2_freq", "formant_3_freq", 
                   "formant_1_bandwidth", "formant_2_bandwidth"],
        "weight": 0.95,
        "outlier_threshold": 2.8
    },
    
    # Voice quality (health/emotion sensitive)
    "voice_quality": {
        "features": ["jitter", "shimmer", "hnr"],
        "weight": 0.7,
        "outlier_threshold": 3.5  # Very lenient - varies with health/emotion
    },
    
    # Extended spectral (helpful but not critical)
    "extended_spectral": {
        "features": ["mfcc_5", "mfcc_6", "mfcc_7", "mfcc_8", "mfcc_9", 
                   "mfcc_10", "mfcc_11", "mfcc_12"],
        "weight": 0.6,
        "outlier_threshold": 3.0
    },
    
    # Energy features (least reliable, environment dependent)
    "energy": {
        "features": ["loudness", "rms_energy", "spectral_centroid", 
                   "spectral_rolloff", "spectral_flux", "zcr"],
        "weight": 0.4,
        "outlier_threshold": 4.0  # Very lenient - depends on mic/environment
    }
}

@dataclass
class SimilarityResult:
    """Enhanced similarity result with uncertainty and diagnostics."""
    similarity: float          # Final similarity score [0.0, 1.0]
    confidence: float         # Confidence in the result [0.0, 1.0]
    feature_count: int        # Number of features successfully compared
    outliers_detected: List[str]  # Features flagged as outliers
    group_scores: Dict[str, float]  # Per-group similarity scores
    uncertainty: float        # Estimated uncertainty in similarity
    method_used: str         # "statistical" or "fallback"


class SimilarityCalculator:
    """
    Voice similarity calculator optimized for RPi4 performance.
    
    Key improvements:
    - Gaussian similarity functions instead of linear
    - Feature grouping with adaptive outlier thresholds
    - Uncertainty quantification using Bayesian approach
    - Efficient computation for real-time use
    - Robust handling of missing/corrupted features
    """
    
    def __init__(self, 
                 base_sigma_scale: float = 2.0,
                 min_features_required: int = 8,
                 uncertainty_penalty: float = 0.1):
        """
        Initialize similarity calculator.
        
        Args:
            base_sigma_scale: Base scaling for Gaussian similarity (lower = stricter)
            min_features_required: Minimum features needed for reliable comparison
            uncertainty_penalty: Penalty factor for high uncertainty results
        """
        self.base_sigma_scale = base_sigma_scale
        self.min_features_required = min_features_required
        self.uncertainty_penalty = uncertainty_penalty
        
        # Pre-compute feature group mappings for efficiency
        self.feature_to_group = {}
        for group_name, group_info in FEATURE_GROUPS.items():
            for feature in group_info["features"]:
                self.feature_to_group[feature] = group_name
    
    def calculate_similarity(self, 
                           profile_mean: Dict[str, float],
                           profile_std: Dict[str, float], 
                           other_features: Dict[str, float],
                           adaptive_tolerance_modifier: float = 1.0) -> SimilarityResult:
        """
        Calculate similarity using Gaussian-based approach with uncertainty.
        
        Args:
            profile_mean: Mean features from stored profile
            profile_std: Standard deviation features from stored profile  
            other_features: Features to compare against profile
            adaptive_tolerance_modifier: Multiplier for tolerance (>1.0 = more lenient)
            
        Returns:
            SimilarityResult with similarity, confidence, and diagnostics
        """
        if not profile_mean or not other_features:
            return SimilarityResult(
                similarity=0.0, confidence=0.0, feature_count=0,
                outliers_detected=[], group_scores={}, uncertainty=1.0,
                method_used="none"
            )
        
        # Group-based similarity calculation
        group_similarities = {}
        group_weights = {}
        group_feature_counts = {}
        all_outliers = []
        total_features_used = 0
        has_statistical_data = any(std > 0 for std in (profile_std or {}).values())
        
        # Process each feature group
        for group_name, group_info in FEATURE_GROUPS.items():
            group_features = group_info["features"]
            group_weight = group_info["weight"]
            outlier_threshold = group_info["outlier_threshold"]
            
            # Find available features in this group
            available_features = [f for f in group_features 
                                if f in profile_mean and f in other_features]
            
            if not available_features:
                continue
            
            # Calculate group similarity
            group_sim, group_outliers = self._calculate_group_similarity(
                available_features, profile_mean, profile_std, 
                other_features, outlier_threshold, adaptive_tolerance_modifier
            )
            
            if group_sim is not None:  # Group had valid features
                group_similarities[group_name] = group_sim
                group_weights[group_name] = group_weight
                group_feature_counts[group_name] = len(available_features) - len(group_outliers)
                all_outliers.extend(group_outliers)
                total_features_used += len(available_features) - len(group_outliers)
        
        # Check if we have enough features for reliable comparison
        if total_features_used < self.min_features_required:
            uncertainty = 1.0 - (total_features_used / self.min_features_required)
            return SimilarityResult(
                similarity=0.0, confidence=0.0, feature_count=total_features_used,
                outliers_detected=all_outliers, group_scores=group_similarities,
                uncertainty=uncertainty, method_used="insufficient_data"
            )
        
        # Combine group similarities using weighted harmonic mean for robustness
        final_similarity = self._combine_group_similarities(group_similarities, group_weights)
        
        # Calculate confidence and uncertainty
        confidence = self._calculate_confidence(
            group_feature_counts, total_features_used, has_statistical_data
        )
        uncertainty = self._calculate_uncertainty(
            group_similarities, group_feature_counts, adaptive_tolerance_modifier
        )
        
        # Apply uncertainty penalty
        adjusted_similarity = final_similarity * (1.0 - uncertainty * self.uncertainty_penalty)
        adjusted_similarity = max(0.0, min(1.0, adjusted_similarity))
        
        method_used = "statistical" if has_statistical_data else "fallback"
        
        result = SimilarityResult(
            similarity=adjusted_similarity,
            confidence=confidence,
            feature_count=total_features_used,
            outliers_detected=all_outliers,
            group_scores=group_similarities,
            uncertainty=uncertainty,
            method_used=method_used
        )
        
        logger.debug(f"Calculated similarity: {adjusted_similarity:.4f} "
                    f"(confidence: {confidence:.3f}, uncertainty: {uncertainty:.3f}, "
                    f"features: {total_features_used}, outliers: {len(all_outliers)})")
        
        return result
    
    def _calculate_group_similarity(self, features: List[str], profile_mean: Dict[str, float],
                                  profile_std: Dict[str, float], other_features: Dict[str, float],
                                  outlier_threshold: float, adaptive_tolerance: float) -> Tuple[Optional[float], List[str]]:
        """Calculate similarity for a group of related features."""
        valid_similarities = []
        outliers = []
        
        for feature in features:
            profile_val = profile_mean[feature]
            other_val = other_features[feature]
            profile_std_val = profile_std.get(feature, 0.0)
            
            # Check for outliers with group-specific threshold
            if self._is_outlier_adaptive(profile_val, profile_std_val, other_val, outlier_threshold):
                outliers.append(feature)
                continue
            
            # Calculate feature similarity using Gaussian approach
            if profile_std_val > 0:
                similarity = self._gaussian_similarity(profile_val, profile_std_val, other_val, adaptive_tolerance)
            else:
                similarity = self._robust_fallback_similarity(profile_val, other_val, feature)
            
            valid_similarities.append(similarity)
        
        if not valid_similarities:
            return None, outliers
        
        # Use geometric mean for robustness against outliers
        group_similarity = np.power(np.prod(valid_similarities), 1.0 / len(valid_similarities))
        return float(group_similarity), outliers
    
    def _gaussian_similarity(self, profile_val: float, profile_std: float, 
                           other_val: float, adaptive_tolerance: float) -> float:
        """Calculate similarity using Gaussian function - mathematically sound approach."""
        if profile_std <= 0:
            return 1.0 if abs(profile_val - other_val) < 1e-6 else 0.0
        
        # Adaptive scaling
        effective_scale = self.base_sigma_scale * adaptive_tolerance
        
        # Gaussian similarity: exp(-0.5 * (z_score / scale)^2)
        z_score = abs(other_val - profile_val) / profile_std
        similarity = np.exp(-0.5 * (z_score / effective_scale) ** 2)
        
        return float(similarity)
    
    def _robust_fallback_similarity(self, profile_val: float, other_val: float, feature: str) -> float:
        """Robust fallback for features without statistical data."""
        # Feature-specific robust tolerances based on voice research
        robust_tolerances = {
            'pitch_mean': 0.15, 'pitch_std': 0.25, 'pitch_range': 0.20,
            'mfcc_1': 0.20, 'mfcc_2': 0.20, 'mfcc_3': 0.25, 'mfcc_4': 0.30,
            'formant_1_freq': 0.12, 'formant_2_freq': 0.15, 'formant_3_freq': 0.18,
            'jitter': 0.40, 'shimmer': 0.35, 'hnr': 0.30,
            'loudness': 0.25, 'rms_energy': 0.30,
        }
        
        tolerance = robust_tolerances.get(feature, 0.25)
        
        # Prevent division by zero
        scale = max(abs(profile_val), abs(other_val), 0.001)
        normalized_diff = abs(profile_val - other_val) / scale
        
        # Gaussian-like fallback
        similarity = np.exp(-2.0 * (normalized_diff / tolerance) ** 2)
        return float(similarity)
    
    def _is_outlier_adaptive(self, profile_val: float, profile_std: float, 
                           other_val: float, threshold: float) -> bool:
        """Adaptive outlier detection with feature-group-specific thresholds."""
        if profile_std <= 0:
            # For features without std data, use magnitude-based detection
            scale = max(abs(profile_val), 0.001)
            ratio = abs(other_val - profile_val) / scale
            return ratio > (threshold / 2.0)  # Less aggressive without std data
        
        z_score = abs(other_val - profile_val) / profile_std
        return z_score > threshold
    
    def _combine_group_similarities(self, group_similarities: Dict[str, float], 
                                  group_weights: Dict[str, float]) -> float:
        """Combine group similarities using weighted harmonic mean for robustness."""
        if not group_similarities:
            return 0.0
        
        # Use harmonic mean for robustness - severely penalizes any very low scores
        weighted_sum = 0.0
        total_weight = 0.0
        
        for group, similarity in group_similarities.items():
            weight = group_weights[group]
            if similarity > 0.001:  # Avoid division by zero
                weighted_sum += weight / similarity
                total_weight += weight
        
        if weighted_sum == 0 or total_weight == 0:
            return 0.0
        
        harmonic_mean = total_weight / weighted_sum
        return min(1.0, harmonic_mean)
    
    def _calculate_confidence(self, group_feature_counts: Dict[str, int], 
                            total_features: int, has_statistical_data: bool) -> float:
        """Calculate confidence in the similarity result."""
        # Base confidence from feature coverage
        max_possible_features = sum(len(info["features"]) for info in FEATURE_GROUPS.values())
        coverage_confidence = min(1.0, total_features / max_possible_features)
        
        # Bonus for having core features
        core_features = group_feature_counts.get("core_spectral", 0)
        core_bonus = min(0.2, core_features / 4 * 0.2)  # Up to 20% bonus
        
        # Bonus for statistical data
        statistical_bonus = 0.1 if has_statistical_data else 0.0
        
        # Group diversity bonus - having features from multiple groups is good
        diversity_bonus = min(0.15, len(group_feature_counts) / len(FEATURE_GROUPS) * 0.15)
        
        confidence = coverage_confidence + core_bonus + statistical_bonus + diversity_bonus
        return min(1.0, confidence)
    
    def _calculate_uncertainty(self, group_similarities: Dict[str, float], 
                             group_feature_counts: Dict[str, int], 
                             adaptive_tolerance: float) -> float:
        """Calculate uncertainty in the similarity result."""
        if not group_similarities:
            return 1.0
        
        # Variance in group similarities indicates uncertainty
        similarities = list(group_similarities.values())
        if len(similarities) > 1:
            variance_uncertainty = np.std(similarities)
        else:
            variance_uncertainty = 0.0
        
        # Low feature count increases uncertainty
        total_features = sum(group_feature_counts.values())
        feature_uncertainty = max(0.0, 1.0 - total_features / 20.0)  # Uncertainty if < 20 features
        
        # High adaptive tolerance indicates we're being lenient due to limited data
        tolerance_uncertainty = max(0.0, (adaptive_tolerance - 1.0) * 0.3)
        
        # Combine uncertainties
        total_uncertainty = min(1.0, variance_uncertainty + feature_uncertainty + tolerance_uncertainty)
        return float(total_uncertainty)
