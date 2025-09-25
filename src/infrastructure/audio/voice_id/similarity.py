"""
Voice similarity calculation module.
Provides configurable algorithms for comparing voice features between speakers.
"""
import logging
from typing import Dict
import numpy as np

logger = logging.getLogger("voice_similarity")

# Feature importance weights for speaker identification (based on research)
SPEAKER_ID_FEATURE_WEIGHTS = {
    # Tier 1: Highest discriminative power (weight 1.0)
    "mfcc_1": 1.0, "mfcc_2": 1.0, "mfcc_3": 1.0, "mfcc_4": 1.0,
    "formant_1_freq": 1.0, "formant_2_freq": 1.0, "formant_3_freq": 1.0,
    "pitch_mean": 1.0,
    
    # Tier 2: High discriminative power (weight 0.8)
    "mfcc_5": 0.8, "mfcc_6": 0.8, "mfcc_7": 0.8, "mfcc_8": 0.8,
    "formant_1_bandwidth": 0.8, "formant_2_bandwidth": 0.8,
    "pitch_std": 0.8, "pitch_range": 0.8,
    "jitter": 0.8, "shimmer": 0.8, "hnr": 0.8,
    
    # Tier 3: Medium discriminative power (weight 0.6)
    "mfcc_9": 0.6, "mfcc_10": 0.6, "mfcc_11": 0.6, "mfcc_12": 0.6,
    "spectral_centroid": 0.6, "spectral_rolloff": 0.6,
    "loudness": 0.6,
    
    # Tier 4: Lower discriminative power (weight 0.4)
    "spectral_flux": 0.4, "rms_energy": 0.4, "zcr": 0.4,
}


class SimilarityCalculator:
    """
    Calculates voice similarity between speakers using statistical methods.
    Uses actual measured standard deviations when available for adaptive tolerance.
    Includes outlier detection to ignore anomalous feature values.
    """
    
    def __init__(self, sigma_tolerance: float = 5.0, outlier_threshold: float = 3.0):
        """
        Initialize similarity calculator.
        
        Args:
            sigma_tolerance: Number of standard deviations to allow for same speaker
                           Higher = more forgiving, Lower = more strict
            outlier_threshold: Z-score threshold for outlier detection
                             Features beyond this threshold are ignored in similarity
        """
        self.sigma_tolerance = sigma_tolerance
        self.outlier_threshold = outlier_threshold
    
    def calculate_similarity(self, 
                           profile_mean: Dict[str, float],
                           profile_std: Dict[str, float], 
                           other_features: Dict[str, float]) -> float:
        """
        Calculate similarity between profile and other features.
        
        Args:
            profile_mean: Mean features from stored profile
            profile_std: Standard deviation features from stored profile  
            other_features: Features to compare against profile
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not profile_mean or not other_features:
            return 0.0
        
        similarities = []
        total_weight = 0.0
        outliers_detected = []
        
        for feature_name in profile_mean.keys():
            if feature_name not in other_features:
                continue
                
            # Get feature weight (default to 0.3 for unknown features)
            weight = SPEAKER_ID_FEATURE_WEIGHTS.get(feature_name, 0.3)
            
            profile_value = profile_mean[feature_name]
            other_value = other_features[feature_name]
            
            # Check for outliers first
            is_outlier = self._is_outlier(profile_value, profile_std.get(feature_name, 0), other_value)
            
            if is_outlier:
                outliers_detected.append(feature_name)
                logger.debug(f"  Outlier detected in {feature_name}: "
                           f"profile={profile_value:.2f}, other={other_value:.2f} - IGNORING")
                continue  # Skip this feature entirely
            
            # Calculate feature similarity for non-outliers
            if (profile_std and feature_name in profile_std and 
                profile_std[feature_name] > 0):
                # Use data-driven approach with actual measured standard deviation
                feature_sim = self._statistical_similarity(
                    profile_value, profile_std[feature_name], other_value
                )
            else:
                # Fallback to percentage-based tolerance
                feature_sim = self._percentage_similarity(profile_value, other_value, feature_name)
            
            similarities.append(feature_sim * weight)
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        # Calculate weighted average
        weighted_score = sum(similarities) / total_weight
        
        # Apply conservative penalty for profiles with limited data
        if not any(std > 0 for std in (profile_std or {}).values()):
            weighted_score *= 0.85  # Penalty for no statistical data
        
        final_score = max(0.0, min(1.0, weighted_score))
        
        # Log result with outlier information
        outlier_info = f", outliers: {outliers_detected}" if outliers_detected else ""
        logger.debug(f"Similarity calculation: {final_score:.4f} "
                    f"(features: {len(similarities)}, total_weight: {total_weight:.2f}{outlier_info})")
        
        if outliers_detected:
            logger.info(f"Outlier features ignored: {outliers_detected}")
        
        return final_score
    
    def _statistical_similarity(self, profile_value: float, profile_std: float, other_value: float) -> float:
        """Calculate similarity using z-score and sigma tolerance."""
        z_score = abs(other_value - profile_value) / profile_std
        
        # Convert z-score to similarity using configurable sigma tolerance
        # More generous tolerance since voice naturally varies
        similarity = max(0.0, 1.0 - (z_score / self.sigma_tolerance))
        
        logger.debug(f"  Statistical: profile={profile_value:.2f}, other={other_value:.2f}, "
                    f"std={profile_std:.2f}, z={z_score:.2f}, sim={similarity:.4f}")
        
        return similarity
    
    def _percentage_similarity(self, profile_value: float, other_value: float, feature_name: str) -> float:
        """Fallback percentage-based similarity for features without std data."""
        # Generous fallback tolerances
        feature_tolerances = {
            'pitch_mean': 0.30, 'pitch_std': 0.60, 'pitch_range': 0.50,
            'mfcc_1': 0.40, 'mfcc_2': 0.40, 'mfcc_3': 0.50, 'mfcc_4': 0.60,
            'jitter': 0.70, 'shimmer': 0.50, 'hnr': 0.70,
            'loudness': 0.40, 'rms_energy': 0.45,
        }
        
        tolerance = feature_tolerances.get(feature_name, 0.50)  # Generous default
        max_allowed_diff = max(abs(profile_value), abs(other_value), 0.001) * tolerance
        diff = abs(profile_value - other_value)
        similarity = max(0.0, 1.0 - (diff / max_allowed_diff))
        
        logger.debug(f"  Percentage: profile={profile_value:.2f}, other={other_value:.2f}, "
                    f"tolerance={tolerance:.2f}, sim={similarity:.4f}")
        
        return similarity
    
    def _is_outlier(self, profile_value: float, profile_std: float, other_value: float) -> bool:
        """
        Detect if a feature value is an outlier that should be ignored.
        
        Uses statistical outlier detection based on z-score. Features that are beyond
        the outlier threshold are likely due to recording issues, environmental changes,
        or other anomalies rather than natural voice variation.
        
        Args:
            profile_value: Mean value from stored profile
            profile_std: Standard deviation from stored profile
            other_value: Value to test for outlier status
            
        Returns:
            True if the value should be considered an outlier and ignored
        """
        # Can't detect outliers without standard deviation data
        if profile_std <= 0:
            return False
        
        # Calculate z-score
        z_score = abs(other_value - profile_value) / profile_std
        
        # Mark as outlier if beyond threshold
        is_outlier = z_score > self.outlier_threshold
        
        if is_outlier:
            logger.debug(f"    Outlier check: z-score={z_score:.2f} > threshold={self.outlier_threshold:.1f}")
        
        return is_outlier
