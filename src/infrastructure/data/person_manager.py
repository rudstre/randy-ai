"""
Unified person data management.
Coordinates between voice biometrics, conversation records, and AI analysis.
"""
import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np

from .conversations import ConversationRecord, ConversationTurn
from ..audio.voice_id.biometrics import VoiceBiometricProfile

logger = logging.getLogger("person_manager")

class PersonManager:
    """
    Manages complete person data including voice biometrics, conversations, and AI analysis.
    Coordinates between different data domains while keeping them separated.
    """
    
    def __init__(self, profiles_dir: str = "./_voice_profiles", sigma_tolerance: float = 5.0, outlier_threshold: float = 3.0):
        self.profiles_dir = profiles_dir
        self.biometric_profiles: Dict[str, VoiceBiometricProfile] = {}
        self.conversation_records: Dict[str, List[ConversationRecord]] = {}  # person_id -> conversations
        self.ai_assessments: Dict[str, Dict[str, Any]] = {}  # person_id -> assessments
        
        # Create similarity calculator
        from ..audio.voice_id.similarity import SimilarityCalculator
        self.similarity_calculator = SimilarityCalculator(
            base_sigma_scale=sigma_tolerance / 2.5,
            min_features_required=max(8, int(outlier_threshold * 3))
        )
        
        # Create directory
        os.makedirs(self.profiles_dir, exist_ok=True)
        
        # Load existing profiles
        self.load_all_profiles()
    
    @property
    def profiles(self) -> Dict[str, Any]:
        """Compatibility property for ProgressiveVoiceIdentifier."""
        # Convert biometric profiles to compatible objects
        compatible_profiles = {}
        for person_id, biometric_profile in self.biometric_profiles.items():
            # Create a compatibility wrapper
            class CompatibilityProfile:
                def __init__(self, biometric_profile, similarity_calc):
                    self._biometric = biometric_profile
                    self._similarity_calc = similarity_calc
                
                def calculate_similarity(self, features, similarity_calculator=None):
                    calc = similarity_calculator or self._similarity_calc
                    result = calc.calculate_similarity(
                        self._biometric.features_mean,
                        self._biometric.features_std,
                        features
                    )
                    return result.similarity
                
                def get_data_confidence(self, scale_factor=None):
                    """Delegate to the underlying biometric profile."""
                    return self._biometric.get_data_confidence(scale_factor)
            
            compatible_profiles[person_id] = CompatibilityProfile(biometric_profile, self.similarity_calculator)
        
        return compatible_profiles
    
    def _get_profile_path(self, person_id: str) -> str:
        """Get file path for a person's complete profile."""
        return os.path.join(self.profiles_dir, f"{person_id}.json")
    
    def load_all_profiles(self) -> None:
        """Load all person profiles from disk."""
        if not os.path.exists(self.profiles_dir):
            return
        
        for filename in os.listdir(self.profiles_dir):
            if filename.endswith('.json'):
                person_id = filename[:-5]  # Remove .json extension
                self.load_person_profile(person_id)
        
        logger.info(f"Loaded {len(self.biometric_profiles)} person profiles")
    
    def load_person_profile(self, person_id: str) -> bool:
        """Load a complete person profile from disk."""
        profile_path = self._get_profile_path(person_id)
        
        if not os.path.exists(profile_path):
            return False
        
        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load biometric profile
            biometric_data = self._extract_biometric_data(data)
            self.biometric_profiles[person_id] = VoiceBiometricProfile.from_dict(biometric_data)
            
            # Load conversation records if available
            if 'conversations' in data and data['conversations']:
                conversations = []
                for conv_data in data['conversations']:
                    # Convert turns
                    turns = []
                    if 'turns' in conv_data and conv_data['turns']:
                        for turn_data in conv_data['turns']:
                            turns.append(ConversationTurn(**turn_data))
                    conv_data['turns'] = turns
                    conversations.append(ConversationRecord(**conv_data))
                self.conversation_records[person_id] = conversations
            
            # Load AI assessments if available
            ai_data = self._extract_ai_assessment_data(data)
            if ai_data:
                self.ai_assessments[person_id] = ai_data
            
            logger.info(f"Loaded complete profile for person {person_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load profile for {person_id}: {e}")
            return False
    
    def save_person_profile(self, person_id: str) -> None:
        """Save a complete person profile to disk."""
        if person_id not in self.biometric_profiles:
            logger.warning(f"Cannot save unknown person: {person_id}")
            return
        
        profile_path = self._get_profile_path(person_id)
        
        try:
            # Combine all data domains into unified JSON
            data = {}
            
            # Add biometric data
            biometric_profile = self.biometric_profiles[person_id]
            data.update(biometric_profile.to_dict())
            
            # Add conversation records
            if person_id in self.conversation_records:
                conversations_data = []
                for conv in self.conversation_records[person_id]:
                    conv_dict = {
                        'conversation_id': conv.conversation_id,
                        'date_time': conv.date_time,
                        'duration_minutes': conv.duration_minutes,
                        'initial_question': conv.initial_question,
                        'turns': [
                            {
                                'turn_idx': turn.turn_idx,
                                'question': turn.question,
                                'transcript': turn.transcript,
                                'timestamp': turn.timestamp,
                                'voice_features': turn.voice_features,
                                'duration_seconds': turn.duration_seconds
                            }
                            for turn in conv.turns
                        ],
                        'final_opinion_word': conv.final_opinion_word,
                        'final_score_overall': conv.final_score_overall,
                        'final_score_text_only': conv.final_score_text_only,
                        'ai_rationale': conv.ai_rationale,
                        'termination_message': conv.termination_message,
                        'termination_reason': conv.termination_reason,
                        'ai_personality_traits': conv.ai_personality_traits,
                        'identification_confidence': conv.identification_confidence,
                        'was_welcomed_back': conv.was_welcomed_back,
                        'aggregated_voice_features': conv.aggregated_voice_features
                    }
                    conversations_data.append(conv_dict)
                data['conversations'] = conversations_data
            
            # Add AI assessments
            if person_id in self.ai_assessments:
                data.update(self.ai_assessments[person_id])
            
            
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved complete profile for person {person_id}")
            
        except Exception as e:
            logger.error(f"Failed to save profile for {person_id}: {e}")
    
    def identify_person(self, features: Dict[str, float], 
                       similarity_threshold: float = 0.7) -> Tuple[Optional[str], float]:
        """
        Identify person based on voice features.
        
        Args:
            features: Voice features to match
            similarity_threshold: Minimum similarity score to consider a match
            
        Returns:
            Tuple of (person_id, confidence_score) or (None, 0.0) if no match
        """
        if not self.biometric_profiles:
            return None, 0.0
        
        best_match = None
        best_score = 0.0
        
        for person_id, profile in self.biometric_profiles.items():
            similarity = profile.calculate_similarity(features, self.similarity_calculator)
            
            if similarity > best_score:
                best_score = similarity
                best_match = person_id
        
        if best_score >= similarity_threshold:
            logger.info(f"Identified person {best_match} with confidence {best_score:.3f}")
            return best_match, best_score
        else:
            logger.info(f"No person match found (best score: {best_score:.3f})")
            return None, best_score
    
    def create_new_person_profile(self, features: Dict[str, float], 
                                 conversation_record: Optional[ConversationRecord] = None,
                                 person_name: Optional[str] = None) -> str:
        """
        Create a new person profile with initial voice features and optional conversation.
        
        Args:
            features: Initial voice features
            conversation_record: Optional conversation record
            person_name: Optional human-readable name
            
        Returns:
            person_id: Generated unique identifier
        """
        # Generate unique person ID
        timestamp = str(int(time.time()))
        person_id = f"person_{timestamp}"
        
        # Ensure uniqueness
        counter = 1
        while person_id in self.biometric_profiles:
            person_id = f"person_{timestamp}_{counter}"
            counter += 1
        
        # Generate a human-readable name if none provided
        if not person_name:
            person_count = len(self.biometric_profiles) + 1
            person_name = f"Person #{person_count}"
        
        # Create biometric profile
        biometric_profile = VoiceBiometricProfile(
            person_id=person_id,
            person_name=person_name,
            conversation_count=0
        )
        
        # Add initial voice features
        if conversation_record:
            individual_samples = [turn.voice_features for turn in conversation_record.turns]
            biometric_profile.update_features(
                features, 
                confidence=conversation_record.identification_confidence,
                sample_count=len(conversation_record.turns),
                individual_samples=individual_samples
            )
            biometric_profile.conversation_count = 1
            
            # Store conversation record
            self.conversation_records[person_id] = [conversation_record]
        else:
            biometric_profile.update_features(features, confidence=1.0, sample_count=1)
        
        # Store biometric profile
        self.biometric_profiles[person_id] = biometric_profile
        
        # Save to disk
        self.save_person_profile(person_id)
        
        logger.info(f"Created new person profile: {person_id}")
        return person_id
    
    def update_person_profile(self, person_id: str, features: Dict[str, float],
                             conversation_record: Optional[ConversationRecord] = None,
                             person_name: Optional[str] = None) -> None:
        """Update an existing person profile with new data."""
        if person_id not in self.biometric_profiles:
            logger.warning(f"Cannot update unknown person: {person_id}")
            return
        
        biometric_profile = self.biometric_profiles[person_id]
        
        # Update name only if current name is a placeholder (keep first real name)
        if person_name and person_name.strip():
            current_name = biometric_profile.person_name or ""
            is_placeholder = (not current_name or 
                            current_name.startswith("Person #") or 
                            current_name == "Unknown")
            
            if is_placeholder:
                logger.info(f"Setting person name: {current_name} -> {person_name}")
                biometric_profile.person_name = person_name.strip()
                # Add to extracted names history
                if person_name not in biometric_profile.extracted_names:
                    biometric_profile.extracted_names.append(person_name.strip())
            else:
                # Keep the original name, but still track this name was mentioned
                logger.info(f"Keeping original name '{current_name}', but noting they also said '{person_name}'")
                if person_name not in biometric_profile.extracted_names:
                    biometric_profile.extracted_names.append(person_name.strip())
        
        if conversation_record:
            # Update biometric profile with conversation data
            individual_samples = [turn.voice_features for turn in conversation_record.turns]
            biometric_profile.update_features(
                features,
                confidence=conversation_record.identification_confidence,
                sample_count=len(conversation_record.turns),
                individual_samples=individual_samples
            )
            biometric_profile.conversation_count += 1
            
            # Add conversation record
            if person_id not in self.conversation_records:
                self.conversation_records[person_id] = []
            self.conversation_records[person_id].append(conversation_record)
        else:
            # Just update biometric features
            biometric_profile.update_features(features, confidence=1.0, sample_count=1)
        
        # Save updated profile
        self.save_person_profile(person_id)
        logger.info(f"Updated person profile for {person_id}")
    
    def get_person_info(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive information about a person."""
        if person_id not in self.biometric_profiles:
            return None
        
        biometric_profile = self.biometric_profiles[person_id]
        conversations = self.conversation_records.get(person_id, [])
        ai_assessment = self.ai_assessments.get(person_id, {})
        
        # Basic info
        info = {
            "person_id": person_id,
            "name": biometric_profile.person_name or "Unknown",
            "extracted_names": biometric_profile.extracted_names,
            "conversations": len(conversations),
            "voice_samples": biometric_profile.sample_count,
            "first_recorded": biometric_profile.first_recorded,
            "last_recorded": biometric_profile.last_recorded,
            "data_confidence": biometric_profile.get_data_confidence(),
        }
        
        # Conversation analysis
        if conversations:
            recent_convs = conversations[-3:]
            info.update({
                "recent_opinions": [c.final_opinion_word for c in recent_convs],
                "recent_scores": [c.final_score_overall for c in recent_convs],
                "avg_recent_score": np.mean([c.final_score_overall for c in recent_convs]),
                "early_terminations": sum(1 for c in recent_convs if c.termination_reason),
                "sample_quotes": [
                    f"'{turn.transcript[:100]}...' ({conv.date_time[:10]})"
                    for conv in recent_convs for turn in conv.turns[:1]
                    if turn.transcript and len(turn.transcript) > 10
                ][:3]
            })
        
        # AI assessment data
        info.update(ai_assessment)
        
        return info
    
    def list_all_persons(self) -> List[Dict[str, Any]]:
        """Get information about all known persons."""
        return [info for person_id in self.biometric_profiles.keys() 
                if (info := self.get_person_info(person_id)) is not None]
    
    def aggregate_conversation_features(self, turns: List) -> Dict[str, float]:
        """Aggregate voice features across all turns in a conversation."""
        if not turns:
            return {}
        
        # Extract all valid feature dictionaries
        valid_features = [turn.voice_features for turn in turns if hasattr(turn, 'voice_features') and turn.voice_features]
        if not valid_features:
            return {}
        
        # Get all feature keys from first valid feature set
        feature_keys = valid_features[0].keys()
        aggregated = {}
        
        # Calculate mean for each feature
        for key in feature_keys:
            values = [features.get(key, 0.0) for features in valid_features if key in features]
            if values:
                aggregated[key] = sum(values) / len(values)
            else:
                aggregated[key] = 0.0
        
        return aggregated
    
    def _extract_biometric_data(self, full_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract biometric-specific data from full profile."""
        biometric_fields = [
            'person_id', 'person_name', 'extracted_names',
            'features_mean', 'features_std', 'features_min', 'features_max',
            'feature_history', 'all_turn_features', 'sample_count', 'conversation_count',
            'first_recorded', 'last_recorded', 'confidence_scores'
        ]
        
        biometric_data = {}
        for field in biometric_fields:
            if field in full_data:
                biometric_data[field] = full_data[field]
        
        return biometric_data
    
    def _extract_ai_assessment_data(self, full_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract AI assessment data from full profile."""
        ai_fields = [
            'personality_assessment', 'behavioral_patterns', 
            'hostility_indicators', 'positive_indicators'
        ]
        
        ai_data = {}
        for field in ai_fields:
            if field in full_data:
                ai_data[field] = full_data[field]
        
        return ai_data
    
    
    def get_speaker_info(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Get speaker information for ProgressiveVoiceIdentifier."""
        if person_id not in self.biometric_profiles:
            return None
        
        biometric = self.biometric_profiles[person_id]
        conversations = self.conversation_records.get(person_id, [])
        
        # Build speaker info dictionary
        info = {
            'name': biometric.person_name or 'Unknown',
            'conversation_count': len(conversations),
            'total_turns': biometric.sample_count,
            'first_seen': biometric.first_recorded,
            'last_seen': biometric.last_recorded
        }
        
        # Add conversation history if available
        if conversations:
            recent_conversations = conversations[-3:]  # Last 3 conversations
            info['recent_scores'] = [c.final_score_overall for c in recent_conversations]
            info['recent_opinions'] = [c.final_opinion_word for c in recent_conversations]
            info['avg_recent_score'] = sum(info['recent_scores']) / len(info['recent_scores'])
        
        return info
