"""
Interview decision engine for handling complex decision logic.
"""
import logging
from typing import Dict, Any, List, Optional
import json

from .schemas import InterviewDecision, InterviewContext, PersonalityTraits, parse_llm_decision
from .models import Turn
from .prompts import InterviewPrompts, PromptFormatter
from ..infrastructure.llm import VertexRestClient

logger = logging.getLogger("decision_engine")


class InterviewDecisionEngine:
    """Handles all interview decision making logic."""
    
    def __init__(self, llm_client: VertexRestClient, prompt_engine: 'PromptEngine'):
        self.llm_client = llm_client
        self.prompt_engine = prompt_engine
        # Give the prompt engine access to the LLM client
        self.prompt_engine._llm_client = llm_client
    
    def decide_next_action(self, context: InterviewContext) -> InterviewDecision:
        """
        Decide whether to ask another question or finalize the interview.
        
        Args:
            context: Current interview context
            
        Returns:
            InterviewDecision with validated structure
        """
        prompt = self.prompt_engine.build_decision_prompt(context)
        
        try:
            raw_response = self.llm_client.generate_content(prompt, temperature=0.0)
            logger.info(f"Raw LLM response: {raw_response}")
            
            decision = parse_llm_decision(raw_response)
            
            # Validate decision against context
            if decision.action == "ask" and context.remaining_questions <= 0:
                logger.warning("LLM wanted to ask but no questions remaining, forcing final")
                decision = self._create_fallback_final_decision()
            elif decision.action == "ask" and not decision.question:
                logger.warning("LLM returned 'ask' but no question provided")
                decision = self._create_fallback_question_decision(context)
            
            return decision
            
        except Exception as e:
            logger.error("LLM decision failed: %s", e)
            return self._create_fallback_decision(context)
    
    def generate_hostile_termination_message(self, 
                                           speaker_id: str, 
                                           current_turns: List[Turn],
                                           voice_profile_manager) -> str:
        """
        Generate a custom termination message for a hostile speaker.
        
        Args:
            speaker_id: ID of the hostile speaker
            current_turns: Current conversation turns
            voice_profile_manager: Voice profile manager for history
            
        Returns:
            Termination message string
        """
        try:
            # Build context about the speaker
            speaker_info = voice_profile_manager.get_speaker_info(speaker_id) if voice_profile_manager else None
            speaker_name = speaker_info.get("name", "Unknown") if speaker_info else "Unknown"
            
            profile = voice_profile_manager.profiles.get(speaker_id) if voice_profile_manager else None
            
            # Prepare termination context
            past_context = self._build_speaker_history_context(profile, speaker_name)
            current_transcript = " ".join([turn.transcript for turn in current_turns[-2:] if turn.transcript])
            
            # Generate termination message
            prompt = self.prompt_engine.build_hostile_termination_prompt(
                speaker_name, past_context, current_transcript
            )
            
            response = self.llm_client.generate_content(prompt, temperature=0.2)
            return response.strip()
            
        except Exception as e:
            logger.error("Failed to generate hostile termination message: %s", e)
            return self._get_default_hostile_termination()
    
    def _build_speaker_history_context(self, profile, speaker_name: str) -> str:
        """Build context string about speaker's past interactions."""
        if not profile:
            return f"No previous interaction history with {speaker_name}."
        
        recent_opinions = profile.opinion_history[-3:] if profile.opinion_history else []
        recent_scores = profile.score_history[-3:] if profile.score_history else []
        conversation_count = profile.conversation_count
        last_summary = profile.conversation_summaries[-1] if profile.conversation_summaries else 'No summary'
        
        return f"""
Past interactions with {speaker_name}:
- Total conversations: {conversation_count}
- Recent opinions: {recent_opinions}
- Recent scores: {recent_scores}
- Last interaction summary: {last_summary}
        """.strip()
    
    def _create_fallback_decision(self, context: InterviewContext) -> InterviewDecision:
        """Create appropriate fallback decision based on context."""
        if context.remaining_questions > 0:
            return self._create_fallback_question_decision(context)
        else:
            return self._create_fallback_final_decision()
    
    def _create_fallback_question_decision(self, context: InterviewContext) -> InterviewDecision:
        """Create fallback question when LLM fails."""
        fallback_questions = [
            "What motivates you most right now, and why?",
            "Can you tell me more about that?",
            "What's been on your mind lately?",
            "How do you see yourself in the next few years?"
        ]
        
        question_idx = min(context.turn_count, len(fallback_questions) - 1)
        question = fallback_questions[question_idx]
        
        return InterviewDecision(
            action="ask",
            question=question
        )
    
    def _create_fallback_final_decision(self) -> InterviewDecision:
        """Create neutral fallback final decision."""
        return InterviewDecision(
            action="final",
            opinion_word="neutral",
            score_overall=0.0,
            score_text_only=0.0,
            rationale="LLM fallback - neutral opinion.",
            termination_message="Thanks for the conversation."
        )
    
    def _get_default_hostile_termination(self) -> str:
        """Get default hostile termination message."""
        return "I need to end our conversation here. Take care."


class PromptEngine:
    """Handles prompt generation and templating."""
    
    def __init__(self, personality_traits: PersonalityTraits):
        self.personality_traits = personality_traits
        self._llm_client: Optional['VertexRestClient'] = None
    
    def generate_opening_question(self) -> str:
        """Generate a personality-based opening question using LLM."""
        import logging
        import random
        
        logger = logging.getLogger("prompt_engine")
        
        try:
            if hasattr(self, '_llm_client') and self._llm_client:
                # Generate personality context
                personality_context = self._generate_personality_context()
                
                # Get prompt from prompts module
                prompt = InterviewPrompts.opening_question_generation(personality_context)
                
                response = self._llm_client.generate_content(prompt, temperature=0.7)
                # Clean up the response
                opening_question = response.strip().strip('"').strip("'")
                logger.info(f"Generated opening question: {opening_question}")
                return opening_question
            else:
                raise ValueError("No LLM client available")
            
        except Exception as e:
            logger.warning(f"Failed to generate opening question: {e}, using fallback")
            # Use fallback from prompts module
            fallbacks = InterviewPrompts.fallback_messages()["opening_questions"]
            return random.choice(fallbacks)
    
    def build_decision_prompt(self, context: InterviewContext) -> str:
        """
        Build the main decision prompt for interview continuation.
        
        Args:
            context: Current interview context
            
        Returns:
            Formatted prompt string
        """
        # Prepare context data
        recent_turns = context.get_recent_turns(2)
        recent_data = [
            {
                "question": turn.question,
                "transcript": turn.transcript,
                "features_sample": {k: v for k, v in turn.features.items() if k in ["pitch_mean", "mfcc_1", "loudness"]}
            }
            for turn in recent_turns
        ]
        
        full_transcript = context.get_full_transcript()
        
        # Generate personality context
        personality_context = self._generate_personality_context()
        
        # Use prompt from prompts module
        return InterviewPrompts.decision_prompt(
            personality_context=personality_context,
            remaining_questions=context.remaining_questions,
            recent_data=recent_data,
            full_transcript=full_transcript,
            acoustic_features=context.acoustic_features_aggregate
        )
    
    def build_hostile_termination_prompt(self, 
                                       speaker_name: str, 
                                       past_context: str, 
                                       current_transcript: str) -> str:
        """Build prompt for generating hostile termination message."""
        personality_context = self._generate_personality_context()
        
        return InterviewPrompts.hostile_termination_prompt(
            personality_context=personality_context,
            speaker_name=speaker_name,
            past_context=past_context,
            current_transcript=current_transcript
        )
    
    def _generate_personality_context(self) -> str:
        """Generate personality context by dynamically embedding all trait values."""
        p = self.personality_traits
        
        # Get formatted trait values
        trait_values = PromptFormatter.format_trait_values(p)
        
        # Get base template and format it
        base_template = InterviewPrompts.personality_context_template()
        base_context = base_template.format(trait_values=trait_values)
        
        # Add custom context if provided
        return PromptFormatter.add_custom_context(base_context, p.custom_context)
