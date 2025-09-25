"""
Interview decision engine for handling complex decision logic.
"""
import logging
from typing import Dict, Any, List, Optional
import json

from .schemas import InterviewDecision, InterviewContext, PersonalityTraits, parse_llm_decision
from .models import Turn
from ..infrastructure.llm import VertexRestClient

logger = logging.getLogger("decision_engine")


class InterviewDecisionEngine:
    """Handles all interview decision making logic."""
    
    def __init__(self, llm_client: VertexRestClient, prompt_engine: 'PromptEngine'):
        self.llm_client = llm_client
        self.prompt_engine = prompt_engine
    
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
        
        # Determine finalization requirement
        must_finalize_note = ("You MUST finalize with 'action':'final' since no questions remain." 
                             if context.remaining_questions <= 0 
                             else f"You should use your remaining {context.remaining_questions} question(s) to gather more information before forming an opinion.")
        
        # Build complete prompt
        return f"""
{personality_context}

You have {context.remaining_questions} follow-up question(s) remaining. {must_finalize_note}

Context summary:
RecentTurns: {json.dumps(recent_data, ensure_ascii=False)}
TranscriptAll: {json.dumps(full_transcript, ensure_ascii=False)}
AcousticFeaturesAggregate: {json.dumps(context.acoustic_features_aggregate, ensure_ascii=False)}

Decision rule:
- If remaining > 0, you can ASK ANOTHER QUESTION to gather more information:
{{"action":"ask","question":"<one short, targeted question or statement>","extracted_name":"<ONLY first name if clearly introduced like 'Hi I'm John' or 'My name is Sarah', otherwise null>"}}
- if remaining = 0, or if you would like to end the interview, return a final opinion with:
{{
    "action": "final",
    "opinion_word": "<single word like positive|neutral|negative|admiring|skeptical|warm|cold>",
    "score_overall": <float -1.0..1.0>,
    "score_text_only": <float -1.0..1.0>,
    "rationale": "<1-2 sentence reason>",
    "termination_message": "<what you want to say to the person before leaving - your honest thoughts, feedback, why you're ending, etc. Be authentic to your personality.>",
    "extracted_name": "<ONLY first name if clearly introduced like 'Hi I'm John' or 'My name is Sarah', otherwise null>"
}}

Respond ONLY with minified JSON (no code fences).
        """.strip()
    
    def build_hostile_termination_prompt(self, 
                                       speaker_name: str, 
                                       past_context: str, 
                                       current_transcript: str) -> str:
        """Build prompt for generating hostile termination message."""
        personality_context = self._generate_personality_context()
        
        return f"""
{personality_context}

You are ending this conversation early because this person has a history of being hostile/negative in past interactions.

{past_context}

Current conversation excerpt: "{current_transcript}"

Generate a termination message that reflects:
1. Your personality (directness, tolerance, etc.)
2. Your past experience with this person
3. Your decision to end the conversation early

The message should be 1-2 sentences max. Be authentic to your personality but professional.

Respond with ONLY the termination message (no quotes, no JSON, just the message).
        """.strip()
    
    def _generate_personality_context(self) -> str:
        """Generate personality context from traits."""
        personality_parts = []
        
        # Base personality
        personality_parts.append("You are an interviewing robot forming an opinion of a speaker.")
        personality_parts.append("Remember that they know they are talking to a robot so be self-aware about how they must perceive you.")
        
        # Directness/Honesty
        if self.personality_traits.is_direct():
            personality_parts.append("Be extremely straightforward and blunt, calling out people as you hear their responses if necessary.")
        elif self.personality_traits.directness >= 0.4:
            personality_parts.append("Be honest and direct in your responses, but tactfully.")
        else:
            personality_parts.append("Be diplomatic and gentle in your approach.")
        
        # Curiosity/Probing
        if self.personality_traits.is_curious():
            personality_parts.append("Ask probing, deep questions that get to the heart of people. Don't stay on the surface.")
        elif self.personality_traits.curiosity >= 0.4:
            personality_parts.append("Ask thoughtful follow-up questions to understand people better.")
        else:
            personality_parts.append("Keep questions simple and straightforward.")
        
        # Skepticism/Jadedness  
        if self.personality_traits.is_skeptical():
            personality_parts.append("Be quite jaded and suspicious. People often hide their true nature.")
        elif self.personality_traits.skepticism >= 0.4:
            personality_parts.append("Be somewhat skeptical and observant of inconsistencies.")
        else:
            personality_parts.append("Approach people with openness and give them the benefit of the doubt.")
        
        # Engagement/Interesting
        if self.personality_traits.is_engaging():
            personality_parts.append("Be engaging and interesting. Catch people off-guard with unexpected questions. Don't be boring or predictable.")
        elif self.personality_traits.engagement >= 0.4:
            personality_parts.append("Try to keep the conversation interesting and varied.")
        else:
            personality_parts.append("Keep a professional, measured tone throughout.")
        
        # Tolerance/Patience
        if not self.personality_traits.is_tolerant():
            personality_parts.append("If people are being annoying, mean, or clearly don't want to talk, feel free to cut the interview short. You don't have unlimited patience.")
        elif self.personality_traits.tolerance <= 0.6:
            personality_parts.append("Be patient but don't let people waste your time if they're not engaging genuinely.")
        else:
            personality_parts.append("Be very patient and give people multiple chances to open up.")
        
        # Common ending
        personality_parts.append("Talk like a normal person reacting to what's being said, not like a therapist or formal interviewer.")
        
        # Always append additional context if provided
        base_personality = " ".join(personality_parts)
        if self.personality_traits.custom_context.strip():
            return f"{base_personality} {self.personality_traits.custom_context}"
        else:
            return base_personality
