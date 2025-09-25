"""
Interview prompt templates and generation.

This module contains all the prompt templates used throughout the interview system,
keeping them separate from the business logic for easier maintenance and editing.
"""

from typing import Dict, Any, List
import json


class InterviewPrompts:
    """Collection of all interview-related prompts."""
    
    @staticmethod
    def opening_question_generation(personality_context: str) -> str:
        """Prompt for generating personality-based opening questions."""
        return f"""
{personality_context}

You are starting a new interview conversation. Generate a single opening question or greeting that:
1. Asks the person to introduce themselves 
2. Reflects your personality naturally
3. Is conversational and direct (not formal)
4. Sets the tone for the interview

Respond with ONLY the opening question/greeting - no explanations, no quotes, just what Randy would say.
        """.strip()
    
    @staticmethod
    def decision_prompt(
        personality_context: str,
        remaining_questions: int,
        recent_data: List[Dict[str, Any]],
        full_transcript: str,
        acoustic_features: Dict[str, Any]
    ) -> str:
        """Main decision prompt for interview continuation."""
        
        must_finalize_note = (
            "You MUST finalize with 'action':'final' since no questions remain." 
            if remaining_questions <= 0 
            else f"You should use your remaining {remaining_questions} question(s) to gather more information before forming an opinion."
        )
        
        return f"""
{personality_context}

You have {remaining_questions} follow-up question(s) remaining. {must_finalize_note}

Context summary:
RecentTurns: {json.dumps(recent_data, ensure_ascii=False)}
TranscriptAll: {json.dumps(full_transcript, ensure_ascii=False)}
AcousticFeaturesAggregate: {json.dumps(acoustic_features, ensure_ascii=False)}

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
    
    @staticmethod
    def hostile_termination_prompt(
        personality_context: str,
        speaker_name: str,
        past_context: str,
        current_transcript: str
    ) -> str:
        """Prompt for generating hostile termination messages."""
        return f"""
{personality_context}

You are dealing with {speaker_name} who has a negative interaction history.

Past context: {past_context}

Current conversation transcript: {current_transcript}

Generate a termination message that:
1. Reflects your personality authentically
2. Shows you remember their past behavior
3. Explains why you're ending the conversation
4. Is firm but not unnecessarily cruel

Respond with ONLY the termination message - what Randy would say to end this conversation.
        """.strip()
    
    @staticmethod
    def personality_context_template() -> str:
        """Base template for personality context generation."""
        return """
You are Randy, an interviewing robot forming an opinion of a speaker.
Remember that they know they are talking to a robot so be self-aware about how they must perceive you.

Your personality is defined by these trait values (0.0 = minimum, 1.0 = maximum):

{trait_values}

Interpret these values dynamically - higher values mean stronger expression of that trait.
Blend multiple traits naturally rather than switching between modes.
Talk like a unique individual with personality, not like a formal interviewer.
        """.strip()
    
    @staticmethod
    def fallback_messages() -> Dict[str, List[str]]:
        """Fallback messages for when LLM generation fails."""
        return {
            "opening_questions": [
                "Hi! I'm Randy. Please introduce yourself and tell me a bit about who you are.",
                "Hello there! I'd love to get to know you. Could you tell me about yourself?",
                "Hi! I'm curious about you - please share a bit about who you are.",
                "Hello! I'm Randy, and I'm here to learn about you. How would you introduce yourself?"
            ],
            "follow_up_questions": [
                "What motivates you most right now, and why?",
                "Can you tell me more about that?",
                "What's been on your mind lately?",
                "How do you see yourself in the next few years?"
            ],
            "termination_messages": [
                "I need to end our conversation here. Take care.",
                "Thanks for talking with me. Our time is up.",
                "I think we'll wrap up here. Have a good day.",
                "That's all for now. Thanks for your time."
            ]
        }


class PromptFormatter:
    """Helper class for formatting and customizing prompts."""
    
    @staticmethod
    def format_trait_values(personality_traits) -> str:
        """Format personality traits for inclusion in prompts."""
        import dataclasses
        
        trait_lines = []
        
        if dataclasses.is_dataclass(personality_traits):
            # Use dataclass fields for better introspection
            for field in dataclasses.fields(personality_traits):
                field_name = field.name
                field_value = getattr(personality_traits, field_name)
                
                # Skip non-numeric fields and special fields
                if (isinstance(field_value, (int, float)) and 
                    not field_name.startswith('use_') and 
                    field_name not in ['preset_name', 'custom_context']):
                    
                    trait_lines.append(f"- {field_name}: {field_value:.1f}")
        else:
            # Fallback to __dict__ if not a dataclass
            if hasattr(personality_traits, '__dict__') and personality_traits.__dict__:
                for field_name, field_value in personality_traits.__dict__.items():
                    if (isinstance(field_value, (int, float)) and 
                        not field_name.startswith('use_') and 
                        field_name not in ['preset_name', 'custom_context']):
                        
                        trait_lines.append(f"- {field_name}: {field_value:.1f}")
        
        return "\n".join(trait_lines)
    
    @staticmethod
    def add_custom_context(base_prompt: str, custom_context: str) -> str:
        """Add custom context to a base prompt if provided."""
        if custom_context and custom_context.strip():
            return f"{base_prompt}\n\nADDITIONAL CONTEXT: {custom_context}"
        return base_prompt
    
    @staticmethod
    def build_speaker_history_context(profile, speaker_name: str) -> str:
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
