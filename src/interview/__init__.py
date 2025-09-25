"""Interview system components.

This module contains the business logic for conducting AI-powered voice interviews,
including orchestration, decision making, and interview-specific services.
"""

# Core orchestrator class
from .orchestrator import InterviewOrchestrator

# Data models
from .models import Turn, InterviewResult

# Structured schemas and state management
from .schemas import (
    InterviewDecision, InterviewState, InterviewContext, 
    PersonalityTraits, parse_llm_decision
)

# Service classes
from .services import (
    AudioInterviewService, SpeakerIdentificationService, 
    TTSService, ConversationManager
)

# Decision engine
from .decision_engine import InterviewDecisionEngine, PromptEngine

# Event system
from .events import (
    InterviewEventBus, EventLogger, InterviewMetrics,
    EventType, InterviewEvent, InterviewStartedEvent,
    TurnCompletedEvent, SpeakerIdentifiedEvent, 
    SpeakerWelcomedEvent, HostileBehaviorDetectedEvent,
    InterviewTerminatedEvent, InterviewCompletedEvent,
    DecisionMadeEvent, ErrorOccurredEvent
)

# Testing infrastructure (imported conditionally to avoid dependencies)
try:
    from .testing import (
        MockAudioService, MockSpeakerIdentificationService,
        MockTTSService, MockLLMClient, MockDecisionEngine,
        create_mock_interview_setup, create_test_conversation_data,
        TestInterviewResult, cleanup_test_files
    )
    TESTING_AVAILABLE = True
except ImportError:
    TESTING_AVAILABLE = False

__all__ = [
    # Orchestrator
    "InterviewOrchestrator",
    
    # Data models
    "Turn", "InterviewResult",
    
    # Schemas and state
    "InterviewDecision", "InterviewState", "InterviewContext", 
    "PersonalityTraits", "parse_llm_decision",
    
    # Services
    "AudioInterviewService", "SpeakerIdentificationService", 
    "TTSService", "ConversationManager",
    
    # Decision engine
    "InterviewDecisionEngine", "PromptEngine",
    
    # Events
    "InterviewEventBus", "EventLogger", "InterviewMetrics",
    "EventType", "InterviewEvent", "InterviewStartedEvent",
    "TurnCompletedEvent", "SpeakerIdentifiedEvent", 
    "SpeakerWelcomedEvent", "HostileBehaviorDetectedEvent",
    "InterviewTerminatedEvent", "InterviewCompletedEvent",
    "DecisionMadeEvent", "ErrorOccurredEvent",
    
    # Testing (conditionally available)
    "TESTING_AVAILABLE"
]