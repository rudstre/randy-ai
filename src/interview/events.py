"""
Event-driven architecture for the interview system.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable, Type, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("events")


class EventType(str, Enum):
    """Types of interview events."""
    INTERVIEW_STARTED = "interview_started"
    TURN_COMPLETED = "turn_completed"
    SPEAKER_IDENTIFIED = "speaker_identified"
    SPEAKER_WELCOMED = "speaker_welcomed"
    HOSTILE_BEHAVIOR_DETECTED = "hostile_behavior_detected"
    INTERVIEW_TERMINATED = "interview_terminated"
    INTERVIEW_COMPLETED = "interview_completed"
    DECISION_MADE = "decision_made"
    ERROR_OCCURRED = "error_occurred"


@dataclass
class InterviewEvent(ABC):
    """Base class for all interview events."""
    event_type: EventType
    conversation_id: str
    timestamp: float
    data: Dict[str, Any]


@dataclass
class InterviewStartedEvent(InterviewEvent):
    """Event fired when interview begins."""
    def __init__(self, conversation_id: str, timestamp: float, max_questions: int):
        super().__init__(
            event_type=EventType.INTERVIEW_STARTED,
            conversation_id=conversation_id,
            timestamp=timestamp,
            data={"max_questions": max_questions}
        )


@dataclass
class TurnCompletedEvent(InterviewEvent):
    """Event fired when a conversation turn is completed."""
    def __init__(self, conversation_id: str, timestamp: float, turn_idx: int, 
                 transcript: str, features: Dict[str, float]):
        super().__init__(
            event_type=EventType.TURN_COMPLETED,
            conversation_id=conversation_id,
            timestamp=timestamp,
            data={
                "turn_idx": turn_idx,
                "transcript": transcript,
                "features": features
            }
        )


@dataclass
class SpeakerIdentifiedEvent(InterviewEvent):
    """Event fired when a speaker is identified."""
    def __init__(self, conversation_id: str, timestamp: float, speaker_id: str, 
                 confidence: float, is_returning: bool):
        super().__init__(
            event_type=EventType.SPEAKER_IDENTIFIED,
            conversation_id=conversation_id,
            timestamp=timestamp,
            data={
                "person_id": speaker_id,
                "confidence": confidence,
                "is_returning": is_returning
            }
        )


@dataclass 
class SpeakerWelcomedEvent(InterviewEvent):
    """Event fired when a returning speaker is welcomed."""
    def __init__(self, conversation_id: str, timestamp: float, speaker_id: str, speaker_name: str):
        super().__init__(
            event_type=EventType.SPEAKER_WELCOMED,
            conversation_id=conversation_id,
            timestamp=timestamp,
            data={
                "person_id": speaker_id,
                "person_name": speaker_name
            }
        )


@dataclass
class HostileBehaviorDetectedEvent(InterviewEvent):
    """Event fired when hostile behavior is detected."""
    def __init__(self, conversation_id: str, timestamp: float, speaker_id: str, 
                 reasoning: str, disposition: str):
        super().__init__(
            event_type=EventType.HOSTILE_BEHAVIOR_DETECTED,
            conversation_id=conversation_id,
            timestamp=timestamp,
            data={
                "person_id": speaker_id,
                "reasoning": reasoning,
                "disposition": disposition
            }
        )


@dataclass
class InterviewTerminatedEvent(InterviewEvent):
    """Event fired when interview is terminated early."""
    def __init__(self, conversation_id: str, timestamp: float, reason: str, 
                 turn_count: int, termination_message: str):
        super().__init__(
            event_type=EventType.INTERVIEW_TERMINATED,
            conversation_id=conversation_id,
            timestamp=timestamp,
            data={
                "reason": reason,
                "turn_count": turn_count,
                "termination_message": termination_message
            }
        )


@dataclass
class InterviewCompletedEvent(InterviewEvent):
    """Event fired when interview completes successfully."""
    def __init__(self, conversation_id: str, timestamp: float, turn_count: int,
                 final_opinion: str, final_score: float, speaker_id: Optional[str]):
        super().__init__(
            event_type=EventType.INTERVIEW_COMPLETED,
            conversation_id=conversation_id,
            timestamp=timestamp,
            data={
                "turn_count": turn_count,
                "final_opinion": final_opinion,
                "final_score": final_score,
                "speaker_id": speaker_id
            }
        )


@dataclass
class DecisionMadeEvent(InterviewEvent):
    """Event fired when LLM makes a decision."""
    def __init__(self, conversation_id: str, timestamp: float, action: str, 
                 question: Optional[str], remaining_questions: int):
        super().__init__(
            event_type=EventType.DECISION_MADE,
            conversation_id=conversation_id,
            timestamp=timestamp,
            data={
                "action": action,
                "question": question,
                "remaining_questions": remaining_questions
            }
        )


@dataclass
class ErrorOccurredEvent(InterviewEvent):
    """Event fired when an error occurs."""
    def __init__(self, conversation_id: str, timestamp: float, error_type: str, 
                 error_message: str, component: str):
        super().__init__(
            event_type=EventType.ERROR_OCCURRED,
            conversation_id=conversation_id,
            timestamp=timestamp,
            data={
                "error_type": error_type,
                "error_message": error_message,
                "component": component
            }
        )


EventHandler = Callable[[InterviewEvent], None]


class InterviewEventBus:
    """Event bus for interview system communication."""
    
    def __init__(self):
        self._handlers: Dict[EventType, List[EventHandler]] = {}
        self._global_handlers: List[EventHandler] = []
    
    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """
        Subscribe to specific event type.
        
        Args:
            event_type: Type of event to listen for
            handler: Function to call when event occurs
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug(f"Subscribed handler to {event_type}")
    
    def subscribe_all(self, handler: EventHandler) -> None:
        """
        Subscribe to all events.
        
        Args:
            handler: Function to call for any event
        """
        self._global_handlers.append(handler)
        logger.debug("Subscribed global handler")
    
    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """
        Unsubscribe from specific event type.
        
        Args:
            event_type: Type of event to stop listening for
            handler: Handler function to remove
        """
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
                logger.debug(f"Unsubscribed handler from {event_type}")
            except ValueError:
                logger.warning(f"Handler not found for {event_type}")
    
    def emit(self, event: InterviewEvent) -> None:
        """
        Emit an event to all subscribers.
        
        Args:
            event: Event to emit
        """
        logger.debug(f"Emitting event: {event.event_type} for conversation {event.conversation_id}")
        
        # Call specific handlers
        if event.event_type in self._handlers:
            for handler in self._handlers[event.event_type]:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Error in event handler for {event.event_type}: {e}")
        
        # Call global handlers
        for handler in self._global_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error in global event handler: {e}")
    
    def clear_handlers(self) -> None:
        """Clear all event handlers."""
        self._handlers.clear()
        self._global_handlers.clear()
        logger.debug("Cleared all event handlers")


class EventLogger:
    """Logs all events for debugging and analysis."""
    
    def __init__(self, log_level: int = logging.INFO):
        self.logger = logging.getLogger("event_logger")
        self.logger.setLevel(log_level)
    
    def handle_event(self, event: InterviewEvent) -> None:
        """Log event details."""
        self.logger.info(f"Event: {event.event_type} | Conversation: {event.conversation_id} | Data: {event.data}")


class InterviewMetrics:
    """Collects metrics from interview events."""
    
    def __init__(self):
        self.interviews_started = 0
        self.interviews_completed = 0
        self.interviews_terminated = 0
        self.total_turns = 0
        self.speakers_identified = 0
        self.hostile_detections = 0
        self.errors_occurred = 0
    
    def handle_event(self, event: InterviewEvent) -> None:
        """Update metrics based on event."""
        if event.event_type == EventType.INTERVIEW_STARTED:
            self.interviews_started += 1
        elif event.event_type == EventType.INTERVIEW_COMPLETED:
            self.interviews_completed += 1
        elif event.event_type == EventType.INTERVIEW_TERMINATED:
            self.interviews_terminated += 1
        elif event.event_type == EventType.TURN_COMPLETED:
            self.total_turns += 1
        elif event.event_type == EventType.SPEAKER_IDENTIFIED:
            self.speakers_identified += 1
        elif event.event_type == EventType.HOSTILE_BEHAVIOR_DETECTED:
            self.hostile_detections += 1
        elif event.event_type == EventType.ERROR_OCCURRED:
            self.errors_occurred += 1
    
    def get_metrics(self) -> Dict[str, int]:
        """Get current metrics snapshot."""
        return {
            "interviews_started": self.interviews_started,
            "interviews_completed": self.interviews_completed,
            "interviews_terminated": self.interviews_terminated,
            "total_turns": self.total_turns,
            "speakers_identified": self.speakers_identified,
            "hostile_detections": self.hostile_detections,
            "errors_occurred": self.errors_occurred
        }
    
    def reset(self) -> None:
        """Reset all metrics to zero."""
        self.interviews_started = 0
        self.interviews_completed = 0
        self.interviews_terminated = 0
        self.total_turns = 0
        self.speakers_identified = 0
        self.hostile_detections = 0
        self.errors_occurred = 0
