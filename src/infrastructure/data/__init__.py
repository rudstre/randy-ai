"""
Data management infrastructure for conversations and person tracking.
"""

from .conversations import ConversationTurn, ConversationRecord
from .person_manager import PersonManager

__all__ = [
    'ConversationTurn',
    'ConversationRecord', 
    'PersonManager'
]
