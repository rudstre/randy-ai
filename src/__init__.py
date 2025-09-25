"""
LastBlackBox: AI-powered voice interview system with speaker identification.

A sophisticated conversation system that conducts interviews using voice analysis,
speech recognition, and LLM-powered decision making.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Main entry points
from .interview.orchestrator import InterviewOrchestrator
from .interview.models import Turn, InterviewResult

__all__ = ["InterviewOrchestrator", "Turn", "InterviewResult"]
