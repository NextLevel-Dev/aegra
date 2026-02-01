"""AIURIS Legal AI Agent - A LangGraph agent for Croatian law assistance.

This agent provides:
- Evidence-grounded legal research
- One-tool-per-iteration control flow
- Structured JSON response envelope
- Citation tracking and management

Usage with Aegra:
    The graph is exported as a builder (not compiled) so Aegra can
    inject the checkpointer for persistence support.
"""

from aiuris_agent.graph import graph

__all__ = ["graph"]
