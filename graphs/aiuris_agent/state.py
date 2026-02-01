"""State definitions for the AIURIS Legal AI Agent.

This module defines the state schema for the legal assistant agent,
including the evidence registry for tracking retrieved sources.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, asdict
from typing import Annotated, Any, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep


@dataclass
class EvidenceRecord:
    """A single piece of evidence retrieved during research.
    
    The evidence registry is append-only and run-scoped.
    Only the title is intended for UI display.
    """
    
    evidence_id: str  # e.g., "E1", "E12"
    kind: Literal["law", "case", "attachment", "rag_chunk", "web"]
    title: str  # Display label only (law name, case name, website name)
    locator: dict[str, Any]  # Internal pointers (IDs, URLs, offsets)
    created_at_step: int
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_display_dict(self) -> dict[str, Any]:
        """Convert to dictionary for client display (no internal locator data)."""
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "title": self.title,
        }


@dataclass
class InputState:
    """Input state for the AIURIS agent.
    
    This defines what can be passed when invoking the graph.
    """
    
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    """Conversation messages. Uses add_messages reducer for proper merging."""


@dataclass
class AiurisState(InputState):
    """Complete state for the AIURIS Legal AI Agent.
    
    Extends InputState with internal state for research tracking,
    evidence collection, and response generation.
    """
    
    # Evidence tracking (run-scoped, not persisted)
    evidence_registry: list[EvidenceRecord] = field(default_factory=list)
    """Append-only registry of all evidence collected during this run."""
    
    # Iteration control
    iteration_count: int = field(default=0)
    """Current iteration number within the research loop."""
    
    iteration_limit: int = field(default=20)
    """Maximum iterations before forced finalization."""
    
    # Phase tracking
    current_phase: Literal[
        "intake", "planning", "execution", "normalization", 
        "evaluation", "finalize", "direct_response"
    ] = field(default="intake")
    """Current phase of the agent's execution."""
    
    # Research state
    research_needed: bool = field(default=False)
    """Whether research/tool use is needed for this query."""
    
    tool_result: Any = field(default=None)
    """Raw result from the most recent tool execution (for normalization)."""
    
    tool_error: str | None = field(default=None)
    """Error message if the tool execution failed."""
    
    # Reasoning trace (for UI progress, not persisted)
    reasoning_titles: list[str] = field(default_factory=list)
    """Human-friendly titles summarizing reasoning steps."""
    
    tool_titles: list[str] = field(default_factory=list)
    """Human-friendly titles for tool calls made."""
    
    # Error handling and diagnostics
    structured_output_retries: int = field(default=0)
    """Number of retries due to structured output validation failures."""
    
    last_error: str | None = field(default=None)
    """Most recent error message, if any."""
    
    citations_used: list[str] = field(default_factory=list)
    """Evidence IDs that were actually cited in the response (e.g., ["E1", "E2"])."""
    
    stop_reason: Literal[
        "finalized", "iteration_limit", "no_research_needed", "error"
    ] | None = field(default=None)
    """Reason for stopping the agent execution."""
    
    # Managed variable
    is_last_step: IsLastStep = field(default=False)
    """Indicates if this is the last step before the graph raises an error."""
    
    def get_next_evidence_id(self) -> str:
        """Generate the next sequential evidence ID."""
        return f"E{len(self.evidence_registry) + 1}"
    
    def add_evidence(
        self,
        kind: Literal["law", "case", "attachment", "rag_chunk", "web"],
        title: str,
        locator: dict[str, Any],
    ) -> EvidenceRecord:
        """Add a new evidence record to the registry.
        
        Args:
            kind: Type of evidence
            title: Display title for the evidence
            locator: Internal location pointers
            
        Returns:
            The created EvidenceRecord
        """
        record = EvidenceRecord(
            evidence_id=self.get_next_evidence_id(),
            kind=kind,
            title=title,
            locator=locator,
            created_at_step=self.iteration_count,
        )
        self.evidence_registry.append(record)
        return record
    
    def is_approaching_limit(self, threshold: int = 2) -> bool:
        """Check if we're approaching the iteration limit.
        
        Args:
            threshold: How many iterations before limit to trigger
            
        Returns:
            True if we should finalize soon
        """
        return self.iteration_count >= self.iteration_limit - threshold
    
    def get_evidence_by_id(self, evidence_id: str) -> EvidenceRecord | None:
        """Look up an evidence record by its ID."""
        for record in self.evidence_registry:
            if record.evidence_id == evidence_id:
                return record
        return None
    
    def reset_transient_state(self) -> dict[str, Any]:
        """Get updates to reset transient state for a new run.
        
        Returns state updates to clear per-run transient fields
        while preserving messages (conversation history).
        """
        return {
            "evidence_registry": [],
            "iteration_count": 0,
            "current_phase": "intake",
            "research_needed": False,
            "tool_result": None,
            "tool_error": None,
            "reasoning_titles": [],
            "tool_titles": [],
            "structured_output_retries": 0,
            "last_error": None,
            "citations_used": [],
            "stop_reason": None,
        }


@dataclass
class OutputState:
    """Output state for client consumption.
    
    This defines what gets returned to clients via the API.
    It excludes internal state like tool_result, selected_tool, etc.
    """
    
    # Conversation messages (what gets persisted)
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    """Conversation messages visible to the client."""
    
    # Evidence for UI display (titles only, no internal locators)
    evidence_registry: list[EvidenceRecord] = field(default_factory=list)
    """Evidence records for citation display."""
    
    # Reasoning trace for UI progress
    reasoning_titles: list[str] = field(default_factory=list)
    """Human-friendly reasoning step titles for UI."""
    
    tool_titles: list[str] = field(default_factory=list)
    """Human-friendly tool call descriptions for UI."""
    
    # Citation tracking
    citations_used: list[str] = field(default_factory=list)
    """Evidence IDs that were actually cited in the response (e.g., ["E1", "E2"])."""
    
    # Run metadata
    stop_reason: Literal[
        "finalized", "iteration_limit", "no_research_needed", "error"
    ] | None = field(default=None)
    """Reason for stopping the agent execution."""
    
    iteration_count: int = field(default=0)
    """Number of research iterations completed."""
