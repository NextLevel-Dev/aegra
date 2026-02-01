"""Pydantic schemas for the AIURIS Legal AI Agent.

This module defines:
1. Response envelope (wire contract) for API responses
2. Structured output schemas for LLM interactions
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# =============================================================================
# Response Envelope (Wire Contract)
# =============================================================================

class RunInfo(BaseModel):
    """Information about the current run."""
    
    run_id: str = Field(description="Unique identifier for this run")
    iteration_count: int = Field(description="Number of iterations completed")
    iteration_limit: int = Field(description="Maximum allowed iterations")
    stop_reason: Literal["finalized", "iteration_limit", "no_research_needed", "error"] = Field(
        description="Reason for stopping the agent"
    )


class CitationEntry(BaseModel):
    """A citation mapping inline markers to evidence."""
    
    cite: int = Field(description="The citation number (1, 2, 3...)")
    evidence_ids: list[str] = Field(description="Evidence record IDs this citation refers to")


class AssistantResponse(BaseModel):
    """The assistant's response content."""
    
    language: Literal["en", "hr"] = Field(
        default="en",
        description="Language of the response (en=English, hr=Croatian)"
    )
    final_text: str = Field(
        description="The final response text with inline citations like [1], [2]"
    )
    citations: list[CitationEntry] = Field(
        default_factory=list,
        description="Mappings from citation numbers to evidence IDs"
    )


class EvidenceRegistryEntry(BaseModel):
    """A single entry in the evidence registry for API response."""
    
    evidence_id: str = Field(description="Unique ID like E1, E2...")
    kind: Literal["law", "case", "attachment", "rag_chunk", "web"] = Field(
        description="Type of evidence"
    )
    title: str = Field(description="Display title for the evidence")


class ReasoningTrace(BaseModel):
    """Reasoning information for UI progress display."""
    
    reasoning_token_count: int = Field(
        default=0,
        description="Number of tokens used for reasoning"
    )
    titles: list[str] = Field(
        default_factory=list,
        description="Human-friendly reasoning step titles"
    )
    tool_titles: list[str] = Field(
        default_factory=list,
        description="Human-friendly tool call descriptions"
    )


class Diagnostics(BaseModel):
    """Diagnostic information about the run."""
    
    structured_output_retries: int = Field(
        default=0,
        description="Number of retries due to structured output failures"
    )
    last_error: str | None = Field(
        default=None,
        description="Most recent error message, if any"
    )


class ResponseEnvelope(BaseModel):
    """Complete response envelope matching the wire contract.
    
    This is the structured JSON response returned by the agent.
    """
    
    run: RunInfo = Field(description="Run metadata")
    assistant: AssistantResponse = Field(description="The assistant's response")
    evidence_registry: list[EvidenceRegistryEntry] = Field(
        default_factory=list,
        description="All evidence collected during the run"
    )
    reasoning_trace: ReasoningTrace = Field(
        default_factory=ReasoningTrace,
        description="Reasoning information for UI progress"
    )
    diagnostics: Diagnostics = Field(
        default_factory=Diagnostics,
        description="Diagnostic information"
    )


# =============================================================================
# Structured Output Schemas for LLM Interactions
# =============================================================================

class IntakeDecision(BaseModel):
    """Decision from the intake phase about whether research is needed."""
    
    research_needed: bool = Field(
        description="True if the query requires research/tool use, False for simple responses"
    )
    reasoning: str = Field(
        description="Brief explanation of why research is or isn't needed"
    )
    direct_response: str | None = Field(
        default=None,
        description="If research_needed is False, the direct response to the user"
    )


class ToolSelection(BaseModel):
    """Selection of a single tool to execute.
    
    IMPORTANT: Only ONE tool should be selected per iteration.
    """
    
    tool_name: Literal["rag_search", "fetch_law_by_id", "fetch_article", "web_search"] = Field(
        description="Name of the tool to execute"
    )
    tool_args: dict = Field(
        description="Arguments to pass to the tool"
    )
    reasoning: str = Field(
        description="Why this tool was selected and what information it should provide"
    )
    reasoning_title: str = Field(
        description="A short (5-10 word) human-friendly title for this step"
    )


class SufficiencyCheck(BaseModel):
    """Evaluation of whether collected evidence is sufficient."""
    
    is_sufficient: bool = Field(
        description="True if evidence is sufficient to answer the query"
    )
    reasoning: str = Field(
        description="Explanation of why evidence is or isn't sufficient"
    )
    missing_information: str | None = Field(
        default=None,
        description="If not sufficient, what information is still needed"
    )


class FinalResponseCitation(BaseModel):
    """A citation in the final response."""
    
    cite_number: int = Field(description="Citation number (1, 2, 3...)")
    evidence_ids: list[str] = Field(description="Evidence IDs being cited")


class FinalResponse(BaseModel):
    """The final response generated by the agent.
    
    The response should be comprehensive, accurate, and grounded in evidence.
    All claims should be supported by citations.
    """
    
    final_text: str = Field(
        description="The complete response text with inline citations [1], [2], etc."
    )
    citations: list[FinalResponseCitation] = Field(
        description="List of citations mapping numbers to evidence IDs"
    )
    reasoning_title: str = Field(
        description="A short title summarizing this response generation"
    )


class DirectResponse(BaseModel):
    """A direct response for queries that don't need research."""
    
    response_text: str = Field(
        description="The response text (for greetings, clarifications, etc.)"
    )
    reasoning_title: str = Field(
        default="Direct response",
        description="A short title for this response"
    )


class CitationExtraction(BaseModel):
    """Extract which evidence sources were actually cited in a response."""
    
    citations_used: list[str] = Field(
        description="List of evidence IDs that were cited in the response (e.g., ['E1', 'E2', 'E3']). "
        "Only include IDs that were actually referenced with [N] markers in the text."
    )
