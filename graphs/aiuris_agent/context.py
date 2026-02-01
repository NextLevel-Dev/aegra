"""Runtime context configuration for the AIURIS Legal AI Agent.

This module defines the configurable parameters that can be passed
to the agent at runtime via the Runtime[Context] pattern.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Annotated


@dataclass(kw_only=True)
class Context:
    """Runtime context for the AIURIS Legal AI Agent.
    
    This context is injected into the graph at runtime and can be
    customized per-invocation via the configurable system.
    """
    
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="google_genai/gemini-2.5-flash",
        metadata={
            "description": "The language model to use. "
            "Format: provider/model-name. Uses Google Gemini 2.5 Flash by default."
        },
    )
    
    thinking_budget: int = field(
        default=1024,
        metadata={
            "description": "Token budget for Gemini's native thinking (reasoning). "
            "Set to 0 to disable, -1 for dynamic (model decides), or positive int for limit. "
            "Only applies to Gemini 2.5 models."
        },
    )
    
    include_thoughts: bool = field(
        default=True,
        metadata={
            "description": "Whether to include Gemini's thinking/reasoning in responses. "
            "When True, the model's thought process is visible in the response."
        },
    )
    
    iteration_limit: int = field(
        default=20,
        metadata={
            "description": "Maximum number of iterations in the research loop. "
            "Prevents infinite loops and controls costs."
        },
    )
    
    finalize_threshold: int = field(
        default=3,
        metadata={
            "description": "Number of iterations before the limit to start "
            "considering finalization with available evidence."
        },
    )
    
    max_evidence_items: int = field(
        default=10,
        metadata={
            "description": "Maximum number of evidence items to collect. "
            "Helps manage context window and response quality."
        },
    )
    
    language: str = field(
        default="en",
        metadata={
            "description": "Output language code. 'en' for English, 'hr' for Croatian."
        },
    )
    
    def __post_init__(self) -> None:
        """Load configuration from environment variables if not provided."""
        for f in fields(self):
            if not f.init:
                continue
            
            # Check for environment variable override
            env_name = f"AIURIS_{f.name.upper()}"
            env_value = os.environ.get(env_name)
            
            if env_value is not None:
                # Convert to appropriate type
                if f.type == int or (hasattr(f.type, "__origin__") and f.type.__origin__ is int):
                    setattr(self, f.name, int(env_value))
                elif f.type == bool:
                    setattr(self, f.name, env_value.lower() in ("true", "1", "yes"))
                else:
                    setattr(self, f.name, env_value)
