"""LangGraph state machine assembly for the AIURIS Legal AI Agent.

This module assembles the complete state machine by:
1. Creating the StateGraph with proper state schema
2. Adding all nodes
3. Defining edges and conditional routing
4. Exporting the builder (NOT compiled) for Aegra to inject checkpointer
"""

from langgraph.graph import StateGraph, START, END

from aiuris_agent.context import Context
from aiuris_agent.state import AiurisState, InputState, OutputState
from aiuris_agent.nodes import (
    intake_node,
    planning_node,
    execution_node,
    normalization_node,
    evaluation_node,
    finalize_node,
    direct_response_node,
    route_after_intake,
    route_after_planning,
    route_after_evaluation,
)


# =============================================================================
# Build the State Machine
# =============================================================================

# Create the graph builder with state and context schemas
# - AiurisState: full internal state
# - InputState: what can be passed when invoking
# - OutputState: what gets returned to clients (filtered view)
builder = StateGraph(
    AiurisState,
    input_schema=InputState,
    output_schema=OutputState,
    context_schema=Context,
)

# -----------------------------------------------------------------------------
# Add Nodes
# -----------------------------------------------------------------------------

# Intake: Classify if research is needed
builder.add_node("intake", intake_node)

# Research loop nodes
builder.add_node("planning", planning_node)
builder.add_node("execution", execution_node)
builder.add_node("normalization", normalization_node)
builder.add_node("evaluation", evaluation_node)

# Response generation nodes
builder.add_node("finalize", finalize_node)
builder.add_node("direct_response", direct_response_node)

# -----------------------------------------------------------------------------
# Add Edges
# -----------------------------------------------------------------------------

# Entry point
builder.add_edge(START, "intake")

# After intake: route based on whether research is needed
builder.add_conditional_edges(
    "intake",
    route_after_intake,
    {
        "planning": "planning",
        "direct_response": "direct_response",
    },
)

# Research loop: planning -> (execution -> normalization ->) evaluation
# Planning either calls a tool (go to execution) or decides it has enough info (skip to evaluation)
builder.add_conditional_edges(
    "planning",
    route_after_planning,
    {
        "execution": "execution",
        "evaluation": "evaluation",
    },
)
builder.add_edge("execution", "normalization")
builder.add_edge("normalization", "evaluation")

# After evaluation: either continue research or finalize
builder.add_conditional_edges(
    "evaluation",
    route_after_evaluation,
    {
        "planning": "planning",
        "finalize": "finalize",
    },
)

# Terminal nodes
builder.add_edge("finalize", END)
builder.add_edge("direct_response", END)

# -----------------------------------------------------------------------------
# Export
# -----------------------------------------------------------------------------

# Export the builder (NOT compiled) so Aegra can inject checkpointer
# Aegra will compile it with persistence support:
#   graph = builder.compile(checkpointer=checkpointer)
graph = builder


# =============================================================================
# Graph Visualization (for debugging)
# =============================================================================

def get_graph_mermaid() -> str:
    """Get a Mermaid diagram representation of the graph.
    
    Useful for documentation and debugging.
    """
    return """
stateDiagram-v2
    [*] --> intake
    intake --> planning: Research Needed
    intake --> direct_response: No Research
    
    state ResearchLoop {
        planning --> execution: Select Tool
        execution --> normalization: Execute
        normalization --> evaluation: Normalize
        evaluation --> planning: Need More
    }
    
    evaluation --> finalize: Sufficient
    finalize --> [*]
    direct_response --> [*]
"""
