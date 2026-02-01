"""Node implementations for the AIURIS Legal AI Agent.

This module contains all the nodes that make up the agent's state machine:
- intake_node: Classify if research is needed
- planning_node: Use standard tool calling to select next action
- execution_node: Execute the tool call from the AIMessage
- normalization_node: Convert tool output to evidence
- evaluation_node: Check if evidence is sufficient
- finalize_node: Generate the final response
- direct_response_node: Handle simple queries
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal, cast

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.runtime import Runtime

from aiuris_agent.context import Context
from aiuris_agent.prompts import (
    INTAKE_PROMPT,
    PLANNING_PROMPT,
    EVALUATION_PROMPT,
    FINALIZE_PROMPT,
    DIRECT_RESPONSE_PROMPT,
    get_system_time,
    format_evidence_summary,
    format_evidence_details,
)
from aiuris_agent.schemas import (
    IntakeDecision,
    SufficiencyCheck,
    CitationExtraction,
)
from aiuris_agent.state import AiurisState
from aiuris_agent.tools import TOOLS, execute_tool_call

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================

def get_model(runtime: Runtime[Context]):
    """Get the configured LLM model with Gemini thinking enabled.
    
    Uses ChatGoogleGenerativeAI directly to enable native thinking (reasoning)
    which provides better quality responses and reasoning traces.
    """
    model_str = runtime.context.model
    
    # Parse provider/model format
    if "/" in model_str:
        provider, model_name = model_str.split("/", maxsplit=1)
    else:
        provider = "google_genai"
        model_name = model_str
    
    # For Google Gemini models, use native thinking
    if provider == "google_genai":
        return ChatGoogleGenerativeAI(
            model=model_name,
            thinking_budget=runtime.context.thinking_budget,
            include_thoughts=runtime.context.include_thoughts,
        )
    else:
        # Fallback for other providers (shouldn't happen in normal use)
        from langchain.chat_models import init_chat_model
        return init_chat_model(model_name, model_provider=provider)


def get_last_user_message(state: AiurisState) -> str:
    """Extract the last user message from state."""
    for message in reversed(state.messages):
        if isinstance(message, HumanMessage):
            content = message.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # Handle multimodal content
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        return item.get("text", "")
                    elif isinstance(item, str):
                        return item
            return str(content)
    return ""


def get_conversation_context(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Filter messages to get only the conversation context for LLM.
    
    This excludes tool calls and tool results from the context window,
    keeping only human messages and final AI responses (without tool_calls).
    
    This is important for controlling what goes into the LLM's context window
    between turns - we don't want to bloat it with tool call details.
    """
    context = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            context.append(msg)
        elif isinstance(msg, AIMessage):
            # Only include AI messages that have actual content and no tool_calls
            # (i.e., final responses, not intermediate tool-calling messages)
            if msg.content and not msg.tool_calls:
                context.append(msg)
        # Skip ToolMessages - they don't need to be in the conversation context
    return context


def extract_text_from_content(content: str | list) -> str:
    """Extract plain text from AI message content.
    
    When Gemini thinking is enabled, content is a list like:
    [{"type": "thinking", "thinking": "..."}, "actual text response"]
    
    This extracts just the text for citation extraction.
    """
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "".join(text_parts)
    
    return str(content)


def parse_tool_result_for_evidence(tool_name: str, result_str: str) -> dict | None:
    """Parse a tool result string back into a dict for evidence processing."""
    try:
        return json.loads(result_str)
    except json.JSONDecodeError:
        return None


# =============================================================================
# Intake Node
# =============================================================================

async def intake_node(
    state: AiurisState,
    runtime: Runtime[Context],
) -> dict[str, Any]:
    """Classify whether the user's query requires research.
    
    This node analyzes the incoming message and decides:
    - If research is needed: proceed to planning phase
    - If no research needed: provide direct response
    
    Also resets transient state for new runs to avoid stale data.
    """
    logger.info("Intake node: Analyzing user query")
    
    # Reset transient state for this new run
    reset_updates = state.reset_transient_state()
    
    model = get_model(runtime)
    structured_model = model.with_structured_output(IntakeDecision)
    
    user_message = get_last_user_message(state)
    
    prompt = INTAKE_PROMPT.format(system_time=get_system_time())
    
    try:
        decision = cast(
            IntakeDecision,
            await structured_model.ainvoke([
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message},
            ])
        )
        
        logger.info(f"Intake decision: research_needed={decision.research_needed}")
        
        return {
            **reset_updates,
            "current_phase": "planning" if decision.research_needed else "direct_response",
            "research_needed": decision.research_needed,
            "reasoning_titles": [f"Analysis: {decision.reasoning[:50]}..."],
        }
        
    except Exception as e:
        logger.error(f"Intake node error: {e}")
        return {
            **reset_updates,
            "current_phase": "direct_response",
            "research_needed": False,
            "last_error": str(e),
            "structured_output_retries": 1,
        }


# =============================================================================
# Planning Node (Standard Tool Calling)
# =============================================================================

async def planning_node(
    state: AiurisState,
    runtime: Runtime[Context],
) -> dict[str, Any]:
    """Select and initiate the next tool call using standard LangChain tool calling.
    
    This node uses bind_tools() to let the LLM naturally decide which tool to use.
    The LLM will produce an AIMessage with tool_calls that the execution node handles.
    """
    logger.info(f"Planning node: Iteration {state.iteration_count + 1}")
    
    # Check iteration limit FIRST - if we've hit the limit, skip to evaluation->finalize
    if state.iteration_count >= state.iteration_limit:
        logger.warning(f"Iteration limit ({state.iteration_limit}) reached in planning, forcing evaluation")
        return {
            "current_phase": "evaluation",  # Will route to evaluation, which will then finalize
            "stop_reason": "iteration_limit",
        }
    
    model = get_model(runtime)
    # Bind our tools to the model for native tool calling
    model_with_tools = model.bind_tools(TOOLS)
    
    user_question = get_last_user_message(state)
    evidence_summary = format_evidence_summary(state.evidence_registry)
    
    # Check if approaching iteration limit
    approaching_limit = state.is_approaching_limit(threshold=3)
    
    prompt = PLANNING_PROMPT.format(
        system_time=get_system_time(),
        iteration_count=state.iteration_count + 1,
        iteration_limit=state.iteration_limit,
        evidence_count=len(state.evidence_registry),
        evidence_summary=evidence_summary,
        user_question=user_question,
    )
    
    if approaching_limit:
        prompt += "\n\n**WARNING: Approaching iteration limit. Consider if current evidence is sufficient.**"
    
    # Add instruction to call exactly one tool
    prompt += "\n\n**IMPORTANT: Call exactly ONE tool to gather information. Do not call multiple tools.**"
    
    try:
        # Build messages for the LLM - use filtered conversation context
        messages = [
            {"role": "system", "content": prompt},
        ]
        
        # Add conversation context (filtered to exclude tool calls/results)
        conversation_context = get_conversation_context(list(state.messages))
        messages.extend(conversation_context)
        
        # Call the model with tools bound
        response = await model_with_tools.ainvoke(messages)
        
        logger.info(f"Planning response: tool_calls={len(response.tool_calls) if response.tool_calls else 0}")
        
        # Check if the model made a tool call
        if response.tool_calls and len(response.tool_calls) > 0:
            # Take only the first tool call (enforce one-tool-per-iteration)
            tool_call = response.tool_calls[0]
            
            # Create AIMessage with just this one tool call for UI display
            ai_message = AIMessage(
                content="",
                tool_calls=[tool_call],
            )
            
            return {
                "messages": [ai_message],
                "current_phase": "execution",
                "iteration_count": state.iteration_count + 1,
                "reasoning_titles": state.reasoning_titles + [f"Calling {tool_call['name']}"],
            }
        else:
            # No tool call - model thinks we have enough info, go to evaluation
            logger.info("No tool call made, proceeding to evaluation")
            return {
                "current_phase": "evaluation",
                "iteration_count": state.iteration_count + 1,
                "reasoning_titles": state.reasoning_titles + ["Evaluating collected evidence"],
            }
        
    except Exception as e:
        logger.error(f"Planning node error: {e}")
        return {
            "current_phase": "evaluation",
            "last_error": str(e),
            "structured_output_retries": state.structured_output_retries + 1,
        }


# =============================================================================
# Execution Node
# =============================================================================

async def execution_node(
    state: AiurisState,
    runtime: Runtime[Context],
) -> dict[str, Any]:
    """Execute the tool call from the last AIMessage.
    
    Reads tool_calls from the most recent AIMessage and executes the tool,
    creating a ToolMessage with the result.
    """
    logger.info("Execution node: Running tool")
    
    # Find the last AIMessage with tool_calls
    last_ai_message = None
    for msg in reversed(state.messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            last_ai_message = msg
            break
    
    if not last_ai_message or not last_ai_message.tool_calls:
        logger.warning("No tool call found in messages")
        return {
            "current_phase": "evaluation",
            "tool_error": "No tool call found",
        }
    
    # Get the first tool call
    tool_call = last_ai_message.tool_calls[0]
    tool_name = tool_call.get("name", "unknown")
    tool_call_id = tool_call.get("id", "")
    
    logger.info(f"Executing tool: {tool_name}")
    
    try:
        # Execute the tool
        result = execute_tool_call(tool_call)
        
        # Create a ToolMessage with the result
        tool_message = ToolMessage(
            content=result,
            name=tool_name,
            tool_call_id=tool_call_id,
        )
        
        # Create human-friendly tool title
        args_str = ", ".join(f"{k}={v!r}" for k, v in tool_call.get("args", {}).items())
        tool_title = f"{tool_name}({args_str})"
        
        logger.info(f"Tool execution successful: {tool_name}")
        
        return {
            "messages": [tool_message],
            "current_phase": "normalization",
            "tool_result": result,  # Store for normalization
            "tool_error": None,
            "tool_titles": state.tool_titles + [tool_title],
        }
        
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        
        # Create error ToolMessage
        error_message = ToolMessage(
            content=f"Error: {str(e)}",
            name=tool_name,
            tool_call_id=tool_call_id,
        )
        
        return {
            "messages": [error_message],
            "current_phase": "evaluation",
            "tool_result": None,
            "tool_error": str(e),
        }


# =============================================================================
# Normalization Node
# =============================================================================

async def normalization_node(
    state: AiurisState,
    runtime: Runtime[Context],
) -> dict[str, Any]:
    """Normalize tool output into evidence records.
    
    This node converts raw tool results into standardized
    EvidenceRecord entries in the evidence registry.
    """
    logger.info("Normalization node: Processing tool result")
    
    if state.tool_result is None:
        return {"current_phase": "evaluation"}
    
    # Parse the tool result (it's a JSON string from the tool)
    result = parse_tool_result_for_evidence("", state.tool_result)
    
    if result is None:
        logger.warning("Could not parse tool result")
        return {"current_phase": "evaluation"}
    
    # Handle RAG search results
    if "chunks" in result:
        for chunk in result.get("chunks", []):
            state.add_evidence(
                kind="rag_chunk",
                title=f"RAG: {chunk.get('law_id', 'Unknown')} Art. {chunk.get('article', '?')}",
                locator={
                    "chunk_id": chunk.get("chunk_id"),
                    "law_id": chunk.get("law_id"),
                    "article": chunk.get("article"),
                    "content": chunk.get("content"),
                    "score": chunk.get("score"),
                },
            )
    
    # Handle law document
    elif "law_id" in result and "articles" in result:
        state.add_evidence(
            kind="law",
            title=result.get("short_title", result.get("law_id")),
            locator={
                "law_id": result.get("law_id"),
                "full_title": result.get("full_title"),
                "articles": list(result.get("articles", {}).keys()),
            },
        )
    
    # Handle article content
    elif "law_id" in result and "article_num" in result and "content" in result:
        state.add_evidence(
            kind="law",
            title=f"{result.get('law_id')} Article {result.get('article_num')}: {result.get('title', '')}",
            locator={
                "law_id": result.get("law_id"),
                "article_num": result.get("article_num"),
                "content": result.get("content"),
            },
        )
    
    # Handle web search results
    elif "results" in result and isinstance(result.get("results"), list):
        for web_result in result.get("results", []):
            state.add_evidence(
                kind="web",
                title=web_result.get("title", "Web Result"),
                locator={
                    "url": web_result.get("url"),
                    "snippet": web_result.get("snippet"),
                },
            )
    
    logger.info(f"Evidence registry now has {len(state.evidence_registry)} items")
    
    return {
        "current_phase": "evaluation",
        "evidence_registry": state.evidence_registry,
    }


# =============================================================================
# Evaluation Node
# =============================================================================

async def evaluation_node(
    state: AiurisState,
    runtime: Runtime[Context],
) -> dict[str, Any]:
    """Evaluate if collected evidence is sufficient.
    
    This node determines whether to:
    - Continue researching (go back to planning)
    - Finalize the response (evidence is sufficient)
    - Force finalize (iteration limit reached)
    """
    logger.info(f"Evaluation node: {len(state.evidence_registry)} evidence items")
    
    # Check iteration limit
    if state.iteration_count >= state.iteration_limit:
        logger.warning("Iteration limit reached, forcing finalization")
        return {
            "current_phase": "finalize",
            "stop_reason": "iteration_limit",
        }
    
    # If no evidence yet, continue planning
    if not state.evidence_registry:
        return {"current_phase": "planning"}
    
    model = get_model(runtime)
    structured_model = model.with_structured_output(SufficiencyCheck)
    
    user_question = get_last_user_message(state)
    evidence_details = format_evidence_details(state.evidence_registry)
    
    prompt = EVALUATION_PROMPT.format(
        system_time=get_system_time(),
        user_question=user_question,
        evidence_count=len(state.evidence_registry),
        evidence_details=evidence_details,
        iteration_count=state.iteration_count,
        iteration_limit=state.iteration_limit,
    )
    
    try:
        # Use filtered conversation context for evaluation
        user_content = f"Evaluate if the collected evidence is sufficient to answer: {user_question}"
        
        check = cast(
            SufficiencyCheck,
            await structured_model.ainvoke([
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ])
        )
        
        logger.info(f"Sufficiency check: is_sufficient={check.is_sufficient}")
        
        if check.is_sufficient:
            return {
                "current_phase": "finalize",
                "reasoning_titles": state.reasoning_titles + [f"Evidence sufficient: {check.reasoning[:40]}..."],
            }
        else:
            return {
                "current_phase": "planning",
                "reasoning_titles": state.reasoning_titles + [f"Need more: {check.missing_information or check.reasoning[:40]}..."],
            }
            
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return {
            "current_phase": "finalize",
            "last_error": str(e),
        }


# =============================================================================
# Citation Extraction Helper
# =============================================================================

async def _extract_citations(
    model,
    response_text: str,
    evidence_registry: list,
) -> list[str]:
    """Extract which evidence IDs were actually cited in the response.
    
    Uses a simple structured output call to identify citations.
    """
    # Build evidence summary for context
    evidence_list = "\n".join(
        f"- {e.evidence_id}: {e.title}"
        for e in evidence_registry
    )
    
    extraction_prompt = f"""Look at this response and identify which evidence sources were cited.

Evidence available:
{evidence_list}

Response text:
{response_text}

List ONLY the evidence IDs (like E1, E2, E3) that were actually cited in the response with [N] markers.
For example, if the response contains [1] and [2], return ["E1", "E2"]."""

    try:
        structured_model = model.with_structured_output(CitationExtraction)
        result = await structured_model.ainvoke([
            {"role": "user", "content": extraction_prompt}
        ])
        return result.citations_used
    except Exception as e:
        logger.warning(f"Citation extraction failed: {e}")
        return []


# =============================================================================
# Finalize Node
# =============================================================================

async def finalize_node(
    state: AiurisState,
    runtime: Runtime[Context],
) -> dict[str, Any]:
    """Generate the final response with citations.
    
    Calls the LLM naturally to generate a response based on collected evidence.
    The response is a standard AIMessage - no structured output needed.
    """
    logger.info("Finalize node: Generating response")
    
    model = get_model(runtime)
    
    user_question = get_last_user_message(state)
    evidence_details = format_evidence_details(state.evidence_registry)
    
    prompt = FINALIZE_PROMPT.format(
        system_time=get_system_time(),
        user_question=user_question,
        evidence_details=evidence_details,
    )
    
    try:
        # Build messages - use conversation context + current question
        messages = [{"role": "system", "content": prompt}]
        conversation_context = get_conversation_context(list(state.messages))
        messages.extend(conversation_context)
        
        # Call model naturally - returns AIMessage
        response = await model.ainvoke(messages)
        
        logger.info("Response generated successfully")
        
        # Extract which citations were actually used
        citations_used = []
        if state.evidence_registry:
            response_text = extract_text_from_content(response.content)
            citations_used = await _extract_citations(
                model, response_text, state.evidence_registry
            )
            logger.info(f"Citations used: {citations_used}")
        
        stop_reason = "finalized"
        if state.stop_reason == "iteration_limit":
            stop_reason = "iteration_limit"
        
        return {
            "messages": [response],
            "citations_used": citations_used,
            "stop_reason": stop_reason,
        }
        
    except Exception as e:
        logger.error(f"Finalize error: {e}")
        
        fallback_text = f"I found {len(state.evidence_registry)} relevant sources about your question, but encountered an error generating the full response. Please try again."
        
        return {
            "messages": [AIMessage(content=fallback_text)],
            "citations_used": [],
            "stop_reason": "error",
            "last_error": str(e),
        }


# =============================================================================
# Direct Response Node
# =============================================================================

async def direct_response_node(
    state: AiurisState,
    runtime: Runtime[Context],
) -> dict[str, Any]:
    """Handle queries that don't need research.
    
    Handles greetings, clarifications, and simple queries
    that don't require tool use. Just calls the LLM naturally.
    """
    logger.info("Direct response node: No research needed")
    
    model = get_model(runtime)
    
    prompt = DIRECT_RESPONSE_PROMPT.format(
        system_time=get_system_time(),
    )
    
    try:
        # Build messages with conversation context
        messages = [{"role": "system", "content": prompt}]
        conversation_context = get_conversation_context(list(state.messages))
        messages.extend(conversation_context)
        
        # Call model naturally - returns AIMessage
        response = await model.ainvoke(messages)
        
        return {
            "messages": [response],
            "stop_reason": "no_research_needed",
        }
        
    except Exception as e:
        logger.error(f"Direct response error: {e}")
        
        fallback = "Hello! I'm AIURIS, your legal assistant. How can I help you today?"
        
        return {
            "messages": [AIMessage(content=fallback)],
            "stop_reason": "no_research_needed",
            "last_error": str(e),
        }


# =============================================================================
# Routing Functions
# =============================================================================

def route_after_intake(state: AiurisState) -> Literal["planning", "direct_response"]:
    """Route after intake based on whether research is needed."""
    if state.research_needed:
        return "planning"
    return "direct_response"


def route_after_planning(state: AiurisState) -> Literal["execution", "evaluation"]:
    """Route after planning based on whether a tool call was made.
    
    If the LLM made a tool call, go to execution.
    If no tool call (LLM thinks it has enough info), skip to evaluation.
    """
    if state.current_phase == "execution":
        return "execution"
    return "evaluation"


def route_after_evaluation(state: AiurisState) -> Literal["planning", "finalize"]:
    """Route after evaluation based on sufficiency check."""
    if state.current_phase == "finalize":
        return "finalize"
    return "planning"
