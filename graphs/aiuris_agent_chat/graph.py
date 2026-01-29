"""Define the AIURIS Agent Chat graph.

Works with a chat model with tool calling support.
"""

from datetime import UTC, datetime
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from aiuris_agent_chat.context import Context
from aiuris_agent_chat.state import InputState, State
from aiuris_agent_chat.tools import TOOLS as BASE_TOOLS, get_all_tools
from aiuris_agent_chat.utils import load_chat_model, extract_thinking_from_message
from aiuris_agent_chat import prompts

# Define the function that calls the model


async def call_model(
    state: State, runtime: Runtime[Context]
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    # Initialize the model with tool binding. Change the model or add more tools here.
    tools = await get_all_tools()
    
    # Use Vertex AI specific tool configuration instead of parallel_tool_calls
    # 
    # VERTEX AI TOOL CONFIGURATION OPTIONS:
    # 
    # 1. AUTO mode (default): Let model decide whether to call tools
    # tool_config = {
    #     "function_calling_config": {
    #         "mode": "AUTO"
    #     }
    # }
    #
    # 2. ANY mode: Force model to always call a tool (use with caution - can cause loops)
    # tool_config = {
    #     "function_calling_config": {
    #         "mode": "ANY",
    #         "allowed_function_names": ["laws_rag_search", "web_search"]  # Required with ANY
    #     }
    # }
    #
    # 3. AUTO mode with restricted tool subset:
    # tool_config = {
    #     "function_calling_config": {
    #         "mode": "AUTO",
    #         "allowed_function_names": ["laws_rag_search", "judicial_practice_rag_search"]
    #     }
    # }
    #
    # The tools_node below will still sequentialize to 1 tool call per turn regardless of config
    tool_config = {
        "function_calling_config": {
            "mode": "AUTO",  # Let model decide whether to call tools
            # Optionally restrict to specific tool names:
            # "allowed_function_names": ["laws_rag_search", "judicial_practice_rag_search", "web_search"]
        }
    }
    
    model = load_chat_model(runtime.context.model, runtime.context).bind_tools(
        tools, 
        tool_config=tool_config
    )

    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = runtime.context.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )

    # Debug prints: raw objects and content to help with parsing issues downstream
    try:
        print("[agent] call_model: raw AIMessage:", repr(response))
    except Exception as e:
        print("[agent] call_model: failed to repr AIMessage:", repr(e))
    try:
        print("[agent] call_model: response.content:", response.content)
    except Exception as e:
        print("[agent] call_model: failed to print content:", repr(e))
    try:
        print("[agent] call_model: response_metadata:", response.response_metadata)
    except Exception as e:
        print("[agent] call_model: failed to print response_metadata:", repr(e))

    # Let the agent respond naturally - the frontend will handle display logic based on source_node metadata

    # Add source node tagging for frontend display control
    try:
        response.response_metadata = dict(response.response_metadata or {})
        response.response_metadata["source_node"] = "call_model"
        print("[agent] call_model: tagged message with source_node=call_model")
    except Exception as e:
        print("[agent] call_model: failed to add source_node metadata:", repr(e))

    # Extract and attach model "thinking" if available
    thinking_text, thinking_parts = extract_thinking_from_message(response)
    if thinking_text:
        try:
            response.response_metadata = dict(response.response_metadata or {})
            response.response_metadata["thinking"] = thinking_text
            response.response_metadata["thinking_parts"] = thinking_parts
            # Also mirror into additional_kwargs for easier JSON serialization on some clients
            response.additional_kwargs = dict(getattr(response, "additional_kwargs", {}) or {})
            response.additional_kwargs["thinking"] = thinking_text
            response.additional_kwargs["thinking_parts"] = thinking_parts
            print("[agent] call_model: extracted thinking length:", len(thinking_text))
        except Exception as e:
            print("[agent] call_model: failed to attach thinking metadata:", repr(e))
    else:
        print("[agent] call_model: no thinking found in message content.")

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        print("[agent] call_model: Last step reached with tool calls, forcing finalization")
        # Clear tool calls to trigger finalization - the finalize node will synthesize from gathered data
        response = AIMessage(
            id=response.id,
            content=response.content,  # Preserve any reasoning/content
            tool_calls=[],  # Clear tool calls to route to finalize
            additional_kwargs=response.additional_kwargs,
            response_metadata=response.response_metadata,
        )

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}


async def finalize_answer(state: State, runtime: Runtime[Context]) -> Dict[str, List[AIMessage]]:
    """Run a final LLM pass to synthesize a response or ask a clarifying question."""
    print(f"[agent] finalize_answer: Starting finalization, is_last_step={state.is_last_step}")

    # Prepare finalization model without tools - use same model and configuration as main model
    model = load_chat_model(runtime.context.model, runtime.context)

    system_message = prompts.FINALIZE_PROMPT

    # Clean messages to remove internal routing messages and thinking parts before sending to the finalizer LLM
    clean_messages: List[BaseMessage] = []
    for msg in state.messages:
        # Skip AIMessages from call_model that have no tool_calls (these are just routing messages)
        if isinstance(msg, AIMessage) and not getattr(msg, 'tool_calls', None):
            # Check if this is from call_model by looking at response_metadata
            if (hasattr(msg, 'response_metadata') and
                msg.response_metadata and
                msg.response_metadata.get('source_node') == 'call_model'):
                continue  # Skip internal routing messages

        # We only want to clean AIMessages that have list-based content
        if isinstance(msg, AIMessage) and isinstance(msg.content, list):
            new_msg = msg.copy(deep=True)
            
            # 1. Filter out thinking parts
            filtered_content = [
                part
                for part in new_msg.content
                if not (
                    isinstance(part, dict)
                    and str(part.get("type", "")).lower()
                    in {"reasoning", "thought", "thoughts", "thinking"}
                )
            ]

            # 2. Process and strip remaining parts, keeping only non-empty string content
            new_content = []
            for part in filtered_content:
                if isinstance(part, str):
                    stripped_part = part.strip()
                    if stripped_part:
                        new_content.append(stripped_part)
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    stripped_text = part["text"].strip()
                    if stripped_text:
                        part["text"] = stripped_text
                        new_content.append(part)

            # If after filtering and stripping, the content is empty, we skip this message entirely.
            if not new_content:
                continue

            # After processing, if content is just a list with one string, flatten it
            if len(new_content) == 1 and isinstance(new_content[0], str):
                new_msg.content = new_content[0]
            else:
                new_msg.content = new_content
            
            clean_messages.append(new_msg)
        else:
            # Append HumanMessages or AIMessages with simple string content
            clean_messages.append(msg)

    # Debug: log the number of messages being passed to finalize
    print(f"[agent] finalize_answer: processing {len(clean_messages)} messages")
    for i, msg in enumerate(clean_messages):
        print(f"[agent] finalize_answer: message {i}: type={type(msg).__name__}, content_preview={str(msg.content)[:100] if msg.content else 'None'}...")

    # Provide the prior messages as context along with the finalize system prompt
    response = cast(
        AIMessage,
        await model.ainvoke(
            [
                {"role": "system", "content": system_message},
                *clean_messages,
            ]
        ),
    )

    # Debug: log the finalizer's raw response
    print(f"[agent] finalize_answer: finalizer raw response: {repr(response)}")
    print(f"[agent] finalize_answer: finalizer response content: {response.content}")

    # Add source node tagging for frontend display control
    try:
        response.response_metadata = dict(response.response_metadata or {})
        response.response_metadata["source_node"] = "finalize"
        print("[agent] finalize_answer: tagged message with source_node=finalize")
    except Exception as e:
        print("[agent] finalize_answer: failed to add source_node metadata:", repr(e))

    # Extract and attach model "thinking" if available (same as call_model)
    thinking_text, thinking_parts = extract_thinking_from_message(response)
    if thinking_text:
        try:
            response.response_metadata = dict(response.response_metadata or {})
            response.response_metadata["thinking"] = thinking_text
            response.response_metadata["thinking_parts"] = thinking_parts
            response.additional_kwargs = dict(
                getattr(response, "additional_kwargs", {}) or {}
            )
            response.additional_kwargs["thinking"] = thinking_text
            response.additional_kwargs["thinking_parts"] = thinking_parts
        except Exception as e:
            print(
                "[agent] finalize_answer: failed to attach thinking metadata:", repr(e)
            )

    # Return the full response with thinking tokens preserved
    print("[agent] finalize_answer: Finalization complete, returning response")
    return {"messages": [response]}


async def tools_node(state: State) -> Dict:
    """Execute only the first requested tool call, sequentializing tool usage.
    
    This ensures only one tool is executed per turn, even if the model requests multiple.
    This node replaces the last message with a new message containing only the executed tool call.
    """
    from copy import deepcopy

    # Take the last AI message
    last = state.messages[-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        # No tool calls to execute
        return {}
    
    try:
        print(f"[agent] tools_node: incoming tool_calls count: {len(last.tool_calls)}")
        print("[agent] tools_node: incoming tool_calls:", [tc.get('name', 'unknown') for tc in last.tool_calls])
    except Exception:
        pass
    
    # Create a sanitized message with only the first tool call
    sanitized_message = AIMessage(
        content=last.content,
        tool_calls=[last.tool_calls[0]],
        # Copy other attributes for continuity
        id=last.id,
        name=getattr(last, "name", None),
        additional_kwargs=deepcopy(getattr(last, "additional_kwargs", {})),
        response_metadata=deepcopy(getattr(last, "response_metadata", {})),
    )
    
    print(f"[agent] tools_node: executing only first tool: {sanitized_message.tool_calls[0].get('name', 'unknown')}")
    
    # Execute the single tool call
    tools = await get_all_tools()
    executor = ToolNode(tools)
    result = await executor.ainvoke({"messages": [sanitized_message]})
    
    # Add source node tagging to tool messages
    if "messages" in result:
        for msg in result["messages"]:
            try:
                # Add source_node metadata to tool messages
                if hasattr(msg, 'response_metadata'):
                    msg.response_metadata = dict(getattr(msg, 'response_metadata', {}) or {})
                    msg.response_metadata["source_node"] = "tools"
                elif hasattr(msg, 'additional_kwargs'):
                    msg.additional_kwargs = dict(getattr(msg, 'additional_kwargs', {}) or {})
                    msg.additional_kwargs["source_node"] = "tools"
                print(f"[agent] tools_node: tagged tool message with source_node=tools")
            except Exception as e:
                print(f"[agent] tools_node: failed to add source_node metadata: {repr(e)}")
    
    # Replace the last message with the sanitized one and append the tool result
    new_messages = state.messages[:-1] + [sanitized_message] + result["messages"]
    
    return {"messages": new_messages}


# Define a new graph

builder = StateGraph(State, input_schema=InputState, context_schema=Context)

# Define the two nodes we will cycle between
builder.add_node(call_model)
builder.add_node("tools", tools_node)
builder.add_node("finalize", finalize_answer)

# Set the entrypoint as `call_model`
# This means that this node is the first one called
builder.add_edge("__start__", "call_model")


def route_model_output(state: State) -> Literal["finalize", "tools"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("finalize" or "tools").
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )

    has_tool_calls = bool(last_message.tool_calls)
    print(f"[agent] route_model_output: is_last_step={state.is_last_step}, remaining_steps={state.remaining_steps}, has_tool_calls={has_tool_calls}")

    # Force finalize if we're close to recursion limit
    if state.remaining_steps <= 2:
        print("[agent] route_model_output: near recursion limit, routing to finalize")
        return "finalize"

    # If there is no tool call, then we go to finalize for confidence check
    if not has_tool_calls:
        print("[agent] route_model_output: routing to finalize")
        return "finalize"
    # Otherwise we execute the requested actions
    print("[agent] route_model_output: routing to tools")
    return "tools"


# Add a conditional edge to determine the next step after `call_model`
builder.add_conditional_edges(
    "call_model",
    # After call_model finishes running, the next node(s) are scheduled
    # based on the output from route_model_output
    route_model_output,
)

# Add a normal edge from `tools` to `call_model`
# This creates a cycle: after using tools, we always return to the model
builder.add_edge("tools", "call_model")

# Finalize ends the run
builder.add_edge("finalize", "__end__")

# Compile the builder into an executable graph
graph = builder.compile(name="ReAct Agent").with_config(recursion_limit=30)
