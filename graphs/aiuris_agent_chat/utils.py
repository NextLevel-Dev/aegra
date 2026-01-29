"""Utility & helper functions."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from typing import Any, List, Tuple, Dict


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(
    fully_specified_name: str,
    context: Any | None = None,
    include_thoughts: bool | None = None,
    thinking_budget: int | None = None,
    reasoning_effort: str | None = None
) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
        context (Any | None): Optional Context with reasoning config.
        include_thoughts (bool | None): Override include_thoughts setting.
        thinking_budget (int | None): Override thinking_budget setting.
        reasoning_effort (str | None): Override reasoning_effort setting.

    Note:
        For Vertex AI models, use tool_config parameter in bind_tools() instead of
        parallel_tool_calls=False to control tool calling behavior:

        tool_config = {
            "function_calling_config": {
                "mode": "AUTO",  # or "ANY" to force tool usage
                "allowed_function_names": ["tool1", "tool2"]  # optional subset
            }
        }
        model.bind_tools(tools, tool_config=tool_config)
    """
    provider, model = fully_specified_name.split("/", maxsplit=1)
    if provider == "google_vertexai":
        try:
            # Local import to avoid hard dependency where not needed
            from langchain_google_vertexai import ChatVertexAI  # type: ignore

            # Use provided overrides or fall back to context values
            final_include_thoughts = include_thoughts if include_thoughts is not None else (
                True if context is None else getattr(context, "include_thoughts", True)
            )
            final_thinking_budget = thinking_budget if thinking_budget is not None else (
                1000 if context is None else getattr(context, "thinking_budget", 1000)
            )
            final_reasoning_effort = reasoning_effort if reasoning_effort is not None else (
                "high" if context is None else getattr(context, "reasoning_effort", "high")
            )

            return ChatVertexAI(
                model=model,
                include_thoughts=final_include_thoughts,
                thinking_budget=final_thinking_budget,
                model_kwargs={"reasoning_effort": final_reasoning_effort},
            )
        except Exception as e:
            print(
                "[agent] Warning: ChatVertexAI init failed, falling back to generic init_chat_model. Error:",
                repr(e),
            )
    return init_chat_model(model, model_provider=provider)


def extract_thinking_from_message(msg: BaseMessage) -> tuple[str, list[dict]]:
    """Extract provider-structured reasoning/thinking parts from a message.

    Returns a tuple of (thinking_text, raw_thinking_parts).
    """
    content = msg.content
    try:
        if isinstance(content, list):
            raw_parts: List[Dict] = [
                part
                for part in content
                if isinstance(part, dict)
                and str(part.get("type", "")).lower() in {"reasoning", "thought", "thoughts", "thinking"}
            ]
            texts: List[str] = []
            for part in raw_parts:
                if isinstance(part.get("reasoning"), str):
                    texts.append(part["reasoning"])
                elif isinstance(part.get("thinking"), str):
                    texts.append(part["thinking"])
                elif isinstance(part.get("text"), str):
                    texts.append(part["text"])
            thinking_text = "\n".join(t.strip() for t in texts if t and t.strip())
            return thinking_text, raw_parts
    except Exception as e:
        print("[agent] extract_thinking_from_message error:", repr(e))
    return "", []
