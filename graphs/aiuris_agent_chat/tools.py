"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.

USAGE in any agent:
- Import base tools only:
    from aiuris_agent_chat.tools import TOOLS
- Import MCP tools (shared cached loader):
    from aiuris_agent_chat.tools import get_mcp_tools
    mcp_tools = await get_mcp_tools()
- Combine base + MCP:
    from aiuris_agent_chat.tools import get_all_tools
    tools = await get_all_tools()
"""

from typing import Any, Callable, List, Optional, cast

from langchain_tavily import TavilySearch  # type: ignore[import-not-found]
from langgraph.runtime import get_runtime
from langchain_core.tools import tool

from aiuris_agent_chat.context import Context


@tool(parse_docstring=True)
async def web_search(query: str) -> Optional[dict[str, Any]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    
    Args:
        query: Natural language search query
    """
    runtime = get_runtime(Context)
    wrapped = TavilySearch(max_results=runtime.context.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))


# Reusable inline MCP loader with simple caching
from langchain_mcp_adapters.client import MultiServerMCPClient  # type: ignore
import asyncio
import os

_MCP_TOOLS_CACHE: List[Callable[..., Any]] = []
_MCP_LOAD_LOCK = asyncio.Lock()


async def get_mcp_tools() -> List[Callable[..., Any]]:
    """Load MCP tools once per process.

    Edit the servers dict below to match your environment. Absolute paths are
    recommended (e.g., inside Docker: /app/mcp-server/build/index.js).
    """
    global _MCP_TOOLS_CACHE
    if _MCP_TOOLS_CACHE:
        return _MCP_TOOLS_CACHE

    async with _MCP_LOAD_LOCK:
        if _MCP_TOOLS_CACHE:
            return _MCP_TOOLS_CACHE

        try:
            mcp_db_url = os.getenv("MCP_DATABASE_URL", "postgresql://postgres:postgres@db:5432/aiuris")
            servers = {
                "laws": {
                    "command": "node",
                    "args": [
                        "/app/mcp-server/build/index.js",
                        "--mcp-toolsets",
                        "laws",
                        "--database-url",
                        mcp_db_url,
                        "--collection-name",
                        "law_collection_langchain",
                    ],
                    "transport": "stdio",
                },
            }

            client = MultiServerMCPClient(servers)
            _MCP_TOOLS_CACHE = await client.get_tools()
        except Exception as e:
            print("[agent] Warning: MCP tools failed to load:", repr(e))
            _MCP_TOOLS_CACHE = []

        return _MCP_TOOLS_CACHE


async def get_all_tools() -> List[Callable[..., Any]]:
    """Combine base tools with MCP tools."""
    mcp = await get_mcp_tools()
    return [*TOOLS, *mcp]


# -----------------------------
# RAG: PGVector retriever tools (3 collections)
# -----------------------------
from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
try:
    from langchain_postgres import PGVector  # type: ignore
except Exception:  # pragma: no cover
    from langchain_postgres.vectorstores import PGVector  # type: ignore

from langchain_core.documents import Document  # type: ignore

# If your documents were embedded with "passage:" prefix, set this to True to prefix queries with "query:".
E5_ADD_QUERY_PREFIX: bool = False

# Hardcoded RAG configuration (simple/testing)
_RAG_MODEL_NAME = "intfloat/multilingual-e5-small"
_RAG_PG_CONN = os.getenv("RAG_DATABASE_URL", "postgresql+psycopg://postgres:postgres@db:5432/aiuris")
_COLLECTION_LAWS = "law_collection_langchain"
_COLLECTION_JUDICIAL = "judicial_practice_collection_langchain"
# _COLLECTION_COURT_NOTICE = "court_notice_collection_langchain"

# Caches
_RAG_EMBEDDINGS = None
_VS_CACHE: dict[str, PGVector] = {}
_RAG_INIT_LOCK = asyncio.Lock()


def _maybe_prefix_query(query: str) -> str:
    return f"query: {query}" if E5_ADD_QUERY_PREFIX and not query.strip().lower().startswith("query:") else query


async def _get_vector_store(collection_name: str) -> PGVector:
    global _RAG_EMBEDDINGS
    if collection_name in _VS_CACHE:
        return _VS_CACHE[collection_name]

    async with _RAG_INIT_LOCK:
        if _RAG_EMBEDDINGS is None:
            _RAG_EMBEDDINGS = HuggingFaceEmbeddings(
                model_name=_RAG_MODEL_NAME,
                encode_kwargs={"normalize_embeddings": True},
            )
        if collection_name in _VS_CACHE:
            return _VS_CACHE[collection_name]

        # Prefer existing index, fallback to constructor
        try:
            loop = asyncio.get_running_loop()
            vs = await loop.run_in_executor(
                None,
                lambda: PGVector.from_existing_index(
                    embeddings=_RAG_EMBEDDINGS,
                    collection_name=collection_name,
                    connection=_RAG_PG_CONN,
                    use_jsonb=True,
                ),
            )
        except Exception:
            vs = PGVector(
                embeddings=_RAG_EMBEDDINGS,
                collection_name=collection_name,
                connection=_RAG_PG_CONN,
                use_jsonb=True,
            )

        _VS_CACHE[collection_name] = vs
        return vs


async def _retrieve_from_collection(collection_name: str, query: str, k: int) -> List[dict[str, Any]]:
    effective_query = _maybe_prefix_query(query)
    vs = await _get_vector_store(collection_name)

    # Always use similarity (cosine with normalized embeddings)
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": k})

    # Prefer async, fallback to sync in thread
    docs: List[Document]
    try:
        docs = await retriever.ainvoke(effective_query)  # type: ignore[attr-defined]
    except Exception:
        loop = asyncio.get_running_loop()
        docs = await loop.run_in_executor(None, lambda: retriever.invoke(effective_query))

    return [
        {"content": d.page_content, "metadata": dict(d.metadata or {})}
        for d in docs
    ]


@tool(parse_docstring=True)
async def laws_rag_search(query: str, k: int = 5) -> List[dict[str, Any]]:
    """Search and retrieve relevant sections from Croatian laws.

    Use for statutes, articles, definitions, obligations, procedures, and legal provisions.
    
    Args:
        query: Legal query or keywords to search for
        k: Number of results to return (default 5)
    """
    return await _retrieve_from_collection(_COLLECTION_LAWS, query, k)


@tool(parse_docstring=True)
async def judicial_practice_rag_search(query: str, k: int = 5) -> List[dict[str, Any]]:
    """Search and retrieve relevant judicial practice (case law).

    Use for precedents, court reasoning, judgments, and interpretation of laws.
    
    Args:
        query: Legal query or keywords to search for
        k: Number of results to return (default 5)
    """
    return await _retrieve_from_collection(_COLLECTION_JUDICIAL, query, k)


# async def court_notice_rag_search(query: str, k: int = 5) -> List[dict[str, Any]]:
#     """Search and retrieve relevant court notices and announcements.
# 
#     Use for court schedules, notices, administrative communications, and announcements.
#     """
#     return await _retrieve_from_collection(_COLLECTION_COURT_NOTICE, query, k)


# Base tools available to all agents
TOOLS: List[Callable[..., Any]] = [
    web_search,
    laws_rag_search,
    judicial_practice_rag_search,
    # court_notice_rag_search,
]
