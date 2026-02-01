"""Mock tools for the AIURIS Legal AI Agent.

This module provides mock implementations of legal research tools
using fishing law data for testing purposes.

In production, these would be replaced with real:
- RAG search against PGVector
- MCP tools for law/case retrieval
- Tavily or similar web search
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any

from langchain_core.tools import tool


# =============================================================================
# Mock Data: Croatian Fishing Law
# =============================================================================

MOCK_LAWS = {
    "fishing-law-2020": {
        "id": "fishing-law-2020",
        "full_title": "Law on Freshwater Fishing (NN 63/2019, 14/2020)",
        "short_title": "Freshwater Fishing Law",
        "articles": {
            "15": {
                "title": "Protected Fishing Zones",
                "content": """Article 15 - Protected Fishing Zones

(1) Protected fishing zones are designated water bodies or sections thereof 
where fishing is prohibited or restricted to protect fish populations during 
spawning seasons or in ecologically sensitive areas.

(2) The Ministry shall publish a list of protected zones annually, including:
   a) spawning grounds during breeding season (March 1 - June 30)
   b) nature reserves and national park waters
   c) fish breeding facilities and their surrounding waters (200m radius)

(3) Fishing in protected zones without special authorization is strictly prohibited."""
            },
            "42": {
                "title": "Penalties for Illegal Fishing",
                "content": """Article 42 - Penalties for Fishing Violations

(1) A fine of 500 to 2,000 EUR shall be imposed on any natural person who:
   a) fishes without a valid fishing license
   b) fishes in protected zones without authorization
   c) uses prohibited fishing methods or equipment

(2) A fine of 2,000 to 5,000 EUR shall be imposed on any natural person who:
   a) fishes during closed season for protected species
   b) catches fish below minimum size limits
   c) exceeds daily catch limits

(3) In addition to fines, the following may be imposed:
   a) confiscation of fishing equipment
   b) confiscation of catch
   c) suspension of fishing license for 1-3 years

(4) For repeated violations within 3 years, fines are doubled."""
            },
            "18": {
                "title": "Fishing Permits and Licenses",
                "content": """Article 18 - Fishing Permits and Licenses

(1) To fish in Croatian freshwaters, a person must possess:
   a) a valid fishing license issued by the Ministry
   b) a fishing permit for the specific water body
   c) proof of payment of fishing fees

(2) Fishing licenses are valid for one calendar year and must be renewed annually.

(3) Special permits are required for:
   a) night fishing
   b) fishing from boats
   c) fishing in designated trophy waters

(4) Foreign nationals may obtain temporary fishing permits valid for 7 or 30 days."""
            },
        }
    }
}

MOCK_RAG_CHUNKS = [
    {
        "chunk_id": "chunk-001",
        "law_id": "fishing-law-2020",
        "article": "15",
        "content": "Protected fishing zones include spawning grounds (March-June), nature reserves, and areas within 200m of fish breeding facilities. Fishing without special authorization is strictly prohibited in these zones.",
        "score": 0.92
    },
    {
        "chunk_id": "chunk-002", 
        "law_id": "fishing-law-2020",
        "article": "42",
        "content": "Fines for fishing in protected zones without authorization range from 500 to 2,000 EUR. Equipment and catch may be confiscated, and fishing license may be suspended for 1-3 years.",
        "score": 0.89
    },
    {
        "chunk_id": "chunk-003",
        "law_id": "fishing-law-2020", 
        "article": "42",
        "content": "Repeated fishing violations within 3 years result in doubled fines. Serious violations like fishing during closed season for protected species carry fines of 2,000 to 5,000 EUR.",
        "score": 0.85
    },
    {
        "chunk_id": "chunk-004",
        "law_id": "fishing-law-2020",
        "article": "18",
        "content": "All fishers must possess a valid fishing license, specific water body permit, and proof of fee payment. Foreign nationals can obtain temporary permits for 7 or 30 days.",
        "score": 0.78
    },
]

MOCK_WEB_RESULTS = [
    {
        "title": "Croatian Fishing Regulations 2024 - Ministry of Agriculture",
        "url": "https://mps.gov.hr/fishing-regulations-2024",
        "snippet": "Updated list of protected fishing zones for 2024. New restrictions apply to Lake Vrana and Neretva Delta during spawning season (March 1 - June 30)."
    },
    {
        "title": "Fishing Fines Increased in Croatia - Vecernji List",
        "url": "https://vecernji.hr/fishing-fines-2024",
        "snippet": "Following EU harmonization, fines for illegal fishing have been increased. Maximum penalties now reach 5,000 EUR for serious violations."
    },
    {
        "title": "Protected Fish Species in Croatian Waters",
        "url": "https://haop.hr/protected-fish-species",
        "snippet": "List of protected fish species including endemic Adriatic sturgeon, softmouth trout, and marble trout. Catching these species carries severe penalties."
    }
]


# =============================================================================
# Tool Result Types
# =============================================================================

@dataclass
class RAGSearchResult:
    """Result from RAG semantic search."""
    query: str
    chunks: list[dict[str, Any]]
    total_results: int


@dataclass
class LawDocument:
    """A retrieved law document."""
    law_id: str
    full_title: str
    short_title: str
    articles: dict[str, dict[str, str]]


@dataclass
class ArticleContent:
    """Content of a specific article."""
    law_id: str
    article_num: str
    title: str
    content: str


@dataclass
class WebSearchResult:
    """Result from web search."""
    query: str
    results: list[dict[str, str]]
    total_results: int


# =============================================================================
# Mock Tool Implementations (as LangChain Tools)
# =============================================================================

def _serialize_result(result: Any) -> str:
    """Serialize a dataclass result to JSON string for tool output."""
    if hasattr(result, "__dataclass_fields__"):
        return json.dumps(asdict(result), indent=2, default=str)
    elif result is None:
        return json.dumps({"error": "Not found"})
    else:
        return json.dumps(result, indent=2, default=str)


@tool
def rag_search(query: str) -> str:
    """Perform semantic search across legal documents to find relevant laws, regulations, and legal information.
    
    Use this tool for initial exploration when you don't know specific law IDs or article numbers.
    Returns ranked chunks with metadata including law IDs and article numbers for follow-up queries.
    
    Args:
        query: Search query describing what legal information you're looking for
    """
    query_lower = query.lower()
    
    # Simple relevance scoring based on keyword presence
    relevant_chunks = []
    
    keywords_to_chunks = {
        "illegal": ["chunk-001", "chunk-002", "chunk-003"],
        "protected": ["chunk-001", "chunk-002"],
        "zone": ["chunk-001", "chunk-002"],
        "penalty": ["chunk-002", "chunk-003"],
        "fine": ["chunk-002", "chunk-003"],
        "fish": ["chunk-001", "chunk-002", "chunk-003", "chunk-004"],
        "license": ["chunk-004"],
        "permit": ["chunk-004"],
    }
    
    matched_chunk_ids = set()
    for keyword, chunk_ids in keywords_to_chunks.items():
        if keyword in query_lower:
            matched_chunk_ids.update(chunk_ids)
    
    # If no specific matches, return all chunks
    if not matched_chunk_ids:
        matched_chunk_ids = {"chunk-001", "chunk-002", "chunk-003", "chunk-004"}
    
    for chunk in MOCK_RAG_CHUNKS:
        if chunk["chunk_id"] in matched_chunk_ids:
            relevant_chunks.append(chunk)
    
    # Sort by score
    relevant_chunks.sort(key=lambda x: x["score"], reverse=True)
    
    result = RAGSearchResult(
        query=query,
        chunks=relevant_chunks[:5],
        total_results=len(relevant_chunks)
    )
    return _serialize_result(result)


@tool
def fetch_law_by_id(law_id: str) -> str:
    """Retrieve a complete law document by its ID.
    
    Use this when you know the specific law ID from a previous search.
    Returns the full law document with all articles.
    
    Args:
        law_id: The law identifier (e.g., "fishing-law-2020")
    """
    if law_id in MOCK_LAWS:
        law = MOCK_LAWS[law_id]
        result = LawDocument(
            law_id=law["id"],
            full_title=law["full_title"],
            short_title=law["short_title"],
            articles=law["articles"]
        )
        return _serialize_result(result)
    return json.dumps({"error": f"Law not found: {law_id}"})


@tool
def fetch_article(law_id: str, article_num: str) -> str:
    """Retrieve a specific article from a law.
    
    Use this to get the full text of a specific article when you know both the law ID and article number.
    
    Args:
        law_id: The law identifier (e.g., "fishing-law-2020")
        article_num: The article number (e.g., "15", "42")
    """
    if law_id in MOCK_LAWS:
        law = MOCK_LAWS[law_id]
        if article_num in law["articles"]:
            article = law["articles"][article_num]
            result = ArticleContent(
                law_id=law_id,
                article_num=article_num,
                title=article["title"],
                content=article["content"]
            )
            return _serialize_result(result)
    return json.dumps({"error": f"Article not found: {law_id} Art. {article_num}"})


@tool
def web_search(query: str) -> str:
    """Search the web for current information, news, and updates.
    
    Use this to supplement legal research with current events, recent changes,
    or practical information not found in the legal database.
    
    Args:
        query: Search query for web search
    """
    query_lower = query.lower()
    
    # Simple filtering based on keywords
    relevant_results = []
    
    for result in MOCK_WEB_RESULTS:
        title_lower = result["title"].lower()
        snippet_lower = result["snippet"].lower()
        
        query_words = query_lower.split()
        matches = sum(1 for word in query_words 
                     if word in title_lower or word in snippet_lower)
        
        if matches > 0:
            relevant_results.append({**result, "_score": matches})
    
    # Sort by relevance and remove score
    relevant_results.sort(key=lambda x: x.get("_score", 0), reverse=True)
    clean_results = [{k: v for k, v in r.items() if k != "_score"} 
                     for r in relevant_results]
    
    # If no matches, return all results
    if not clean_results:
        clean_results = MOCK_WEB_RESULTS
    
    result = WebSearchResult(
        query=query,
        results=clean_results[:5],
        total_results=len(clean_results)
    )
    return _serialize_result(result)


# =============================================================================
# Tool List for binding to models
# =============================================================================

# List of all tools to bind to the model
TOOLS = [rag_search, fetch_law_by_id, fetch_article, web_search]

# Map tool names to functions for execution
TOOL_MAP = {
    "rag_search": rag_search,
    "fetch_law_by_id": fetch_law_by_id,
    "fetch_article": fetch_article,
    "web_search": web_search,
}


def execute_tool_call(tool_call: dict) -> str:
    """Execute a tool call from an AIMessage.
    
    Args:
        tool_call: Tool call dict with 'name', 'args', 'id'
        
    Returns:
        Tool result as string
    """
    tool_name = tool_call.get("name")
    tool_args = tool_call.get("args", {})
    
    if tool_name not in TOOL_MAP:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    
    tool_func = TOOL_MAP[tool_name]
    return tool_func.invoke(tool_args)
