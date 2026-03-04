"""LLM chat and analysis routes — NVIDIA NIM with agentic tool calling."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    max_tokens: int = 8192
    temperature: float = 1.0
    enable_tools: bool = True


class ChatResponse(BaseModel):
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    provider: str = ""
    tool_calls: List[Dict[str, Any]] = []


class AnalysisRequest(BaseModel):
    analysis_type: str = Field("general", pattern="^(general|asset|climate|report)$")
    question: str = ""
    context: Optional[Dict[str, Any]] = None
    report_type: str = "summary"


class AnalysisResponse(BaseModel):
    text: str
    analysis_type: str


class LLMStatusResponse(BaseModel):
    available: bool
    provider: str
    model: str


# Lazy-loaded singletons
_llm_client = None
_client_checked = False
_tool_handlers = None


def _get_llm_client():
    global _llm_client, _client_checked
    if _client_checked:
        return _llm_client
    _client_checked = True
    try:
        from src.llm.cloud_client import create_cloud_client
        client = create_cloud_client()
        if client:
            _llm_client = client
            logger.info(f"Using NVIDIA NIM: {client.model}")
            return _llm_client
    except Exception as e:
        logger.warning(f"NVIDIA NIM not available: {e}")
    return None


def _get_tool_handlers():
    global _tool_handlers
    if _tool_handlers is not None:
        return _tool_handlers
    try:
        from api.main import handlers
        if handlers:
            _tool_handlers = handlers
            return _tool_handlers
    except Exception:
        pass
    from src.mcp.tools import ToolHandlers
    _tool_handlers = ToolHandlers()
    return _tool_handlers


@router.get("/llm/status", response_model=LLMStatusResponse)
async def llm_status():
    client = _get_llm_client()
    if client is None:
        return LLMStatusResponse(available=False, provider="none", model="none")
    return LLMStatusResponse(
        available=True,
        provider=getattr(client, "provider", "nvidia_nim"),
        model=getattr(client, "model", "unknown"),
    )


@router.post("/llm/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    client = _get_llm_client()
    if client is None:
        raise HTTPException(503, "No LLM available. Set NVIDIA_API_KEY in .env.")

    from src.llm.client import ChatMessage
    messages = [ChatMessage(role=m["role"], content=m["content"]) for m in req.messages]

    try:
        if req.enable_tools:
            from src.llm.cloud_client import CloudLLMClient
            from src.mcp.tools import get_openai_tools
            if isinstance(client, CloudLLMClient):
                tools = get_openai_tools()
                handlers = _get_tool_handlers()
                result = client.agentic_chat(
                    messages, handlers, tools,
                    max_tokens=req.max_tokens, temperature=req.temperature,
                )
                return ChatResponse(
                    text=result.text,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    model=getattr(client, "model", "unknown"),
                    provider=getattr(client, "provider", "nvidia_nim"),
                    tool_calls=result.tool_calls,
                )

        # Plain chat fallback
        result = client.chat(messages, max_tokens=req.max_tokens, temperature=req.temperature)
        return ChatResponse(
            text=result.text,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            model=getattr(client, "model", "unknown"),
            provider=getattr(client, "provider", "nvidia_nim"),
        )
    except Exception as e:
        raise HTTPException(500, f"LLM error: {str(e)}")


@router.post("/llm/analyze", response_model=AnalysisResponse)
async def analyze(req: AnalysisRequest):
    client = _get_llm_client()
    if client is None:
        raise HTTPException(503, "No LLM available. Set NVIDIA_API_KEY in .env.")
    try:
        if req.analysis_type == "asset" and req.context:
            text = client.analyze_asset(req.context)
        elif req.analysis_type == "climate" and req.context:
            text = client.analyze_climate_risk(req.context)
        elif req.analysis_type == "report" and req.context:
            text = client.generate_report(req.context, req.report_type)
        else:
            text = client.query(req.question, req.context)
        return AnalysisResponse(text=text, analysis_type=req.analysis_type)
    except Exception as e:
        raise HTTPException(500, f"Analysis error: {str(e)}")
