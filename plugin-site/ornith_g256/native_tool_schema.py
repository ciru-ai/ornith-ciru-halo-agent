"""Native Qwen tool-schema defaults for Ciru's serving plugin."""
from openai.types.responses import FunctionTool
from vllm import envs
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag


def native_tool_tag(parser, request, *, reasoning, fallback):
    """Constrain default-auto calls without rewriting prompt-visible tools.

    An explicit strict=False remains an opt-out. Explicit output formats and
    vLLM's global enforcement switch retain their existing meaning. Required
    and named choices continue through the upstream implementation.
    """
    if request.tool_choice not in (None, "auto"):
        return fallback(request, reasoning=reasoning)
    if not request.tools or not envs.VLLM_ENFORCE_STRICT_TOOL_CALLING:
        return None
    if getattr(request, "structured_outputs", None) is not None:
        return None
    response_format = getattr(request, "response_format", None)
    if response_format is not None and response_format.type != "text":
        return None
    text = getattr(request, "text", None)
    if text is not None and text.format is not None and text.format.type != "text":
        return None

    compiler_tools = []
    for tool in request.tools:
        if isinstance(tool, ChatCompletionToolsParam) and tool.function.strict is None:
            function = tool.function.model_copy(update={"strict": True})
            tool = tool.model_copy(update={"function": function})
        elif isinstance(tool, FunctionTool) and tool.strict is None:
            tool = tool.model_copy(update={"strict": True})
        compiler_tools.append(tool)
    return get_model_structural_tag(
        model=parser.structural_tag_model,
        tools=compiler_tools,
        tool_choice=request.tool_choice or "auto",
        reasoning=reasoning,
    )
