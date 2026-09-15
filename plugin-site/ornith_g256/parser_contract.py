"""Install strict Qwen XML adapters in this bundle's processes only."""
from functools import wraps
from http import HTTPStatus

from vllm.parser.engine.adapters import make_adapters
from .strict_qwen.qwen3 import Qwen3Parser
from .strict_qwen.qwen3_contract import Qwen3ToolContractError

_ReasoningAdapter, _ToolAdapter = make_adapters(Qwen3Parser)


class CiruQwen3ReasoningParser(_ReasoningAdapter):
    pass


class CiruQwen3ToolParser(_ToolAdapter):
    structural_tag_model = "qwen_3_coder"

    def get_structural_tag(self, request, *, reasoning=False):
        from .native_tool_schema import native_tool_tag
        return native_tool_tag(
            self, request, reasoning=reasoning,
            fallback=super().get_structural_tag,
        )


_installed = False


def _validate_reasoning_tool_boundary(parser):
    # Separate adapters are per request. Never infer this from text matching:
    # the native lexer/token-ID scanner supplies the terminal observations.
    reasoning = getattr(parser, "_reasoning_parser", None)
    tool = getattr(parser, "_tool_parser", None)
    if not isinstance(reasoning, CiruQwen3ReasoningParser) or not isinstance(tool, CiruQwen3ToolParser):
        return
    r = reasoning._parser_engine._engine
    t = tool._parser_engine._engine
    if r.contract_implicit_reasoning_end and (r.contract_late_think_end or t.contract_late_think_end):
        raise Qwen3ToolContractError(
            "ambiguous tool call inside reasoning before a later </think>"
        )


def install():
    global _installed
    if _installed:
        return
    from vllm.reasoning import ReasoningParserManager
    from vllm.tool_parsers import ToolParserManager
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.parser.abstract_parser import DelegatingParser
    from vllm.logger import init_logger

    # Immediate registration replaces cached class lookups too. Decorator-style
    # registration only updates a lazy map and can leave a previously loaded
    # class active. Other tool/reasoning parser aliases keep their original class.
    ToolParserManager.register_module(
        name="qwen3_xml", force=True, module=CiruQwen3ToolParser
    )
    ReasoningParserManager.register_module(
        name="qwen3", force=True, module=CiruQwen3ReasoningParser
    )

    original_delta = DelegatingParser.parse_delta
    if not getattr(original_delta, "_ciru_reasoning_tool_boundary", False):
        @wraps(original_delta)
        def delta_with_boundary_check(self, *args, **kwargs):
            delta = original_delta(self, *args, **kwargs)
            if kwargs.get("finished"):
                _validate_reasoning_tool_boundary(self)
            return delta
        delta_with_boundary_check._ciru_reasoning_tool_boundary = True
        DelegatingParser.parse_delta = delta_with_boundary_check

    original_parse = DelegatingParser.parse
    if not getattr(original_parse, "_ciru_reasoning_tool_boundary", False):
        @wraps(original_parse)
        def parse_with_boundary_check(self, *args, **kwargs):
            result = original_parse(self, *args, **kwargs)
            _validate_reasoning_tool_boundary(self)
            return result
        parse_with_boundary_check._ciru_reasoning_tool_boundary = True
        DelegatingParser.parse = parse_with_boundary_check

    original = OpenAIServingChat.chat_completion_full_generator
    if not getattr(original, "_ciru_qwen_contract_errors", False):
        @wraps(original)
        async def full_with_contract_error(self, *args, **kwargs):
            try:
                return await original(self, *args, **kwargs)
            except Qwen3ToolContractError as exc:
                return self.create_error_response(
                    str(exc), err_type="ToolParserContractError",
                    status_code=HTTPStatus.BAD_REQUEST,
                )
        full_with_contract_error._ciru_qwen_contract_errors = True
        OpenAIServingChat.chat_completion_full_generator = full_with_contract_error

    _installed = True
    init_logger("vllm.ciru_qwen_contract").info(
        "CIRU_STRICT_QWEN3_TOOLS_LOADED parser=%s", Qwen3Parser.__module__
    )
