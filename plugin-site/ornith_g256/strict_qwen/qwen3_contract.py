# SPDX-License-Identifier: Apache-2.0
"""Strict finalization for the installed Qwen3 XML grammar only.

No command repair and no changes to partial argument deltas. Raising before the
final parser return lets the existing chat serving generator emit its SSE error.
"""
from __future__ import annotations

import json
import re

from jsonschema import ValidationError
from jsonschema.validators import validator_for

from vllm.parser.engine.parser_engine_config import ParserState
from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine


class Qwen3ToolContractError(ValueError):
    pass


class Qwen3ContractEngine(StreamingParserEngine):
    """Track native framing using the installed lexer/token-ID scanner.

This does not tokenize text twice or treat ordinary HTML as XML tool syntax.
Bare-function fallback remains allowed; an explicit tool wrapper must close.
"""

    def reset(self, initial_state=None):
        super().reset(initial_state=initial_state)
        self.contract_error = None
        self.contract_wrapper = False
        self.contract_function = False
        self.contract_parameter = False
        self.contract_implicit_reasoning_end = False
        self.contract_tool_started = False
        self.contract_late_think_end = False

    def _invalid(self, reason):
        if self.contract_error is None:
            self.contract_error = reason

    def _on_terminal(self, terminal, value, token_count=0):
        # The reasoning adapter observes the implicit transition; the separate
        # tool adapter observes a later explicit closer. Keep both observations
        # even in reasoning-only passes, then reconcile them at request finish.
        if terminal == "TOOL_START" and self.state in (
            ParserState.REASONING, ParserState.CONTENT, ParserState.TOOL_BETWEEN,
        ):
            self.contract_tool_started = True
            if self.state == ParserState.REASONING:
                self.contract_implicit_reasoning_end = True
        elif (
            terminal == "THINK_END" and self.contract_tool_started
            and self.state in (
                ParserState.REASONING, ParserState.CONTENT, ParserState.TOOL_BETWEEN,
            )
            and not (self.skip_tool_parsing and self._in_skipped_tool_span)
        ):
            self.contract_late_think_end = True
        if not self.skip_tool_parsing:
            state = self.state
            if terminal == "TOOL_START" and state in (
                ParserState.CONTENT, ParserState.REASONING,
                ParserState.TOOL_BETWEEN, ParserState.TOOL_PREAMBLE,
            ):
                if self.contract_wrapper or self.contract_function:
                    self._invalid("nested or unclosed tool wrapper")
                self.contract_wrapper = True
            elif terminal == "FUNC_PREFIX" and state in (
                ParserState.CONTENT, ParserState.TOOL_PREAMBLE,
                ParserState.TOOL_BETWEEN,
            ):
                if self.contract_function:
                    self._invalid("nested function")
                self.contract_function = True
            elif terminal == "FUNC_END" and self.contract_function:
                if state != ParserState.TOOL_ARGS:
                    self._invalid("incomplete function name")
                if self.contract_parameter:
                    self._invalid("function closes inside a parameter")
                self.contract_function = False
            elif terminal == "TOOL_END" and self.contract_wrapper:
                if self.contract_function or self.contract_parameter:
                    self._invalid("tool wrapper closes before its function")
                self.contract_wrapper = False
            elif terminal == "PARAM_START" and state == ParserState.TOOL_ARGS:
                if self.contract_parameter:
                    self._invalid("nested parameter or delimiter collision")
                self.contract_parameter = True
            elif terminal == "PARAM_END" and state == ParserState.TOOL_ARGS:
                if not self.contract_parameter:
                    self._invalid("parameter closer without opener")
                self.contract_parameter = False
            elif state == ParserState.TOOL_ARGS and terminal in (
                "TOOL_START", "TOOL_END", "FUNC_PREFIX",
            ):
                self._invalid("reserved tool delimiter inside a parameter")
        return super()._on_terminal(terminal, value, token_count)

    def validate_contract_frames(self):
        if self.contract_implicit_reasoning_end and self.contract_late_think_end:
            raise Qwen3ToolContractError(
                "ambiguous tool call inside reasoning before a later </think>"
            )
        if self.contract_error:
            raise Qwen3ToolContractError(self.contract_error)
        if self.contract_wrapper or self.contract_function or self.contract_parameter:
            raise Qwen3ToolContractError("incomplete Qwen3 tool framing at end of output")


_OPEN_PARAM = re.compile(r"<\s*parameter\s*=\s*([^>]*)>")
_PARAM_MARKER = re.compile(r"<\s*(/\s*parameter\s*>|parameter\s*=)")


def validate_parameter_frames(raw):
    """Require complete, unique parameters and no discarded non-frame text."""
    pos = 0
    names = set()
    while raw[pos:].strip():
        pos += len(raw[pos:]) - len(raw[pos:].lstrip())
        match = _OPEN_PARAM.match(raw, pos)
        if match is None or not match[1]:
            raise Qwen3ToolContractError("invalid parameter framing")
        name = match[1]
        if name in names:
            raise Qwen3ToolContractError("duplicate parameter name")
        names.add(name)
        end = _PARAM_MARKER.search(raw, match.end())
        if end is None or not end[1].lstrip().startswith("/"):
            raise Qwen3ToolContractError("incomplete or nested parameter")
        pos = end.end()


def _strict_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise Qwen3ToolContractError("duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(_value):
        raise Qwen3ToolContractError("non-finite JSON number")

    try:
        obj = json.loads(text, object_pairs_hook=unique, parse_constant=invalid_constant)
    except (json.JSONDecodeError, TypeError):
        raise Qwen3ToolContractError("incomplete or invalid final tool JSON") from None
    if not isinstance(obj, dict):
        raise Qwen3ToolContractError("tool arguments must be a JSON object")
    return obj


def _reject_external_refs(node):
    # Validation must not fetch schemas over the network during generation.
    if isinstance(node, dict):
        if "$ref" in node and not node["$ref"].startswith("#"):
            raise Qwen3ToolContractError("external tool schema references unsupported")
        for value in node.values():
            _reject_external_refs(value)
    elif isinstance(node, list):
        for value in node:
            _reject_external_refs(value)


def validate_qwen3_slots(parser, converter):
    tool_map = {}
    strict_tools = set()
    for tool in parser._tools or []:
        obj = tool.model_dump() if hasattr(tool, "model_dump") else tool
        if obj.get("type") == "function":
            fn = obj.get("function", obj)
            tool_map[fn["name"]] = fn.get("parameters") or {"type": "object"}
            if fn.get("strict") is True:
                strict_tools.add(fn["name"])
    for slot in parser._tool_slots:
        if slot.name not in tool_map:
            raise Qwen3ToolContractError("unknown or incomplete tool name")
        validate_parameter_frames(slot.args)
        final_json = parser._fix_arg_types(converter(slot.args, False), slot.name)
        final_obj = _strict_json(final_json)
        streamed_obj = _strict_json(slot.streamed_json)
        if final_obj != streamed_obj or final_json != slot.streamed_json:
            raise Qwen3ToolContractError("streamed and finalized tool arguments disagree")
        # Non-strict calls may fail the tool schema. Return the complete call
        # so the agent executor can report validation feedback to the model.
        # Framing and lossless JSON checks above apply to every call.
        if slot.name not in strict_tools:
            continue
        schema = tool_map[slot.name]
        _reject_external_refs(schema)
        try:
            validator_for(schema)(schema).validate(final_obj)
        except ValidationError:
            # Do not include generated command/code content in the error message.
            raise Qwen3ToolContractError("final tool arguments fail their schema") from None
