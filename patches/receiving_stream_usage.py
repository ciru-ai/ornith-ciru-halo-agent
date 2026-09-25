"""Reviewed Ciru baseline -> qualified framing/usage source; exact hash guards."""
import hashlib
from stream_usage import patch_source

RECEIVING_SHA256='9982953285e9df469032a82fffa4095d0e9d86278bede6e2b91d03d02373d182'
FINISH_GUARDS=[
    ('                        if tools_streamed[i] and not tool_choice_function_name:\n',
     '                        if (\n                            tools_streamed[i]\n                            and not tool_choice_function_name\n                            and output.finish_reason == "stop"\n                        ):\n'),
    ('            is_finish_reason_tool_calls = auto_tools_called or (\n',
     '            is_finish_reason_tool_calls = (\n                auto_tools_called and output.finish_reason == "stop"\n            ) or (\n'),
]


def patch_receiving_source(source):
    if hashlib.sha256(source.encode()).hexdigest()!=RECEIVING_SHA256:
        raise RuntimeError('Receiving chat source changed; review its delta before integration')
    for old,new in FINISH_GUARDS:
        if source.count(old)!=1:raise RuntimeError('Receiving finish-reason guard anchor changed')
        source=source.replace(old,new)
    # These two guards reproduce the owner's qualified baseline byte-for-byte.
    # A length-limited output must not be reported as a completed tool call.
    return patch_source(source)
