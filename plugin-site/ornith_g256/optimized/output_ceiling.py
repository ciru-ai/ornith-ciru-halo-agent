"""Let this profile's full-context output ceiling use the prompt's remainder."""
from dataclasses import replace


def install(context):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    original = ChatCompletionRequest.build_tok_params

    def build_tok_params(request, model_config):
        params = original(request, model_config)
        maximum = (request.max_completion_tokens if request.max_completion_tokens is not None
                   else request.max_tokens)
        if maximum == context == model_config.max_model_len:
            # Validate the whole prompt against context, then let the existing
            # get_max_tokens clamp the unchanged requested ceiling to the actual
            # rendered prompt's remainder. Reserving the full ceiling here would
            # leave zero input tokens and reject every nonempty client prompt.
            params = replace(params, max_output_tokens=0)
        return params

    ChatCompletionRequest.build_tok_params = build_tok_params
