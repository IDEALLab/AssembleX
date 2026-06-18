"""Shared OpenAI chat + image-payload helpers for the core LLM/VLM call sites.

Centralizes the three things that were copy-pasted at ~20 sites: client
construction, the chat-completion call, and token-budget accounting. The budget
*check* stays at the call site (`evaluation.tokens_exhausted(...)`) because each
caller has its own fallback; this records usage only after a successful call.

Note: core/tool_eval.py deliberately does NOT use these helpers. It is a frozen
copy of the data_gen code path, kept byte-identical for reproducibility.
"""

from openai import OpenAI

from core.models import encode_image


def image_part(path, detail=None):
    """Build an OpenAI chat `image_url` content part from a PNG file path."""
    image_url = {"url": f"data:image/png;base64,{encode_image(path)}"}
    if detail is not None:
        image_url["detail"] = detail
    return {"type": "image_url", "image_url": image_url}


def llm_chat(evaluation, api_key=None, *, parse=False, **params):
    """Run an OpenAI chat completion and record token usage on `evaluation`.

    Args:
        evaluation: the `Eval` instance (or None). When set, the response's
            `usage.total_tokens` is added to `evaluation.tokens_used`.
        api_key: forwarded to `OpenAI(api_key=...)`; None uses the default
            (env-var) client.
        parse: when True call `chat.completions.parse` (structured output),
            otherwise `chat.completions.create`.
        **params: forwarded verbatim to the call (model, messages,
            response_format, temperature, ...).

    The caller is responsible for the budget check (`tokens_exhausted`); this
    only accounts usage after the call returns.
    """
    client = OpenAI(api_key=api_key) if api_key is not None else OpenAI()
    method = client.chat.completions.parse if parse else client.chat.completions.create
    response = method(**params)
    usage = getattr(response, "usage", None)
    if evaluation is not None and usage is not None:
        evaluation.tokens_used += int(getattr(usage, "total_tokens", 0) or 0)
    return response
