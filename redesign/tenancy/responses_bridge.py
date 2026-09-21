"""The stateless Responses subset served by K3's Chat-only deployment.

Validate before LiteLLM 1.102.0 can drop native features or look up conversation
state, then again after Router defaults. Conversion itself belongs to LiteLLM.
Only request envelopes and protocol fields are inspected: schemas and function
arguments/results are caller data, not another envelope of server controls.
"""

from __future__ import annotations

from typing import NoReturn


class BridgeRequestError(ValueError):
    """A caller error that the callback exposes as HTTP 400."""


_UNSUPPORTED_FIELDS = frozenset({
    "previous_response_id", "conversation", "conversation_id", "session_id",
    "thread_id", "assistant_id", "attachments", "file_ids", "files",
    "prompt", "prompt_id", "context_management", "max_tool_calls",
    "web_search_options", "search_parameters", "retrieval", "vector_store_ids",
    "mcp_servers", "messages", "functions", "function_call", "response_format",
})
# OpenAI merges extra_body AFTER the SDK deployment hook, with extra_body
# winning duplicate keys. Keep the policy's messages, tools, limits and target
# authoritative. Provider settings such as top_k and chat_template_kwargs stay
# supported; cache controls are scrubbed by the existing cache policy.
_EXTRA_RESERVED = _UNSUPPORTED_FIELDS | frozenset({
    "input", "instructions", "tools", "tool_choice", "parallel_tool_calls",
    "max_tokens", "max_completion_tokens", "max_output_tokens", "text",
    "model", "stream", "stream_options", "store", "background", "include",
    "truncation", "extra_body", "k3_batch", "use_chat_completions_api",
    "api_base", "base_url", "api_key", "custom_llm_provider", "client",
    "litellm_logging_obj",
})
_FUNCTION_FIELDS = frozenset({"name", "description", "parameters", "strict"})
_ROLES = frozenset({"system", "developer", "user", "assistant"})


def _error(field: str, reason: str) -> NoReturn:
    raise BridgeRequestError(f"{field}: {reason}")


def _limit(value: object) -> None:
    if value is not None and (type(value) is not int or value <= 0):
        _error("max_output_tokens", "must be a positive integer")


def validate_extra_body(data: dict) -> None:
    extra = data.get("extra_body")
    if extra is None:
        return
    if not isinstance(extra, dict):
        _error("extra_body", "must be an object")
    reserved = _EXTRA_RESERVED.intersection(extra)
    if reserved:
        _error(f"extra_body.{sorted(reserved)[0]}", "must be supplied at the request root")


def prepare_chat_request(data: dict) -> None:
    """Portal alias: max_completion_tokens > max_output_tokens > max_tokens.

    The modern Chat alias still wins in TenancyPolicy. Remove the Responses
    spelling before hashing/sending Chat, including when its value is null.
    """
    validate_extra_body(data)
    output = data.get("max_output_tokens")
    _limit(output)
    data.pop("max_output_tokens", None)
    if output is not None:
        data["max_tokens"] = output


def _function(tool: object, path: str) -> dict:
    if not isinstance(tool, dict) or tool.get("type") != "function":
        _error(path, "only function tools are supported by the stateless Chat bridge")
    # A deployment shared with Chat may declare Chat-shaped function defaults.
    # Flatten these at the outer Responses barrier, before LiteLLM conversion.
    definition = tool.get("function", tool)
    allowed = {"type", "function"} if "function" in tool else {"type"} | _FUNCTION_FIELDS
    if set(tool) - allowed or not isinstance(definition, dict):
        _error(path, "unsupported function tool fields")
    if "function" in tool and set(definition) - _FUNCTION_FIELDS:
        _error(path, "unsupported function tool fields")
    if not isinstance(definition.get("name"), str) or not definition["name"].strip():
        _error(f"{path}.name", "must be a nonempty string")
    if definition.get("parameters") is not None and not isinstance(definition["parameters"], dict):
        _error(f"{path}.parameters", "must be a JSON schema object")
    if definition.get("description") is not None and not isinstance(definition["description"], str):
        _error(f"{path}.description", "must be a string")
    if definition.get("strict") is not None and type(definition["strict"]) is not bool:
        _error(f"{path}.strict", "must be a boolean")
    return definition


def _content(content: object, path: str, *, tool_output: bool = False) -> None:
    if isinstance(content, str) or (tool_output and isinstance(content, dict)):
        return
    if not isinstance(content, list):
        _error(path, "must contain text or text/image content parts")
    for index, part in enumerate(content):
        field = f"{path}[{index}]"
        if not isinstance(part, dict):
            _error(field, "must be a content part object")
        kind = part.get("type")
        if kind in ("input_text", "output_text", "text"):
            if not isinstance(part.get("text"), str):
                _error(f"{field}.text", "must be a string")
        elif kind == "input_image" or (tool_output and kind == "image_url"):
            if part.get("file_id") is not None:
                _error(f"{field}.file_id", "stored image references are unsupported; supply image_url")
            url = part.get("image_url")
            if tool_output and isinstance(url, dict):
                url = url.get("url")
            if not isinstance(url, str) or not url.startswith(("https://", "http://", "data:image/")):
                _error(f"{field}.image_url", "must be an HTTP(S) URL or image data URI")
        else:
            _error(field, "only text and input_image content are supported")


def _input(value: object) -> None:
    if isinstance(value, str):
        return
    if not isinstance(value, list):
        _error("input", "must be a string or a list of inline messages/function calls")
    calls: set[str] = set()
    outputs: set[str] = set()
    for index, item in enumerate(value):
        path = f"input[{index}]"
        if not isinstance(item, dict):
            _error(path, "must be an input item object")
        kind = item.get("type", "message")
        if kind == "message":
            if not isinstance(item.get("role"), str) or item["role"] not in _ROLES:
                _error(f"{path}.role", "must be system, developer, user or assistant")
            if any(item.get(key) is not None for key in ("tool_calls", "tool_call_id", "function_call")):
                _error(path, "use inline function_call/function_call_output items for tool history")
            _content(item.get("content"), f"{path}.content")
        elif kind in ("function_call", "function_call_output"):
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id.strip():
                _error(f"{path}.call_id", "must be a nonempty string")
            if kind == "function_call":
                if call_id in calls:
                    _error(f"{path}.call_id", "duplicate function call in supplied history")
                if not isinstance(item.get("name"), str) or not item["name"].strip():
                    _error(f"{path}.name", "must be a nonempty string")
                if item.get("namespace"):
                    _error(f"{path}.namespace", "namespaced tools are unsupported")
                if not isinstance(item.get("arguments"), (str, dict)):
                    _error(f"{path}.arguments", "must be a JSON string or object")
                calls.add(call_id)
            else:
                # LiteLLM otherwise reconstructs calls from a process-global,
                # unscoped TOOL_CALLS_CACHE. Every result must be self-contained.
                if call_id not in calls or call_id in outputs:
                    _error(f"{path}.call_id", "must match an earlier, unanswered function_call in supplied input")
                _content(item.get("output"), f"{path}.output", tool_output=True)
                outputs.add(call_id)
        else:
            _error(path, "only inline messages, function_call and function_call_output are supported")


def _text(value: object) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or set(value) - {"format"}:
        _error("text", "only text.format is supported")
    fmt = value.get("format")
    if fmt is None:
        return
    if not isinstance(fmt, dict):
        _error("text.format", "must be a format object")
    kind = fmt.get("type")
    if kind == "json_schema":
        if set(fmt) - {"type", "name", "schema", "strict"}:
            _error("text.format", "supported JSON schema fields are type, name, schema and strict")
        if not isinstance(fmt.get("name"), str) or not fmt["name"].strip():
            _error("text.format.name", "must be a nonempty string")
        if not isinstance(fmt.get("schema"), dict):
            _error("text.format.schema", "must be a JSON schema object")
        if fmt.get("strict") is not None and type(fmt["strict"]) is not bool:
            _error("text.format.strict", "must be a boolean")
    elif kind not in ("text", "json_object") or set(fmt) - {"type"}:
        _error("text.format", "supported formats are text, json_object and json_schema")


def validate_responses_request(data: dict, *, deployment: bool = False) -> None:
    validate_extra_body(data)
    for field in sorted(_UNSUPPORTED_FIELDS):
        if data.get(field) is not None:
            _error(field, "unsupported by the stateless Responses bridge")
    for field in ("store", "background"):
        if data.get(field) is not None and data[field] is not False:
            _error(field, "only false is supported by the stateless Responses bridge")
    if data.get("include") not in (None, []):
        _error("include", "native output expansions are unsupported")
    if data.get("truncation") not in (None, "disabled"):
        _error("truncation", "automatic truncation is unsupported")
    if data.get("use_chat_completions_api") is not None and data["use_chat_completions_api"] is not True:
        _error("use_chat_completions_api", "this deployment requires the Chat bridge")
    for field in ("stream", "parallel_tool_calls"):
        if data.get(field) is not None and type(data[field]) is not bool:
            _error(field, "must be a boolean")
    options = data.get("stream_options")
    if options is not None:
        if not isinstance(options, dict) or any(
            not ((key == "include_usage" and value is True) or (key == "include_obfuscation" and value is False))
            for key, value in options.items()
        ):
            _error("stream_options", "Responses requires usage and does not support obfuscation")
    if data.get("instructions") is not None and not isinstance(data["instructions"], str):
        _error("instructions", "must be a string")
    _limit(data.get("max_output_tokens"))
    _input(data.get("input"))
    if data.get("input") == [] and not data.get("instructions"):
        _error("input", "must supply messages or instructions")
    _text(data.get("text"))
    reasoning = data.get("reasoning")
    if reasoning is not None and (
        not isinstance(reasoning, dict) or set(reasoning) - {"effort"}
        or reasoning.get("effort") not in (None, "none", "minimal", "low", "medium", "high", "xhigh", "max")
    ):
        _error("reasoning", "only reasoning.effort is supported")
    tools = data.get("tools")
    if tools is not None and not isinstance(tools, list):
        _error("tools", "must be a list of function tools")
    names = [_function(tool, f"tools[{i}]")["name"] for i, tool in enumerate(tools or [])]
    if len(set(names)) != len(names):
        _error("tools", "function names must be unique")
    choice = data.get("tool_choice")
    if isinstance(choice, dict):
        if choice.get("type") != "function" or set(choice) - {"type", "name", "function"}:
            _error("tool_choice", "only a named function or auto/none/required is supported")
        if "function" in choice:
            function = choice["function"]
            if set(choice) != {"type", "function"} or not isinstance(function, dict) or set(function) != {"name"}:
                _error("tool_choice", "must contain exactly one function name")
            name = function["name"]
        else:
            name = choice.get("name")
        if not isinstance(name, str) or not name.strip():
            _error("tool_choice.name", "must be a nonempty string")
        if deployment and name not in names:
            _error("tool_choice.name", "must name a declared function")
    elif choice not in (None, "auto", "none", "required"):
        _error("tool_choice", "only a named function or auto/none/required is supported")
    elif deployment and choice == "required" and not names:
        _error("tool_choice", "required needs a declared function")


def prepare_responses_request(data: dict) -> None:
    validate_responses_request(data, deployment=True)
    if data.get("max_output_tokens") is not None:
        # Conversion maps max_output_tokens to max_tokens. A lingering modern
        # Chat alias would win over it in both tenancy and the gateway.
        data.pop("max_tokens", None)
        data.pop("max_completion_tokens", None)
    if data.get("tools"):
        data["tools"] = [{"type": "function", **_function(tool, f"tools[{i}]")}
                         for i, tool in enumerate(data["tools"])]
    data["use_chat_completions_api"] = True
