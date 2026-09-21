"""The Chat/Completion subset whose K3 prompt we can count faithfully.

Call ``normalize_engine_prompt`` before BOTH counting and inference. It is pure:
it returns a detached payload, retains generation fields, and never fetches or
decodes media. Media validation/normalization belongs to the caller's media
stage. The tokenizer projection is specific to the installed vLLM K3 protocol,
not a claim that arbitrary OpenAI extensions are supported.
"""

from __future__ import annotations

import copy
import json
from typing import NoReturn


CHAT_PATHS = frozenset({"/v1/chat/completions", "/chat/completions"})
COMPLETION_PATHS = frozenset({"/v1/completions", "/completions"})
PROTOCOL_REVISION = "kimi-k3-tokenize-v1"
_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
_NATIVE_EFFORTS = frozenset({"low", "high", "max"})


class TokenCountInputError(ValueError):
    """Invalid/unsupported prompt; the gateway should return HTTP 400.

    Messages identify fields, never their potentially sensitive values. An
    upstream 4xx is deliberately not copied verbatim into this exception.
    """

    status_code = 400

    def __init__(
        self, message: str, *, param: str | None = None,
        reason: str = "invalid_input", upstream_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.param = param
        self.reason = reason
        self.upstream_status = upstream_status


def _invalid(param: str, message: str, *, unsupported: bool = False) -> NoReturn:
    raise TokenCountInputError(
        f"{param}: {message}", param=param,
        reason="unsupported_input" if unsupported else "invalid_input",
    )


def _boolean(body: dict, key: str, path: str = "") -> None:
    if key in body and type(body[key]) is not bool:
        _invalid(path + key, "must be a boolean")


def _function(value: object, path: str) -> None:
    if not isinstance(value, dict):
        _invalid(path, "must be a function definition")
    if not isinstance(value.get("name"), str) or not value["name"].strip():
        _invalid(path + ".name", "must be a nonempty string")
    if value.get("description") is not None and not isinstance(value["description"], str):
        _invalid(path + ".description", "must be a string")
    if value.get("parameters") is not None and not isinstance(value["parameters"], dict):
        _invalid(path + ".parameters", "must be a schema object")
    for key in ("strict", "defer_loading"):
        if value.get(key) is not None:
            _boolean(value, key, path + ".")


def _tools(value: object, path: str, *, empty_allowed: bool = False) -> None:
    if value is None:
        return
    if not isinstance(value, list) or (not value and not empty_allowed):
        _invalid(path, "must be a nonempty list of function tools")
    for index, tool in enumerate(value):
        location = f"{path}[{index}]"
        if not isinstance(tool, dict) or tool.get("type", "function") != "function":
            _invalid(location, "only function tools are supported", unsupported=True)
        _function(tool.get("function"), location + ".function")
        if tool.get("defer_loading") is not None:
            _boolean(tool, "defer_loading", location + ".")


def _choice(value: object, path: str) -> None:
    if value is None or (isinstance(value, str) and value in {"none", "auto", "required"}):
        return
    if (
        not isinstance(value, dict)
        or set(value) - {"type", "function"}
        or value.get("type", "function") != "function"
        or not isinstance(value.get("function"), dict)
        or set(value["function"]) != {"name"}
        or not isinstance(value["function"]["name"], str)
        or not value["function"]["name"].strip()
    ):
        _invalid(path, "must be none, auto, required, or a named function choice")


def _response_format(value: object, path: str, *, template: bool = False) -> None:
    if value is None:
        return
    if template and isinstance(value, str) and value in {"text", "json_object", "json_schema"}:
        return
    if not isinstance(value, dict):
        _invalid(path, "must be a response-format object")
    if not isinstance(value.get("type"), str) or value["type"] not in {"text", "json_object", "json_schema"}:
        _invalid(path, "only text, json_object and json_schema are supported", unsupported=True)
    if value["type"] == "json_schema":
        schema = value.get("json_schema")
        if not isinstance(schema, dict):
            _invalid(path + ".json_schema", "must contain a named schema")
        if not template and (not isinstance(schema.get("name"), str) or not schema["name"].strip()):
            _invalid(path + ".json_schema.name", "must be a nonempty string")
        # The HTTP protocol's model_dump uses the internal json_schema alias;
        # K3 accepts that spelling in template kwargs as well as public schema.
        definition = schema.get("schema", schema.get("json_schema") if template else None)
        if not isinstance(definition, dict):
            _invalid(path + ".json_schema.schema", "must be a schema object")
        if schema.get("strict") is not None:
            _boolean(schema, "strict", path + ".json_schema.")


def _legacy_controls(body: dict) -> None:
    functions = body.pop("functions", None)
    if functions is not None:
        if body.get("tools") is not None:
            _invalid("functions", "cannot be combined with tools")
        if not isinstance(functions, list) or not functions:
            _invalid("functions", "must be a nonempty list")
        for index, function in enumerate(functions):
            _function(function, f"functions[{index}]")
        body["tools"] = [{"type": "function", "function": function} for function in functions]

    choice = body.pop("function_call", None)
    if choice is not None:
        if body.get("tool_choice") is not None:
            _invalid("function_call", "cannot be combined with tool_choice")
        if isinstance(choice, str) and choice in {"none", "auto"}:
            body["tool_choice"] = choice
        elif (
            isinstance(choice, dict) and set(choice) == {"name"}
            and isinstance(choice["name"], str) and choice["name"].strip()
        ):
            body["tool_choice"] = {"type": "function", "function": choice}
        else:
            _invalid("function_call", "must be none, auto, or a function name object")


def _arguments(value: object, path: str) -> None:
    if not isinstance(value, str):
        _invalid(path, "must be a JSON object string")
    try:
        parsed = json.loads(value) if value else {}
    except (ValueError, RecursionError):
        _invalid(path, "must be a JSON object string")
    if not isinstance(parsed, dict):
        _invalid(path, "must encode an object")


def _content(value: object, path: str, *, assistant: bool) -> None:
    if isinstance(value, str) or (assistant and value is None):
        return
    if not isinstance(value, list):
        _invalid(path, "must be text or a list of text/image parts")
    for index, part in enumerate(value):
        location = f"{path}[{index}]"
        if isinstance(part, str):
            continue
        if not isinstance(part, dict):
            _invalid(location, "must be a text/image part")
        kind = part.get("type")
        if kind is not None and not isinstance(kind, str):
            _invalid(location + ".type", "must be a string")
        if kind in {"text", "input_text", "output_text", "refusal", "thinking"}:
            key = {"refusal": "refusal", "thinking": "thinking"}.get(kind, "text")
            if not isinstance(part.get(key), str):
                _invalid(location + "." + key, "must be a string")
        elif kind in {"image_url", "input_image"} or (kind is None and "image_url" in part):
            if part.get("uuid") is not None:
                _invalid(location + ".uuid", "cache-only media identity is unsupported for counting", unsupported=True)
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            if not isinstance(url, str) or not url.strip():
                _invalid(location + ".image_url", "requires image data or a URL")
        else:
            _invalid(location, "normalize media to text/image_url parts before counting", unsupported=True)


def _messages(body: dict) -> None:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        _invalid("messages", "must be a nonempty message list")
    # Generated legacy IDs must not collide with a caller's modern history.
    ids = set()
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            _invalid(f"messages[{index}]", "must be a message object")
        calls = message.get("tool_calls")
        if calls is not None and not isinstance(calls, list):
            _invalid(f"messages[{index}].tool_calls", "must be a function-call list")
        ids.update(call["id"] for call in (calls or [])
                   if isinstance(call, dict) and isinstance(call.get("id"), str))
    legacy = None
    for index, message in enumerate(messages):
        path = f"messages[{index}]"
        if not isinstance(message, dict):
            _invalid(path, "must be a message object")
        if "tools" in message:
            _invalid(path + ".tools", "message-scoped tools are dropped by this vLLM parser", unsupported=True)
        role = message.get("role")
        if not isinstance(role, str):
            _invalid(path + ".role", "must be a string")
        if role == "developer":
            _invalid(path + ".role", "K3 does not render developer messages; normalize their semantics explicitly", unsupported=True)

        old_call = message.pop("function_call", None)
        if old_call is not None:
            if role != "assistant" or message.get("tool_calls") is not None:
                _invalid(path + ".function_call", "requires an assistant without modern tool_calls")
            if not isinstance(old_call, dict) or set(old_call) != {"name", "arguments"}:
                _invalid(path + ".function_call", "must contain name and arguments")
            _function(old_call, path + ".function_call")
            _arguments(old_call["arguments"], path + ".function_call.arguments")
            call_id = f"k3_legacy_function_call_{index}"
            if call_id in ids:
                _invalid(path + ".function_call", "generated legacy ID conflicts with modern history")
            message["tool_calls"] = [{"id": call_id, "type": "function", "function": old_call}]
            legacy = (call_id, old_call["name"])
        elif role == "assistant":
            legacy = None
        if role == "function":
            if legacy is None or message.get("name") != legacy[1] or message.get("tool_call_id") is not None:
                _invalid(path, "legacy function result must match the preceding legacy call")
            message["role"] = role = "tool"
            message["tool_call_id"] = legacy[0]
            legacy = None
        if role not in {"system", "user", "assistant", "tool"}:
            _invalid(path + ".role", "unsupported K3 message role", unsupported=True)

        old_reasoning = message.pop("reasoning_content", None)
        if old_reasoning is not None and message.get("reasoning") is None:
            message["reasoning"] = old_reasoning
        if message.get("reasoning") is not None:
            if role != "assistant" or not isinstance(message["reasoning"], str):
                _invalid(path + ".reasoning", "requires assistant reasoning text")
        for key in ("name", "tool", "tool_call_id"):
            if message.get(key) is not None and not isinstance(message[key], str):
                _invalid(path + "." + key, "must be a string")
        calls = message.get("tool_calls")
        if calls is not None:
            if role != "assistant" or not isinstance(calls, list):
                _invalid(path + ".tool_calls", "requires an assistant function-call list")
            for number, call in enumerate(calls):
                location = f"{path}.tool_calls[{number}]"
                if not isinstance(call, dict) or call.get("type", "function") != "function":
                    _invalid(location, "only function calls are supported", unsupported=True)
                _function(call.get("function"), location + ".function")
                _arguments(call["function"].get("arguments"), location + ".function.arguments")
        _content(message.get("content"), path + ".content", assistant=role == "assistant")


def _chat_kwargs(body: dict) -> dict:
    """Mirror ChatCompletionRequest.build_chat_params, including unset values.

    TokenizeChatRequest only forwards generation-prompt flags on its own. The
    root Chat fields below must therefore be supplied as template kwargs.
    """
    kwargs = dict(body.get("chat_template_kwargs") or {})
    tools = body.get("tools")
    derived = {
        "add_generation_prompt": body.get("add_generation_prompt", True),
        "continue_final_message": body.get("continue_final_message", False),
        "documents": body.get("documents"),
        "reasoning_effort": body.get("reasoning_effort"),
        "tool_choice": body.get("tool_choice", "auto" if tools else "none") if tools else None,
        "response_format": body.get("response_format"),
    }
    if body.get("reasoning_effort") is not None and "enable_thinking" not in kwargs:
        derived["enable_thinking"] = body["reasoning_effort"] != "none"
    kwargs.update({key: value for key, value in derived.items() if value not in (None, "auto")})
    return kwargs


def normalize_engine_prompt(payload: dict, path: str) -> dict:
    """Return a detached, idempotent payload to use for inference AND counting.

    Requires the post-SDK Chat/Completion body (extra_body already flattened),
    with media in OpenAI text/image form. Does not clamp or change generation
    policy. Unsupported fields fail explicitly rather than being discarded.
    """
    if path not in CHAT_PATHS | COMPLETION_PATHS:
        _invalid("path", "only Chat and plain-text Completions are supported", unsupported=True)
    if not isinstance(payload, dict):
        _invalid("body", "must be a JSON object")
    try:
        body = copy.deepcopy(payload)
    except (TypeError, ValueError, RecursionError):
        _invalid("body", "must be a finite JSON object")
    for key in ("extra_body", "input", "instructions", "prompt_embeds", "truncate_prompt_tokens", "truncation_side"):
        if body.get(key) is not None:
            _invalid(key, "unsupported in the normalized engine prompt", unsupported=True)
    if body.get("model") is not None and (not isinstance(body["model"], str) or not body["model"].strip()):
        _invalid("model", "must be a served-model name")
    _boolean(body, "add_special_tokens")
    if body.get("chat_template") is not None:
        _invalid("chat_template", "request chat templates are not supported", unsupported=True)
    if body.get("continue_final_message") is True:
        _invalid("continue_final_message", "K3 continuation is not supported", unsupported=True)

    if path in COMPLETION_PATHS:
        if not isinstance(body.get("prompt"), str):
            _invalid("prompt", "only a single plain-text string is supported; batches/token arrays are unsupported", unsupported=True)
        for key in (
            "messages", "functions", "function_call", "tools", "tool_choice", "response_format",
            "chat_template_kwargs", "reasoning_effort", "documents", "suffix",
            "media_io_kwargs", "mm_processor_kwargs", "add_generation_prompt", "continue_final_message",
        ):
            if body.get(key) is not None:
                _invalid(key, "unsupported for plain-text Completions", unsupported=True)
        return body

    if "prompt" in body:
        _invalid("prompt", "cannot be combined with messages")
    _legacy_controls(body)
    _messages(body)
    _tools(body.get("tools"), "tools", empty_allowed=True)
    _choice(body.get("tool_choice"), "tool_choice")
    choice = body.get("tool_choice")
    if choice is not None and choice != "none" and not body.get("tools"):
        _invalid("tool_choice", "requires tools")
    if isinstance(choice, dict) and choice["function"]["name"] not in {
        tool["function"]["name"] for tool in body["tools"]
    }:
        _invalid("tool_choice", "must name a declared function")
    _response_format(body.get("response_format"), "response_format")
    for key in ("add_generation_prompt", "continue_final_message"):
        _boolean(body, key)
    effort = body.get("reasoning_effort")
    if effort is not None and (not isinstance(effort, str) or effort not in _EFFORTS):
        _invalid("reasoning_effort", "unsupported reasoning effort")
    for key in ("chat_template_kwargs", "media_io_kwargs", "mm_processor_kwargs"):
        if body.get(key) is not None and not isinstance(body[key], dict):
            _invalid(key, "must be an object")
    if body.get("documents") is not None and (
        not isinstance(body["documents"], list)
        or any(not isinstance(doc, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                               for k, v in doc.items()) for doc in body["documents"])
    ):
        _invalid("documents", "must be a list of string-valued document objects")
    kwargs = body.get("chat_template_kwargs") or {}
    # The media/control normalizer can retain native root fields after binding
    # their effective value into the template. Those redundant fields are fine;
    # a native control ONLY at the root would be silently ignored by vLLM.
    bound_keys = {
        "thinking": ("thinking",),
        "thinking_effort": ("thinking_effort", "reasoning_effort"),
        "enable_thinking": ("thinking", "enable_thinking"),
        "response_schema": ("response_schema",),
    }
    for key, targets in bound_keys.items():
        if body.get(key) is not None and not any(kwargs.get(target) not in (None, "auto") for target in targets):
            _invalid(key, "must be normalized into chat_template_kwargs", unsupported=True)
    for key in ("chat_template", "image_prompts", "truncation", "truncate_prompt_tokens", "truncation_side", "max_length", "padding", "return_tensors"):
        if kwargs.get(key) is not None:
            _invalid("chat_template_kwargs." + key, "unsupported prompt override", unsupported=True)
    if kwargs.get("continue_final_message") is True:
        _invalid("chat_template_kwargs.continue_final_message", "K3 continuation is not supported", unsupported=True)
    for key in ("thinking", "enable_thinking", "add_generation_prompt", "continue_final_message"):
        if kwargs.get(key) is not None:
            _boolean(kwargs, key, "chat_template_kwargs.")
    _tools(kwargs.get("tools"), "chat_template_kwargs.tools", empty_allowed=True)
    _choice(kwargs.get("tool_choice"), "chat_template_kwargs.tool_choice")
    _response_format(kwargs.get("response_format"), "chat_template_kwargs.response_format", template=True)
    effective = {key: value for key, value in _chat_kwargs(body).items() if value not in (None, "auto")}
    thinking = effective.get("thinking", effective.get("enable_thinking", True))
    effort = effective.get("thinking_effort", effective.get("reasoning_effort", "max"))
    if thinking and (not isinstance(effort, str) or effort not in _NATIVE_EFFORTS):
        _invalid("chat_template_kwargs.thinking_effort", "K3 thinking requires low, high, or max")
    return body


def build_tokenize_request(payload: dict) -> dict:
    """Build the installed TokenizeChatRequest/TokenizeCompletionRequest body."""
    path = "/v1/chat/completions" if isinstance(payload, dict) and "messages" in payload else "/v1/completions"
    body = normalize_engine_prompt(payload, path)
    if path in COMPLETION_PATHS:
        request = {"prompt": body["prompt"], "add_special_tokens": body.get("add_special_tokens", True)}
    else:
        request = {
            "messages": body["messages"],
            "add_generation_prompt": body.get("add_generation_prompt", True),
            "continue_final_message": False,
            "add_special_tokens": body.get("add_special_tokens", False),
            "chat_template_kwargs": _chat_kwargs(body),
        }
        for key in ("tools", "media_io_kwargs", "mm_processor_kwargs"):
            if key in body:
                request[key] = body[key]
    if "model" in body:
        request["model"] = body["model"]
    request["return_token_strs"] = False
    return request
