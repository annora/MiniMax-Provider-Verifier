import pytest

from helpers import *


REASONING_EFFORT_ENUM = ("low", "medium", "high", "xhigh", "max")
COMPLEX_PROMPT = (
    "Let N be the number of positive divisors of 17017^17 that are "
    "congruent to 5 modulo 12. Find N modulo 1000 and explain your method."
)


def _reasoning_tokens(result: dict) -> int:
    """缺失的 reasoning_tokens 视为 0。"""
    if result.get("stream"):
        usage = next(
            (chunk["usage"] for chunk in reversed(result.get("chunks") or [])
             if isinstance(chunk, dict) and chunk.get("usage")),
            {},
        )
    else:
        usage = (result.get("body") or {}).get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    try:
        return int(details.get("reasoning_tokens") or 0)
    except (TypeError, ValueError):
        return 0


def _request(prompt: str, *, stream: bool = False, **fields) -> dict:
    payload = {"messages": oai_simple_messages(prompt), **fields}
    if stream:
        payload["stream_options"] = {"include_usage": True}
    result = oai_chat(payload, stream=stream)
    if stream:
        assert_oai_stream_success(result)
    else:
        assert_oai_success(result)
    assert get_oai_content(result).strip(), "预期返回非空的最终回答"
    return result


def _assert_usage_if_thinking(result: dict, context: str) -> None:
    if get_thinking_signals(result)["any"]:
        assert _reasoning_tokens(result) > 0, (
            f"{context}: 返回思考内容时，reasoning_tokens 应大于 0；"
            f"实际为 {_reasoning_tokens(result)}"
        )


def _assert_disabled(result: dict, context: str) -> None:
    assert_thinking_absent(result, msg=context)
    assert _reasoning_tokens(result) == 0, (
        f"{context}: disabled 时 reasoning_tokens 应为 0 或缺失，"
        f"实际为 {_reasoning_tokens(result)}"
    )


class TestReasoningEffort:
    @pytest.mark.parametrize("thinking", [None, {"type": "adaptive"}],
                             ids=["default", "adaptive"])
    @pytest.mark.parametrize("prompt", ["Say hello.", COMPLEX_PROMPT],
                             ids=["short", "complex"])
    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_01_01_adaptive(self, thinking, prompt, stream):
        """默认和显式 adaptive 都应正常回答；有思考时校验 token。"""
        fields = {} if thinking is None else {"thinking": thinking}
        result = _request(prompt, stream=stream, **fields)
        _assert_usage_if_thinking(result, f"adaptive prompt={prompt[:12]}, stream={stream}")

    @pytest.mark.parametrize("prompt", ["Hi", COMPLEX_PROMPT],
                             ids=["short", "complex"])
    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_01_02_thinking_disabled(self, prompt, stream):
        """单独传 disabled 时，应直接回答且不返回推理内容。"""
        result = _request(prompt, stream=stream, thinking={"type": "disabled"})
        _assert_disabled(result, f"disabled prompt={prompt[:12]}, stream={stream}")

    @pytest.mark.parametrize("effort", REASONING_EFFORT_ENUM)
    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_01_03_valid_reasoning_effort_accepted(self, effort, stream):
        """合法 effort 值应被接受；有思考时校验 token 计数。"""
        result = _request(COMPLEX_PROMPT, stream=stream, reasoning_effort=effort)
        _assert_usage_if_thinking(result, f"effort={effort}, stream={stream}")

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_01_04_disabled_repeated(self, stream):
        """复杂题重复采样，检查 disabled 是否偶发泄露思考。"""
        for sample in range(5):
            result = _request(COMPLEX_PROMPT, stream=stream,
                              thinking={"type": "disabled"})
            _assert_disabled(result, f"disabled sample={sample + 1}, stream={stream}")

    @pytest.mark.parametrize("prompt", ["Say hello.", COMPLEX_PROMPT],
                             ids=["short", "complex"])
    def test_01_05_adaptive_repeated(self, prompt):
        """adaptive 重复采样，只检查实际返回的推理与 token 是否对应。"""
        for sample in range(5):
            result = _request(prompt, thinking={"type": "adaptive"})
            _assert_usage_if_thinking(result, f"adaptive sample={sample + 1}")
