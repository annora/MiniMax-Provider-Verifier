"""MiniMax-M3 reasoning_effort compatibility cases.

MiniMax-M3 ignores reasoning_effort. Its thinking.type remains adaptive by
default, while thinking.type=disabled skips thinking. Set M3_MODEL to the M3
model ID or provider alias. This suite uses the same OpenAI Chat Completions
helpers and case naming as the M3.1 effort suite.

Run with:

    M3_BASE_URL=... M3_API_KEY=... M3_MODEL=MiniMax-M3 \\
        python3 -m pytest m3_reasoning_effort_tests.py -v

Case naming convention: test_<module_id>_<index_within_module>_<scenario>
Module id / topic:
    01  reasoning_effort     ignored-field compatibility

All cases go through helpers.oai_chat() against /v1/chat/completions; jsonl is
written to RUN_LOG_PATH (injected by conftest).
"""
import pytest

from helpers import *


REASONING_EFFORT_ENUM = ("low", "medium", "high", "xhigh", "max")
IGNORED_EFFORT_VALUES = (*REASONING_EFFORT_ENUM, "none", "minimal")
MAX_TOKENS = 1024


def _request(effort: str, *, thinking: str | None = None, stream: bool = False) -> dict:
    payload = {
        "messages": oai_simple_messages("Compute 23 * 47 and give the answer."),
        "reasoning_effort": effort,
        "max_tokens": MAX_TOKENS,
    }
    if thinking is not None:
        payload["thinking"] = {"type": thinking}
    if stream:
        payload["stream_options"] = {"include_usage": True}
    return oai_chat(payload, stream=stream)


def _reasoning_tokens(result: dict) -> int:
    """Read reasoning_tokens from the final usage, or return zero if absent."""
    if result.get("stream"):
        usage = next(
            (chunk["usage"] for chunk in reversed(result.get("chunks") or [])
             if isinstance(chunk, dict) and isinstance(chunk.get("usage"), dict)),
            {},
        )
    else:
        usage = (result.get("body") or {}).get("usage") or {}
    value = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _assert_success(result: dict, context: str) -> None:
    if result.get("stream"):
        assert_oai_stream_success(result)
        assert_stream_complete(result, msg=context)
    else:
        assert_oai_success(result)
    assert get_oai_content(result).strip(), f"{context}: missing final answer"


def _assert_adaptive(result: dict, context: str) -> None:
    """Adaptive may skip thinking; emitted reasoning must have token usage."""
    _assert_success(result, context)
    if get_thinking_signals(result)["any"]:
        assert _reasoning_tokens(result) > 0, (
            f"{context}: reasoning present but reasoning_tokens <= 0"
        )


def _assert_disabled(result: dict, context: str) -> None:
    _assert_success(result, context)
    assert_thinking_absent(result, msg=context)
    assert _reasoning_tokens(result) == 0, (
        f"{context}: disabled thinking reported reasoning tokens"
    )


# ============================================================
# 01 reasoning_effort — ignored-field compatibility
# ============================================================

class TestReasoningEffort:
    """Check that reasoning_effort does not change M3 thinking behavior."""

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_01_07_thinking_without_effort(self, stream):
        """Check the adaptive and disabled baselines without reasoning_effort."""
        for thinking in (None, "adaptive", "disabled"):
            payload = {
                "messages": oai_simple_messages("Compute 23 * 47 and give the answer."),
                "max_tokens": MAX_TOKENS,
            }
            if thinking is not None:
                payload["thinking"] = {"type": thinking}
            if stream:
                payload["stream_options"] = {"include_usage": True}
            result = oai_chat(payload, stream=stream)
            context = f"thinking={thinking or 'default'} without reasoning_effort"
            if thinking == "disabled":
                _assert_disabled(result, context)
            else:
                _assert_adaptive(result, context)

    @pytest.mark.parametrize("effort", REASONING_EFFORT_ENUM)
    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_01_01_valid_effort_ignored(self, effort, stream):
        """All documented effort values are accepted with default adaptive."""
        _assert_adaptive(
            _request(effort, stream=stream), f"reasoning_effort={effort}"
        )

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_01_02_none_ignored(self, stream):
        """M3 ignores none rather than rejecting it as M3.1 does."""
        _assert_adaptive(_request("none", stream=stream), "reasoning_effort=none")

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_01_03_minimal_ignored(self, stream):
        """The minimal alias is accepted without enabling depth control on M3."""
        _assert_adaptive(
            _request("minimal", stream=stream), "reasoning_effort=minimal"
        )

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_01_04_out_of_enum_ignored(self, stream):
        """An unknown effort value falls back to default M3 behavior."""
        _assert_adaptive(
            _request("ultra_super_max_xyz", stream=stream),
            "reasoning_effort=out_of_enum",
        )

    @pytest.mark.parametrize("effort", IGNORED_EFFORT_VALUES)
    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_01_05_effort_does_not_override_disabled(self, effort, stream):
        """thinking.type=disabled skips reasoning regardless of effort."""
        _assert_disabled(
            _request(effort, thinking="disabled", stream=stream),
            f"thinking.disabled + reasoning_effort={effort}",
        )

    @pytest.mark.parametrize("effort", ["low", "max"])
    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_01_06_effort_does_not_break_adaptive(self, effort, stream):
        """Explicit adaptive remains accepted at low and max effort."""
        _assert_adaptive(
            _request(effort, thinking="adaptive", stream=stream),
            f"thinking.adaptive + reasoning_effort={effort}",
        )
