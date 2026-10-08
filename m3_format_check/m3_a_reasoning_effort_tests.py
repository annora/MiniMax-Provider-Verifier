"""
M3.1 API Test — reasoning_effort (thinking-depth) case collection

reasoning_effort controls thinking depth. Per the spec
(/api-reference/text-chat-openai):

  - reasoning_effort adjusts thinking depth. Models that do not support it
    ignore the field entirely.
  - Valid enum: low / medium / high / xhigh / max.
  - `none` is NOT accepted and returns HTTP 400.
  - `minimal` is mapped to `low`.
  - Out-of-enum values are ignored and use the default thinking depth.
  - `thinking.type=disabled` is NOT accepted and returns HTTP 400.
  - Thinking runs regardless of reasoning_effort,
    and the thinking content is split into `reasoning_content`
    (reasoning_split defaults to true for such a model).

This is a standalone suite (deliberately separate from the general M3 case
set). Set M3_MODEL to the M3.1 model ID or provider alias. Run with:

    M3_BASE_URL=... M3_API_KEY=... M3_MODEL=MiniMax-M3.1-Flash-Preview \\
        python3 -m pytest m3_a_reasoning_effort_tests.py -v

Case naming convention: test_<module_id>_<index_within_module>_<scenario>
Module id / topic:
    01  reasoning_effort     reasoning_effort thinking-depth control

All cases go through helpers.oai_chat() against /v1/chat/completions; jsonl is
written to RUN_LOG_PATH (injected by conftest).
"""
import pytest

from helpers import *
from image_tools import make_png_base64


REASONING_EFFORT_ENUM = ("low", "medium", "high", "xhigh", "max")
MAX_TOKENS = 2048


def _stream_options(stream: bool) -> dict:
    """Request the terminal usage chunk when reasoning tokens are asserted."""
    return {"stream_options": {"include_usage": True}} if stream else {}


def _reasoning_tokens(r: dict) -> int:
    """usage.completion_tokens_details.reasoning_tokens (0 when absent).

    Non-stream reads body.usage; stream reads the last chunk carrying usage.
    """
    if r.get("stream"):
        usage = {}
        for chunk in reversed(r.get("chunks") or []):
            if isinstance(chunk, dict) and chunk.get("usage"):
                usage = chunk["usage"]
                break
    else:
        usage = (r.get("body") or {}).get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    try:
        return int(details.get("reasoning_tokens") or 0)
    except (TypeError, ValueError):
        return 0


def _assert_reasoning_usage_consistent(r: dict, context: str) -> None:
    """自适应思考可能跳过；只要返回了 reasoning，就必须报告推理 token。"""
    if get_thinking_signals(r)["any"]:
        assert _reasoning_tokens(r) > 0, (
            f"{context}: reasoning present but reasoning_tokens <= 0"
        )


# ============================================================
# 01 reasoning_effort — thinking-depth control
# ============================================================

class TestReasoningEffort:
    """reasoning_effort field: valid-enum acceptance, forced thinking,
    `none`/`thinking.disabled` rejection, `minimal`->low mapping, invalid-value fallback, and
    streaming coexistence."""

    @pytest.mark.parametrize("effort", REASONING_EFFORT_ENUM)
    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_01_01_valid_effort_accepted(self, effort, stream):
        """Each valid enum value (low/medium/high/xhigh/max) is accepted → HTTP 200.

        Low effort can answer simple turns without emitting reasoning. When
        reasoning is emitted, usage.reasoning_tokens must be positive.
        """
        r = oai_chat({
            "messages": oai_simple_messages("What is 23 * 47? Think it through."),
            "reasoning_effort": effort,
            "max_tokens": MAX_TOKENS,
            **_stream_options(stream),
        }, stream=stream)

        if stream:
            assert_oai_stream_success(r)
        else:
            assert_oai_success(r)
        assert get_oai_content(r).strip(), f"reasoning_effort={effort}: missing final answer"
        _assert_reasoning_usage_consistent(r, f"reasoning_effort={effort}")

    def test_01_02_none_rejected(self):
        """M3.1 forces thinking, so reasoning_effort=none returns HTTP 400."""
        r = oai_chat({
            "messages": oai_simple_messages("Say hello"),
            "reasoning_effort": "none",
        })
        assert_error(r, 400)

    def test_01_03_minimal_accepted(self):
        """minimal 按文档映射为 low；此 case 验证其可被接受。"""
        r = oai_chat({
            "messages": oai_simple_messages("What is 12 + 30?"),
            "reasoning_effort": "minimal",
            "max_tokens": MAX_TOKENS,
        })
        assert_oai_success(r)
        _assert_reasoning_usage_consistent(r, "reasoning_effort=minimal")

    def test_01_04_out_of_enum_ignored(self):
        """按官方文档，枚举外取值应忽略并使用默认思考深度。"""
        r = oai_chat({
            "messages": oai_simple_messages("What is 2+2?"),
            "reasoning_effort": "ultra_super_max_xyz",
            "max_tokens": MAX_TOKENS,
        })
        assert_oai_success(r)
        _assert_reasoning_usage_consistent(r, "reasoning_effort=枚举外取值")

    def test_01_05_effort_with_thinking_adaptive(self):
        """reasoning_effort combined with thinking.type=adaptive (the only
        thinking value accepted when thinking is forced on) → HTTP 200 +
        thinking present."""
        r = oai_chat({
            "messages": oai_simple_messages("Explain why the sky is blue, briefly."),
            "reasoning_effort": "high",
            "thinking": {"type": "adaptive"},
            "max_tokens": MAX_TOKENS,
        })
        assert_oai_success(r)
        _assert_reasoning_usage_consistent(r, "reasoning_effort=high + thinking.adaptive")

    def test_01_06_effort_with_thinking_disabled(self):
        """M3.1 rejects thinking.type=disabled even with reasoning_effort."""
        r = oai_chat({
            "messages": oai_simple_messages("Hi"),
            "reasoning_effort": "high",
            "thinking": {"type": "disabled"},
        })
        assert_error(r, 400)

    def test_01_07_reasoning_content_split(self):
        """reasoning_split defaults to true, so WHEN the model thinks, the
        thinking is returned in the dedicated `reasoning_content` field rather
        than as an inline <think> tag in content.

        Adaptive may skip reasoning on a given turn; this case only checks
        placement when reasoning is emitted.
        """
        r = oai_chat({
            "messages": oai_simple_messages(
                "A train travels 60 km in 45 minutes. What is its average speed "
                "in km/h? Show the reasoning."
            ),
            "reasoning_effort": "high",
            "max_tokens": MAX_TOKENS,
        })
        assert_oai_success(r)
        sig = get_thinking_signals(r)
        if not sig["any"]:
            pytest.skip("自适应思考跳过了本轮，无法判定 reasoning_content 的位置")
        # A thinking signal exists: with reasoning_split defaulting to true it
        # must land in reasoning_content (not only as an inline <think> tag).
        assert sig["reasoning_content"].strip(), (
            "thinking signal present but reasoning_content is empty; "
            "reasoning_split=true should place thinking in message.reasoning_content"
        )
        assert _reasoning_tokens(r) > 0, (
            f"thinking signal present but reasoning_tokens <= 0, "
            f"got {_reasoning_tokens(r)}"
        )

    @pytest.mark.parametrize("effort", ["low", "max"])
    def test_01_08_effort_stream_thinking(self, effort):
        """reasoning_effort under streaming coexists with the SSE protocol; the
        stream completes cleanly and carries a thinking signal."""
        r = oai_chat({
            "messages": oai_simple_messages("Compute 17 * 19 step by step."),
            "reasoning_effort": effort,
            "max_tokens": MAX_TOKENS,
            **_stream_options(True),
        }, stream=True)
        assert_oai_stream_success(r)
        assert_stream_complete(r, msg=f"reasoning_effort={effort} stream")
        _assert_reasoning_usage_consistent(r, f"reasoning_effort={effort} stream")

    @pytest.mark.parametrize("effort", ["low", "max"])
    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_01_09_effort_with_image(self, effort, stream):
        """Image input accepts low/max effort with thinking and a final answer,
        in both non-streaming and complete streaming responses.
        """
        r = oai_chat({
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": make_png_base64(672, 672, r=255, g=0, b=0),
                }},
                {"type": "text", "text": (
                    "Identify the dominant color in this image. Use 17 if it is "
                    "red, 23 if green, or 31 if blue, then multiply that number "
                    "by 19. Think it through and give the color and result."
                )},
            ]}],
            "reasoning_effort": effort,
            "max_tokens": MAX_TOKENS,
            **_stream_options(stream),
        }, stream=stream)

        context = f"image + reasoning_effort={effort}, stream={stream}"
        if stream:
            assert_oai_stream_success(r)
            assert_stream_complete(r, msg=context)
        else:
            assert_oai_success(r)
        assert get_oai_content(r).strip(), f"{context}: expected non-empty final answer"
        _assert_reasoning_usage_consistent(r, context)

    @pytest.mark.parametrize("effort", ["low", "high", "max"])
    def test_01_10_complex_problem_triggers_reasoning(self, effort):
        """用多步数论题检查实际思考；高档位应返回推理及对应 token。"""
        question = (
            "How many positive divisors of 2^8 * 3^5 are divisible by 12? "
            "Show the calculation step by step."
        )
        r = oai_chat({
            "messages": oai_simple_messages(question),
            "reasoning_effort": effort,
            "max_tokens": 4096,
        })
        assert_oai_success(r)
        assert get_oai_content(r).strip(), f"reasoning_effort={effort}: missing final answer"
        if effort in ("high", "max"):
            assert_thinking_present(r, msg=f"complex problem + reasoning_effort={effort}")
        _assert_reasoning_usage_consistent(r, f"complex problem + reasoning_effort={effort}")

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_01_11_thinking_adaptive_without_effort(self, stream):
        """显式 adaptive、不传 effort：流式与非流式均应正常响应。"""
        r = oai_chat({
            "messages": oai_simple_messages("What is 23 * 47? Show the calculation."),
            "thinking": {"type": "adaptive"},
            "max_tokens": MAX_TOKENS,
            **_stream_options(stream),
        }, stream=stream)
        if stream:
            assert_oai_stream_success(r)
            assert_stream_complete(r, msg="thinking.adaptive without effort")
        else:
            assert_oai_success(r)
        assert get_oai_content(r).strip(), "thinking.adaptive without effort: missing final answer"
        _assert_reasoning_usage_consistent(r, "thinking.adaptive without effort")

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_01_12_thinking_disabled_without_effort(self, stream):
        """M3.1 的 disabled 应返回 HTTP 400，与是否传 effort 无关。"""
        r = oai_chat({
            "messages": oai_simple_messages("Hi"),
            "thinking": {"type": "disabled"},
        }, stream=stream)
        assert_error(r, 400)
