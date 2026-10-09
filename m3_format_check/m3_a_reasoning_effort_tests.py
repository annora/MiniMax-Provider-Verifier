from statistics import median

import pytest

from helpers import *
from image_tools import make_png_base64


REASONING_EFFORT_ENUM = ("low", "medium", "high", "xhigh", "max")
COMPLEX_PROMPT = (
    "Let N be the number of positive divisors of 17017^17 that are "
    "congruent to 5 modulo 12. Find N modulo 1000 and explain your method."
)


def _reasoning_tokens(result: dict) -> int:
    """读取非流式或流式 usage 中的推理 token 数；缺失视为 0。"""
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


def _assert_usage_if_thinking(result: dict, context: str) -> bool:
    """只有实际返回思考内容时，才要求对应的 reasoning_tokens > 0。"""
    has_thinking = get_thinking_signals(result)["any"]
    if has_thinking:
        assert _reasoning_tokens(result) > 0, (
            f"{context}: 返回了思考内容，但 reasoning_tokens={_reasoning_tokens(result)}"
        )
    return has_thinking


def _assert_disabled(result: dict, context: str) -> None:
    assert_thinking_absent(result, msg=context)
    assert _reasoning_tokens(result) == 0, (
        f"{context}: disabled 时 reasoning_tokens 应为 0 或缺失，"
        f"实际为 {_reasoning_tokens(result)}"
    )


class TestReasoningEffort:
    @pytest.mark.parametrize("effort", REASONING_EFFORT_ENUM)
    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_01_01_valid_effort_accepted(self, effort, stream):
        """合法档位应正常响应；若返回思考，usage 应有正数计数。"""
        result = _request("What is 23 * 47? Explain briefly.", stream=stream,
                          reasoning_effort=effort)
        _assert_usage_if_thinking(result, f"effort={effort}, stream={stream}")

    def test_01_03_minimal_mapped_to_low(self):
        """兼容历史文档的 minimal → low 映射。"""
        result = _request("What is 12 + 30?", reasoning_effort="minimal")
        _assert_usage_if_thinking(result, "effort=minimal")

    def test_01_04_out_of_enum_ignored(self):
        """枚举外取值可忽略；兼容严格校验时的 400/422。"""
        result = oai_chat({
            "messages": oai_simple_messages("What is 2 + 2?"),
            "reasoning_effort": "ultra_super_max_xyz",
        })
        assert result["status"] in (200, 400, 422)
        if result["status"] == 200:
            assert_oai_success(result)
            _assert_usage_if_thinking(result, "effort=out_of_enum")

    def test_01_05_effort_with_thinking_adaptive(self):
        """effort 与 adaptive 同传时应正常响应。"""
        result = _request("Explain why the sky is blue, briefly.",
                          reasoning_effort="high", thinking={"type": "adaptive"})
        _assert_usage_if_thinking(result, "effort=high + adaptive")

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_01_06_effort_with_thinking_disabled(self, stream):
        """effort=high + disabled：HTTP 200，无推理内容，token 为 0 或缺失。"""
        result = _request("Hi", stream=stream, reasoning_effort="high",
                          thinking={"type": "disabled"})
        _assert_disabled(result, f"effort=high + disabled, stream={stream}")

    def test_01_07_reasoning_content_split(self):
        """实际思考时，应写入专门的 reasoning_content 并计入 usage。"""
        result = _request(COMPLEX_PROMPT, reasoning_effort="high")
        signals = get_thinking_signals(result)
        if not signals["any"]:
            pytest.skip("adaptive 本次未返回可见思考，无法校验字段位置")
        assert signals["reasoning_content"].strip(), (
            "有思考内容，但 reasoning_content 为空"
        )
        _assert_usage_if_thinking(result, "reasoning_content placement")

    @pytest.mark.parametrize("effort", ["low", "max"])
    def test_01_08_effort_stream_thinking(self, effort):
        """流式 low/max 应完整结束；有思考时校验计数。"""
        result = _request("Compute 17 * 19 step by step.", stream=True,
                          reasoning_effort=effort)
        _assert_usage_if_thinking(result, f"stream effort={effort}")

    @pytest.mark.parametrize("effort", ["low", "max"])
    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_01_09_effort_with_image(self, effort, stream):
        """图像输入下 low/max 均应正常响应。"""
        payload = {
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": make_png_base64(672, 672, r=255, g=0, b=0),
                }},
                {"type": "text", "text": "Identify the color and multiply its RGB red value by 19."},
            ]}],
            "reasoning_effort": effort,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        result = oai_chat(payload, stream=stream)
        if stream:
            assert_oai_stream_success(result)
        else:
            assert_oai_success(result)
        assert get_oai_content(result).strip()
        _assert_usage_if_thinking(result, f"image effort={effort}, stream={stream}")

    @pytest.mark.parametrize("effort", ["low", "high", "max"])
    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_01_10_complex_effort(self, effort, stream):
        """复杂题覆盖高低档位，以及流式和非流式计数。"""
        result = _request(COMPLEX_PROMPT, stream=stream, reasoning_effort=effort)
        _assert_usage_if_thinking(result, f"complex effort={effort}, stream={stream}")

    @pytest.mark.parametrize("thinking", [None, {"type": "adaptive"}],
                             ids=["default", "adaptive"])
    @pytest.mark.parametrize("prompt", ["Say hello.", COMPLEX_PROMPT],
                             ids=["short", "complex"])
    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_01_11_adaptive(self, thinking, prompt, stream):
        """默认与显式 adaptive 均允许按问题决定是否产生思考。"""
        fields = {} if thinking is None else {"thinking": thinking}
        result = _request(prompt, stream=stream, **fields)
        _assert_usage_if_thinking(result, f"adaptive prompt={prompt[:12]}, stream={stream}")

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_01_12_reasoning_tokens_when_present(self, stream):
        """专门校验：有思考内容时，reasoning_tokens 必须存在且大于 0。"""
        result = _request(COMPLEX_PROMPT, stream=stream, reasoning_effort="high")
        if not get_thinking_signals(result)["any"]:
            pytest.skip("本次没有可见思考，无法判定计数")
        assert _reasoning_tokens(result) > 0, (
            f"stream={stream}: 有思考内容，但 reasoning_tokens 缺失或为 0"
        )

    @pytest.mark.parametrize("prompt", ["Hi", COMPLEX_PROMPT],
                             ids=["short", "complex"])
    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_01_13_thinking_disabled(self, prompt, stream):
        """单独传 disabled：正常回答，不返回 reasoning_content，token 为 0 或缺失。"""
        result = _request(prompt, stream=stream, thinking={"type": "disabled"})
        _assert_disabled(result, f"disabled prompt={prompt[:12]}, stream={stream}")

    def test_01_14_effort_depth_repeated(self):
        """同题重复比较：max 的思考 token 中位数应高于 low。"""
        lengths = {"low": [], "max": []}
        for _ in range(5):
            for effort in ("low", "max"):
                result = _request(COMPLEX_PROMPT, reasoning_effort=effort)
                if _assert_usage_if_thinking(result, f"repeated effort={effort}"):
                    lengths[effort].append(_reasoning_tokens(result))
        if min(map(len, lengths.values())) < 3:
            pytest.skip(f"可见思考样本不足：{lengths}")
        assert median(lengths["max"]) > median(lengths["low"]), (
            f"max 思考长度未高于 low：{lengths}"
        )

    @pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
    def test_01_15_disabled_repeated(self, stream):
        """复杂题重复采样，检查 disabled 是否偶发返回推理。"""
        for sample in range(5):
            result = _request(COMPLEX_PROMPT, stream=stream,
                              thinking={"type": "disabled"})
            _assert_disabled(result, f"disabled sample={sample + 1}, stream={stream}")
