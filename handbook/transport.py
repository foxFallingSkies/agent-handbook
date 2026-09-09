"""传输层：真正和模型说话的地方。

手册第 2 章的 `llm.py` 之所以能被讲清楚，是因为它把"怎么发请求"隔离在了
这一层。这里有三个实现：

- `AnthropicTransport`  真实调用。
- `ReplayTransport`     回放录制好的响应。**离线、确定性**，测试和 CI 用它。
- `RecordingTransport`  包住真实调用并把响应录到 fixtures/，供 Replay 使用。

这个三件套解决一个真实问题：**一本教材里的代码必须能在没有 API key 的
情况下被读者跑起来**，同时又不能是假的。录制-回放让两者都成立。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .errors import (
    BadRequestError,
    ContextOverflowError,
    RateLimitError,
    ServerError,
    TransportError,
)

# ---------------------------------------------------------------------------
# 定价（美元 / 每百万 token）
#
# 这四个数字决定了第 2 章 §2.5 的全部论证。注意 cache_write 比普通 input 更贵
# ——缓存不是白拿的，**短会话可能不划算**。
#
# 数字会变，以官方定价页为准；这里的默认值用于演示与本地估算。
# ---------------------------------------------------------------------------
PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-5": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,   # 约 1.25 × input
        "cache_read": 0.30,    # 约 0.10 × input —— 这就是"十倍差距"的来源
    },
    "claude-haiku-4-5": {
        "input": 1.00,
        "output": 5.00,
        "cache_write": 1.25,
        "cache_read": 0.10,
    },
}

# 缓存的三个硬约束（第 2 章 §2.5 的读者最容易在这里翻车）
CACHE_MIN_TOKENS = 1024        # 少于这个长度的前缀不会被缓存
CACHE_TTL_SECONDS = 300        # 默认 5 分钟；超过就要重新写入
CACHE_MAX_BREAKPOINTS = 4      # 一次请求最多标几个缓存断点


def default_pricing(model: str) -> dict[str, float]:
    for key, table in PRICING.items():
        if model.startswith(key):
            return table
    return PRICING["claude-sonnet-4-5"]


@dataclass
class Request:
    """一次模型调用的完整描述。

    刻意把 `system` / `tools` / `messages` 分开——它们在真实 API 里就是
    三个独立参数，而且**缓存断点是标在前两者上的**。
    第 2 章早期版本把它们拼成一个字符串，那是错的：拼起来就没法标断点了。
    """

    model: str
    system: str
    tools: list[dict]
    messages: list[dict]
    max_tokens: int = 4096
    temperature: float = 0.0
    cache_system: bool = True      # 在 system 末尾标缓存断点
    cache_tools: bool = True       # 在工具定义末尾标缓存断点

    def fingerprint(self) -> str:
        """用于 Replay 查找的稳定指纹。

        注意 sort_keys=True —— 第 2 章 §2.5.3 规则二：序列化必须确定性，
        否则同一个请求两次会算出不同的指纹。这里犯错，回放就会随机失效。
        """
        payload = json.dumps(
            {
                "model": self.model,
                "system": self.system,
                "tools": self.tools,
                "messages": self.messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class Usage:
    """一次调用的用量。字段名对齐 Anthropic API 的 usage 对象。"""

    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0   # 写入缓存的 token（有溢价）
    cache_read_input_tokens: int = 0       # 命中缓存的 token（便宜十倍）
    latency_ms: float = 0.0

    @property
    def total_input(self) -> int:
        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    @property
    def cache_hit_rate(self) -> float:
        """第 2 章称之为"生产 agent 的头号指标"。"""
        return self.cache_read_input_tokens / max(self.total_input, 1)

    def cost_usd(self, pricing: dict[str, float] | None = None) -> float:
        p = pricing or default_pricing(self.model)
        return (
            self.input_tokens * p["input"]
            + self.output_tokens * p["output"]
            + self.cache_creation_input_tokens * p["cache_write"]
            + self.cache_read_input_tokens * p["cache_read"]
        ) / 1_000_000


@dataclass
class ToolUse:
    id: str
    name: str
    input: dict


@dataclass
class Response:
    """模型的一次响应。

    `text` 和 `tool_uses` 并存，因为模型可以在同一轮里既说话又调工具。
    """

    text: str = ""
    tool_uses: list[ToolUse] = field(default_factory=list)
    stop_reason: str = "end_turn"      # end_turn / max_tokens / tool_use
    usage: Usage = field(default_factory=Usage)
    raw: dict = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        return self.stop_reason == "max_tokens"


class Transport(Protocol):
    async def send(self, req: Request) -> Response: ...


# ---------------------------------------------------------------------------


def _build_system_blocks(req: Request) -> list[dict]:
    """把 system 渲染成 block 列表，并在末尾标缓存断点。

    这是第 2 章 §2.5.3 "规则三：需要时显式标记缓存断点"的实际实现。
    """
    block: dict[str, Any] = {"type": "text", "text": req.system}
    if req.cache_system:
        block["cache_control"] = {"type": "ephemeral"}
    return [block]


def _build_tools(req: Request) -> list[dict]:
    """工具定义。缓存断点标在**最后一个**工具上——前缀匹配是累积的，
    标在末尾等于把"system + 全部工具"这一整段都纳入缓存。
    """
    if not req.tools:
        return []
    tools = [dict(t) for t in req.tools]
    if req.cache_tools:
        tools[-1]["cache_control"] = {"type": "ephemeral"}
    return tools


class AnthropicTransport:
    """真实调用。"""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:  # pragma: no cover
            raise TransportError(
                "需要安装 anthropic SDK：pip install anthropic"
            ) from e

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise TransportError(
                "未找到 API key。设置环境变量 ANTHROPIC_API_KEY，"
                "或改用 ReplayTransport 离线运行（见 README）。"
            )
        kwargs: dict[str, Any] = {"api_key": key}
        url = base_url or os.environ.get("ANTHROPIC_BASE_URL")
        if url:
            kwargs["base_url"] = url
        self._client = AsyncAnthropic(**kwargs)

    async def send(self, req: Request) -> Response:
        import time

        from anthropic import (
            APIStatusError,
            APITimeoutError,
            InternalServerError,
            RateLimitError as SDKRateLimit,
        )

        t0 = time.perf_counter()
        try:
            msg = await self._client.messages.create(
                model=req.model,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                system=_build_system_blocks(req),
                tools=_build_tools(req) or None,
                messages=req.messages,
            )
        except SDKRateLimit as e:
            retry_after = None
            headers = getattr(getattr(e, "response", None), "headers", {}) or {}
            if "retry-after" in headers:
                try:
                    retry_after = float(headers["retry-after"])
                except (TypeError, ValueError):
                    retry_after = None
            raise RateLimitError(str(e), retry_after=retry_after) from e
        except (InternalServerError, APITimeoutError) as e:
            raise ServerError(str(e)) from e
        except APIStatusError as e:
            text = str(e)
            # 上下文超长要走压缩路径，不是修参数重试——所以单独成类
            if "context" in text.lower() and (
                "long" in text.lower() or "exceed" in text.lower()
            ):
                raise ContextOverflowError(text) from e
            raise BadRequestError(text) from e

        latency = (time.perf_counter() - t0) * 1000
        return _from_sdk_message(msg, latency)


def _from_sdk_message(msg: Any, latency_ms: float) -> Response:
    """把 SDK 的返回对象转成我们的 Response。"""
    text_parts: list[str] = []
    tool_uses: list[ToolUse] = []

    for block in msg.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_parts.append(block.text)
        elif btype == "tool_use":
            tool_uses.append(
                ToolUse(id=block.id, name=block.name, input=dict(block.input))
            )

    u = msg.usage
    usage = Usage(
        model=msg.model,
        input_tokens=getattr(u, "input_tokens", 0) or 0,
        output_tokens=getattr(u, "output_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        latency_ms=latency_ms,
    )
    return Response(
        text="".join(text_parts),
        tool_uses=tool_uses,
        stop_reason=msg.stop_reason or "end_turn",
        usage=usage,
        raw={"id": msg.id},
    )


# ---------------------------------------------------------------------------


class ReplayTransport:
    """从 fixtures/ 回放录制好的响应。

    离线、确定性、零成本。测试、CI、以及没有 API key 的读者用它。
    找不到对应指纹时**明确失败**——不要伪造一个响应蒙混过去。
    """

    def __init__(self, fixture_dir: str | Path = "fixtures"):
        self.dir = Path(fixture_dir)
        self.calls: list[str] = []

    async def send(self, req: Request) -> Response:
        fp = req.fingerprint()
        self.calls.append(fp)
        path = self.dir / f"{fp}.json"
        if not path.exists():
            raise TransportError(
                f"没有找到指纹 {fp} 对应的录制响应。\n"
                f"请用 RecordingTransport 录一次（需要 API key），或检查请求是否变了。\n"
                f"提示：请求内容变一个字符，指纹就会变——这正是 KV-cache 的行为。"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        return Response(
            text=data.get("text", ""),
            tool_uses=[ToolUse(**t) for t in data.get("tool_uses", [])],
            stop_reason=data.get("stop_reason", "end_turn"),
            usage=Usage(**data.get("usage", {})),
            raw=data.get("raw", {}),
        )


class RecordingTransport:
    """包住一个真实 transport，把响应录到 fixtures/。"""

    def __init__(self, inner: Transport, fixture_dir: str | Path = "fixtures"):
        self.inner = inner
        self.dir = Path(fixture_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    async def send(self, req: Request) -> Response:
        resp = await self.inner.send(req)
        payload = {
            "text": resp.text,
            "tool_uses": [
                {"id": t.id, "name": t.name, "input": t.input} for t in resp.tool_uses
            ],
            "stop_reason": resp.stop_reason,
            "usage": resp.usage.__dict__,
            "raw": resp.raw,
        }
        (self.dir / f"{req.fingerprint()}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return resp


class ScriptedTransport:
    """按顺序返回预设的响应。

    用于单元测试里精确构造某种情形（例如"第一次返回非法 JSON，第二次修好"），
    以及在没有 API key 时演示完整链路。

    `estimate_usage=True` 时会**用本地估算填充输入 token**，并模拟前缀缓存的行为：
    与上一次请求相同的那段前缀算作命中。这样离线运行也能产出有意义（但仍是估算）
    的成本报表。⚠️ 估算值不等于实测值，报表里会标出来。
    """

    def __init__(self, responses: list[Response], *, estimate_usage: bool = True):
        self._responses = list(responses)
        self.requests: list[Request] = []
        self.estimate_usage = estimate_usage
        self._prev_prefix: str | None = None

    async def send(self, req: Request) -> Response:
        self.requests.append(req)
        if not self._responses:
            raise TransportError(
                f"ScriptedTransport 的预设响应已用尽（已消费 {len(self.requests)} 次）。"
                "说明 agent 走的步数比脚本预期的多——去看 trace 找出多出来的那一步。"
            )
        await asyncio.sleep(0)     # 让出事件循环，暴露并发问题
        resp = self._responses.pop(0)

        if self.estimate_usage:
            from . import tokens as tk

            prefix = req.system + json.dumps(req.tools, sort_keys=True, ensure_ascii=False)
            prefix_tokens = tk.estimate(prefix)
            msg_tokens = tk.estimate_messages(req.messages)

            # 前缀没变 → 算作缓存命中；变了 → 算作写入
            if self._prev_prefix == prefix and prefix_tokens >= CACHE_MIN_TOKENS:
                cached, created, fresh = prefix_tokens, 0, msg_tokens
            elif prefix_tokens >= CACHE_MIN_TOKENS:
                cached, created, fresh = 0, prefix_tokens, msg_tokens
            else:
                # 前缀太短，静默不缓存——这是真实 API 的行为
                cached, created, fresh = 0, 0, prefix_tokens + msg_tokens
            self._prev_prefix = prefix

            resp.usage.input_tokens = fresh
            resp.usage.cache_read_input_tokens = cached
            resp.usage.cache_creation_input_tokens = created
            if not resp.usage.model:
                resp.usage.model = req.model
        return resp
