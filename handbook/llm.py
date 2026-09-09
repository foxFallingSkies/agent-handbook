"""模型调用层。手册第 2 章。

这一层的职责：把"一个非确定性、按量计价、会以多种方式失败的外部服务"
包装成上层可以放心调用的东西。

它做四件事，每件都对应第 2 章的一节：
1. 结构化输出与自修链      §2.3
2. 成本记账与缓存可见性    §2.4 §2.5
3. 按异常类型分流的重试    §2.6
4. 模型分级路由            §2.4（便宜模型做简单事）
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .errors import (
    BadRequestError,
    OutputFormatError,
    OutputTruncatedError,
    RateLimitError,
    RetryExhausted,
    ServerError,
)
from .transport import Request, Response, Transport, Usage, default_pricing

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------


@dataclass
class Tier:
    """一档模型。

    第 2 章 §2.4：**用便宜模型做简单推理和工具调用，贵模型做复杂逻辑**，
    是比缓存更直接的省钱手段。把它做成一等概念，而不是调用点的一个字符串。
    """

    name: str
    model: str
    max_tokens: int = 4096


FAST = Tier("fast", "claude-haiku-4-5-20251001", max_tokens=2048)
SMART = Tier("smart", "claude-sonnet-4-5-20250929", max_tokens=8192)


@dataclass
class CallRecord:
    """一次调用的完整记录，供成本报表和 trace 使用。"""

    tier: str
    usage: Usage
    repairs: int = 0          # 为了拿到合法输出，自修了几次
    retries: int = 0          # 因为传输失败，重试了几次


class LLM:
    """模型调用的统一入口。

    用法：

        llm = LLM(transport)
        resp = await llm.call(system="...", messages=[...], tools=[...])
        obj  = await llm.structured(MySchema, system="...", messages=[...])
        print(llm.cost_report())
    """

    def __init__(
        self,
        transport: Transport,
        *,
        default_tier: Tier = SMART,
        fallback_tier: Tier | None = FAST,
        max_retries: int = 3,
        max_repairs: int = 2,
    ):
        self.transport = transport
        self.default_tier = default_tier
        self.fallback_tier = fallback_tier
        self.max_retries = max_retries
        self.max_repairs = max_repairs
        self.records: list[CallRecord] = []

    # ------------------------------------------------------------------
    # 基础调用
    # ------------------------------------------------------------------

    async def call(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        tier: Tier | None = None,
        temperature: float = 0.0,
    ) -> Response:
        """一次普通调用（可能带工具）。"""
        tier = tier or self.default_tier
        req = Request(
            model=tier.model,
            system=system,
            tools=tools or [],
            messages=messages,
            max_tokens=tier.max_tokens,
            temperature=temperature,
        )
        resp, retries = await self._send_with_retry(req, tier)
        self.records.append(CallRecord(tier=tier.name, usage=resp.usage, retries=retries))
        return resp

    # ------------------------------------------------------------------
    # 结构化输出
    # ------------------------------------------------------------------

    async def structured(
        self,
        schema: type[T],
        *,
        system: str,
        messages: list[dict],
        tier: Tier | None = None,
    ) -> T:
        """要求模型按 schema 输出，并在不合法时带着**具体的校验错误**让它自修。

        第 2 章 §2.3.2：回传"格式错了，重来"是无效的；必须告诉它
        错在哪个字段、期望什么、以及怎么改。
        """
        tier = tier or self.default_tier
        schema_json = json.dumps(
            schema.model_json_schema(), ensure_ascii=False, indent=2
        )
        sys_with_schema = (
            f"{system}\n\n"
            f"你必须只输出一个 JSON 对象，符合以下 JSON Schema，"
            f"不要输出任何其它内容（不要 markdown 代码块，不要解释）：\n"
            f"{schema_json}"
        )

        attempt_messages = list(messages)
        last_raw = ""
        total_retries = 0

        for repair in range(self.max_repairs + 1):
            req = Request(
                model=tier.model,
                system=sys_with_schema,
                tools=[],
                messages=attempt_messages,
                max_tokens=tier.max_tokens,
                temperature=0.0,
            )
            resp, retries = await self._send_with_retry(req, tier)
            total_retries += retries
            self.records.append(
                CallRecord(tier=tier.name, usage=resp.usage, repairs=repair,
                           retries=retries)
            )

            # 截断的 JSON 看起来像"格式错误"，但处置方式完全不同：
            # 它写对了，只是没写完。丢进自修链只会让它试图"修复"正确的内容。
            if resp.truncated:
                raise OutputTruncatedError(
                    f"输出被 max_tokens={tier.max_tokens} 截断。"
                    f"提高上限，或让模型分段输出。"
                )

            last_raw = _strip_fence(resp.text)
            try:
                return schema.model_validate_json(last_raw)
            except ValidationError as e:
                if repair == self.max_repairs:
                    raise OutputFormatError(
                        f"{self.max_repairs} 次自修后输出仍不符合 schema",
                        last_raw=last_raw,
                        attempts=self.max_repairs + 1,
                    ) from e
                attempt_messages = attempt_messages + [
                    {"role": "assistant", "content": last_raw},
                    {"role": "user", "content": _repair_prompt(e)},
                ]

        raise AssertionError("unreachable")   # 循环内必定 return 或 raise

    # ------------------------------------------------------------------
    # 重试
    # ------------------------------------------------------------------

    async def _send_with_retry(self, req: Request, tier: Tier) -> tuple[Response, int]:
        """按异常类型分流的重试。第 2 章 §2.6。"""
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                return await self.transport.send(req), attempt
            except RateLimitError as e:
                last_error = e
                # 服务端给了建议就听它的；否则指数退避 + 抖动。
                # 抖动不是可选项：没有它，并发请求会同步重试形成脉冲。
                delay = e.retry_after if e.retry_after else (2**attempt)
                await asyncio.sleep(delay + random.random())
            except ServerError as e:
                last_error = e
                if attempt == self.max_retries - 1:
                    break
                await asyncio.sleep((2**attempt) + random.random() * 0.5)
            except BadRequestError:
                # 4xx 不重试——重试只会重复失败
                raise

        # 重试耗尽：降级到便宜/更小的模型再试一次，而不是直接失败
        if self.fallback_tier and tier.name != self.fallback_tier.name:
            fb = Request(
                model=self.fallback_tier.model,
                system=req.system,
                tools=req.tools,
                messages=req.messages,
                max_tokens=min(req.max_tokens, self.fallback_tier.max_tokens),
                temperature=req.temperature,
            )
            try:
                return await self.transport.send(fb), self.max_retries
            except Exception:
                pass

        raise RetryExhausted(
            f"{self.max_retries} 次重试后仍失败，且降级路径不可用"
        ) from last_error

    # ------------------------------------------------------------------
    # 报表
    # ------------------------------------------------------------------

    def cost_report(self) -> dict[str, Any]:
        """第 2 章的核心指标都在这里。

        `cache_hit_rate` 是 Manus 所说的"生产 agent 头号指标"；
        `io_ratio` 用来和他们公开的 100:1 对标。
        """
        if not self.records:
            return {"calls": 0}

        total_in = sum(r.usage.total_input for r in self.records)
        total_out = sum(r.usage.output_tokens for r in self.records)
        cache_read = sum(r.usage.cache_read_input_tokens for r in self.records)
        cache_write = sum(r.usage.cache_creation_input_tokens for r in self.records)
        cost = sum(
            r.usage.cost_usd(default_pricing(r.usage.model or ""))
            for r in self.records
        )

        by_tier: dict[str, dict[str, float]] = {}
        for r in self.records:
            t = by_tier.setdefault(r.tier, {"calls": 0, "cost": 0.0, "input": 0})
            t["calls"] += 1
            t["cost"] += r.usage.cost_usd(default_pricing(r.usage.model or ""))
            t["input"] += r.usage.total_input

        return {
            "calls": len(self.records),
            "input_tokens": total_in,
            "output_tokens": total_out,
            "cache_read_tokens": cache_read,
            "cache_write_tokens": cache_write,
            "cache_hit_rate": round(cache_read / max(total_in, 1), 4),
            "io_ratio": round(total_in / max(total_out, 1), 1),
            "total_cost_usd": round(cost, 6),
            "repairs": sum(r.repairs for r in self.records),
            "retries": sum(r.retries for r in self.records),
            "by_tier": by_tier,
        }


# ---------------------------------------------------------------------------


def _strip_fence(text: str) -> str:
    """模型有时会把 JSON 包在 markdown 代码块里，尽管你让它别这么做。

    与其在 prompt 里反复叮嘱，不如在代码里容错——
    这是"机制优于劝告"的一个小号版本。
    """
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def _repair_prompt(err: ValidationError) -> str:
    """把 pydantic 的校验错误翻译成模型能照着改的指令。

    这个函数是第 2 章 §2.3.2 那条原则的全部落点：
    **回传的必须是"哪个字段、错在哪、期望什么"，不是"格式错了"。**
    """
    lines = []
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"]) or "(根对象)"
        lines.append(f"- 字段 `{loc}`：{e['msg']}（收到的值：{e.get('input')!r}）")
    detail = "\n".join(lines)
    return (
        f"上一次输出不符合要求，具体问题如下：\n{detail}\n\n"
        f"请只修正这些字段，重新输出完整的 JSON 对象，不要输出其它内容。"
    )
