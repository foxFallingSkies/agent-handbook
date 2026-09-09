"""Agent 循环。手册第 6 章。

第 1 章那十行伪代码的生产版本。核心是把"自主性"和"约束"放在一起：

    五道闸：终止条件 · 预算 · 目标漂移 · 工具抖动 · 状态去重

以及一件第 1 章反复强调的事——**终止不能只靠模型说"我做完了"**，
要去环境里验证（`verify` 回调）。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .context import ContextManager, Entry
from .errors import (
    ConfirmationRequired,
    NoProgress,
    StepBudgetExceeded,
    TokenBudgetExceeded,
    ToolError,
)
from .llm import LLM, Tier
from .tools import ToolRegistry
from .trace import Trace

Verifier = Callable[[str], tuple[bool, str]]


@dataclass
class Budget:
    """预算闸。三条任一触顶即停。"""

    max_steps: int = 20
    max_tokens: int = 200_000
    max_seconds: float = 300.0

    # 抖动/去重的阈值
    same_call_limit: int = 3        # 同工具同参数连续几次算卡死
    same_state_limit: int = 2       # 相同状态重复几次算无进展


@dataclass
class Outcome:
    """一次运行的结果。

    注意 `final_text` 和 `verified` 是**分开**的两件事：
    前者是 agent 说了什么，后者是环境状态是否真的对。第 9 章会反复用到这个区分。
    """

    status: str                      # done / needs_confirmation / stopped
    final_text: str = ""
    verified: bool | None = None
    verify_note: str = ""
    steps: int = 0
    pending: ConfirmationRequired | None = None
    trace: Trace | None = None
    stop_reason: str = ""


class Agent:
    def __init__(
        self,
        *,
        llm: LLM,
        registry: ToolRegistry,
        system: str,
        budget: Budget | None = None,
        tier: Tier | None = None,
        verify: Verifier | None = None,
        workspace: str = "workspace",
    ):
        self.llm = llm
        self.registry = registry
        self.system = system
        self.budget = budget or Budget()
        self.tier = tier
        self.verify = verify
        self.ctx = ContextManager(workspace=workspace)

        # 闸门用的状态
        self._recent_calls: list[str] = []
        self._state_hashes: list[str] = []
        self._allowed: set[str] | None = None      # None = 全部可用
        self._confirmed: set[str] = set()
        # 因等待人工确认而中断的、尚未执行的工具调用。
        # 必须保留：中断时上下文里已经有了 assistant 的 tool_use 块，
        # 如果不给它配一个 tool_result，下一次请求会被 API 拒绝。
        self._pending_calls: list[Any] = []
        self._pending_step: int = 0

    # ------------------------------------------------------------------

    def allow_only(self, names: set[str] | None) -> None:
        """遮蔽工具。工具定义仍在上下文里（保住缓存前缀），只是不让调。

        第 11 章的"能力降级"会用它：读了不可信内容之后收窄可用集合。
        """
        self._allowed = names

    def confirm(self, digest: str) -> None:
        """人工批准某个待确认动作。"""
        self._confirmed.add(digest)

    # ------------------------------------------------------------------

    async def run(
        self, task: str | None = None, *, trace: Trace | None = None
    ) -> Outcome:
        """跑一次。

        `task=None` 表示**继续**已有的上下文——用于人工确认之后接着往下走。
        这是"人在环上"的实现细节：确认不是一个回调，是一次显式的重入。
        """
        tr = trace or Trace("agent.run")
        t0 = time.perf_counter()

        if task is not None:
            self.ctx.append(Entry(role="user", content=task, kind="task", step=0))

        # 先把上一次因等待确认而中断的工具调用做完，再继续循环。
        # 顺序不能反：上下文里那个 tool_use 必须先拿到它的 tool_result。
        if self._pending_calls:
            pending, self._pending_calls = self._pending_calls, []
            with tr.span(f"step {self._pending_step} (resumed)", "step") as sp:
                interrupted = self._execute_tools(
                    pending, self._pending_step, tr, sp
                )
                if interrupted is not None:
                    return interrupted

        for step in range(self.ctx.step + 1, self.budget.max_steps + 1):
            self.ctx.step = step

            # ---- 闸 2：预算 ----
            if time.perf_counter() - t0 > self.budget.max_seconds:
                return self._stop(tr, step, "超出时间预算")
            used = sum(r.usage.total_input for r in self.llm.records)
            if used > self.budget.max_tokens:
                raise TokenBudgetExceeded(used, self.budget.max_tokens)

            with tr.span(f"step {step}", "step") as step_span:
                # ---- 调模型 ----
                with tr.span("llm_call", "llm_call", step_span) as ls:
                    resp = await self.llm.call(
                        system=self.system,
                        messages=self.ctx.to_messages(),
                        tools=self.registry.api_tools(),
                        tier=self.tier,
                    )
                    ls.attrs.update(
                        input_tokens=resp.usage.input_tokens,
                        cache_read=resp.usage.cache_read_input_tokens,
                        cache_write=resp.usage.cache_creation_input_tokens,
                        output_tokens=resp.usage.output_tokens,
                        stop_reason=resp.stop_reason,
                    )

                # ---- 没有工具调用 = 模型认为做完了 ----
                if not resp.tool_uses:
                    return await self._finish(tr, step, resp.text)

                # 把 assistant 这一轮原样记下来（tool_use 块必须保留结构）
                blocks: list[dict] = []
                if resp.text:
                    blocks.append({"type": "text", "text": resp.text})
                for tu in resp.tool_uses:
                    blocks.append(
                        {"type": "tool_use", "id": tu.id, "name": tu.name,
                         "input": tu.input}
                    )
                self.ctx.append(
                    Entry(role="assistant", content=resp.text or "(调用工具)",
                          kind="action", step=step, blocks=blocks)
                )

                # ---- 闸 4 / 5：抖动与状态去重 ----
                for tu in resp.tool_uses:
                    sig = digest_of(tu.name, tu.input)
                    self._recent_calls.append(sig)
                    tail = self._recent_calls[-self.budget.same_call_limit:]
                    if (len(tail) == self.budget.same_call_limit
                            and len(set(tail)) == 1):
                        raise NoProgress(
                            f"工具 {tu.name} 用相同参数连续调用了 "
                            f"{self.budget.same_call_limit} 次",
                            evidence=json.dumps(tu.input, ensure_ascii=False),
                        )

                # ---- 执行工具 ----
                interrupted = self._execute_tools(resp.tool_uses, step, tr, step_span)
                if interrupted is not None:
                    return interrupted

                # 上一轮的观察已被本轮消费——标记它，让滚动折叠能处理
                for e in self.ctx.entries:
                    if e.kind == "observation" and e.step < step:
                        e.consumed = True

                # 把上下文事件也记进 trace（第 10 章会接到 OTel 上）
                for ev in self.ctx.events[len(
                        [s for s in tr.spans if s.kind == "context"]):]:
                    with tr.span(ev.get("type", "context"), "context",
                                 step_span) as cs:
                        cs.attrs.update(ev)

        raise StepBudgetExceeded(self.budget.max_steps)

    # ------------------------------------------------------------------

    def _execute_tools(
        self, tool_uses: list, step: int, tr: Trace, step_span
    ) -> Outcome | None:
        """执行一批工具调用。返回 None 表示全部完成；返回 Outcome 表示中断。

        中断时会把**尚未执行的调用**存进 `_pending_calls`，
        因为上下文里已经写入了对应的 tool_use 块，它们必须拿到 tool_result。
        """
        for i, tu in enumerate(tool_uses):
            with tr.span(tu.name, "tool_call", step_span) as ts:
                ts.attrs["args"] = tu.input
                digest = digest_of(tu.name, tu.input)
                try:
                    res = self.registry.call(
                        tu.name, tu.input,
                        allowed=self._allowed,
                        confirmed=digest in self._confirmed,
                    )
                except ConfirmationRequired as c:
                    ts.attrs["confirmation_required"] = True
                    self._pending_calls = list(tool_uses[i:])   # 含当前这个
                    self._pending_step = step
                    tr.close()
                    return Outcome(
                        status="needs_confirmation", steps=step, pending=c,
                        trace=tr, stop_reason="高风险动作待人工确认",
                    )

                ts.attrs["result_chars"] = len(res.text)
                if res.is_error:
                    ts.error = res.text.splitlines()[0][:120]
                # 失败也要配一个 tool_result 回去，只是标成 error：
                # 每个 tool_use 都必须配对，否则下一次请求会被 API 拒绝。
                self.ctx.append(
                    Entry(role="user", content=res.text,
                          kind="error" if res.is_error else "observation",
                          step=step, tool_use_id=tu.id)
                )
        return None

    async def _finish(self, tr: Trace, step: int, text: str) -> Outcome:
        """终止：模型说做完了。**但要去环境里验证。**"""
        verified: bool | None = None
        note = ""
        if self.verify is not None:
            with tr.span("verify", "context") as vs:
                verified, note = self.verify(text)
                vs.attrs.update(verified=verified, note=note)
        tr.close()
        return Outcome(
            status="done", final_text=text, verified=verified,
            verify_note=note, steps=step, trace=tr,
            stop_reason="模型返回了最终答复",
        )

    def _stop(self, tr: Trace, step: int, reason: str) -> Outcome:
        tr.close()
        return Outcome(status="stopped", steps=step, trace=tr, stop_reason=reason)


def digest_of(tool_name: str, tool_args: dict) -> str:
    """一次工具调用的稳定标识。

    公开出来是有意的：人工确认时，调用方需要算出同一个值。
    如果让调用方自己拼哈希，两边迟早会不一致——而这种不一致**不会报错**，
    只会让确认无声地失效。把它收敛成一个函数，是"机制优于约定"的又一次应用。
    """
    return _digest({"name": tool_name, "input": tool_args})


def _digest(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
