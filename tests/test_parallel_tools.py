"""并行工具调用。手册第 5 章 §5.9。

⚠️ 这个文件里每条测试针对的都是一个**静默错误**——
并行做错了不会抛异常、不会变红，只会给出一个看起来完全正常的错结果。
这正是它值得单独测的原因。
"""

from __future__ import annotations

import asyncio
import time

import pytest

from handbook.loop import Agent
from handbook.tools import ToolRegistry, ToolSpec
from handbook.transport import Response, ScriptedTransport, ToolUse, Usage
from handbook import LLM, Budget


def _registry(delay_map: dict[str, float] | None = None) -> ToolRegistry:
    delays = delay_map or {}
    reg = ToolRegistry()

    def mk(name: str, read_only: bool):
        def fn():
            if delays.get(name):
                time.sleep(delays[name])
            return f"{name} 的结果"
        reg.register(ToolSpec(
            name=name, description=name,
            parameters={"type": "object", "properties": {}},
            fn=fn, read_only=read_only,
        ))

    for n in ("read_a", "read_b", "read_c"):
        mk(n, True)
    mk("write_x", False)
    return reg


def _agent(reg: ToolRegistry, tool_uses: list[ToolUse], *, parallel=True) -> Agent:
    script = [
        Response(text="", stop_reason="tool_use", tool_uses=tool_uses,
                 usage=Usage(output_tokens=10)),
        Response(text="做完了", stop_reason="end_turn", usage=Usage(output_tokens=5)),
    ]
    return Agent(llm=LLM(ScriptedTransport(script)), registry=reg, system="s",
                 budget=Budget(max_steps=4), parallel_tools=parallel)


def _observations(agent: Agent) -> list[str]:
    return [e.content for e in agent.ctx.entries if e.tool_use_id]


# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_results_are_written_back_in_model_order_not_completion_order():
    """⚠️ **这是这个文件里最重要的一条。**

    并发执行，但写回上下文必须按**模型给出的顺序**，不是完成顺序。

    这里让第一个调用慢、最后一个快，所以按完成顺序写回会得到倒序。
    按完成顺序写回的后果不是"报错"，是：
      ① 同样的输入每次跑出来的上下文不一样 → 不可复现
      ② 从第一个乱序的位置起，KV-cache 前缀全部失配（§2.5）
    并发省下的那点时间，会连本带利还给缓存未命中。
    """
    reg = _registry({"read_a": 0.15, "read_b": 0.05, "read_c": 0.0})
    tus = [ToolUse(id="t1", name="read_a", input={}),
           ToolUse(id="t2", name="read_b", input={}),
           ToolUse(id="t3", name="read_c", input={})]
    ag = _agent(reg, tus)
    await ag.run("干活")
    assert _observations(ag) == ["read_a 的结果", "read_b 的结果", "read_c 的结果"]


@pytest.mark.asyncio
async def test_read_only_calls_actually_run_concurrently():
    """三个各 sleep 0.1s 的只读调用，总耗时必须显著小于 0.3s。

    ⚠️ 判据不是"跑完了"——串行也跑得完。判据是**墙上时钟**。
    """
    reg = _registry({"read_a": 0.1, "read_b": 0.1, "read_c": 0.1})
    tus = [ToolUse(id=f"t{i}", name=n, input={})
           for i, n in enumerate(("read_a", "read_b", "read_c"), 1)]
    t0 = time.perf_counter()
    await _agent(reg, tus).run("干活")
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.22, f"耗时 {elapsed:.3f}s，看起来还是串行的"


@pytest.mark.asyncio
async def test_grouping_stops_at_the_first_side_effecting_call():
    """⚠️ 只取**开头连续**的只读段，不是把所有只读的挑出来一起跑。

    [读, 写, 读] 里第二个读很可能是想看那次写的结果。
    把两个读并在一起先跑，它读到的是写之前的世界——
    不报错，结果看起来完全正常。这是并行最典型的静默错误。
    """
    reg = _registry({"read_a": 0.1, "read_c": 0.1})
    tus = [ToolUse(id="t1", name="read_a", input={}),
           ToolUse(id="t2", name="write_x", input={}),
           ToolUse(id="t3", name="read_c", input={})]
    ag = _agent(reg, tus)
    t0 = time.perf_counter()
    await ag.run("干活")
    elapsed = time.perf_counter() - t0

    # 组只有 1 个元素 → 不并发 → 两次 0.1s 的 sleep 必须都串行发生
    assert elapsed >= 0.2, f"耗时 {elapsed:.3f}s，说明跨过写操作去并行了"
    assert _observations(ag) == ["read_a 的结果", "write_x 的结果", "read_c 的结果"]


@pytest.mark.asyncio
async def test_unmarked_tools_fall_back_to_serial():
    """⚠️ fail-closed：`read_only` 默认 False，忘了标就串行。

    一个「忘了标」的工具应该退化成慢，而不是退化成错。
    """
    reg = ToolRegistry()
    for n in ("u1", "u2"):
        reg.register(ToolSpec(name=n, description=n,
                              parameters={"type": "object", "properties": {}},
                              fn=(lambda nm: (lambda: (time.sleep(0.1), f"{nm} ok")[1]))(n)))
    assert reg.tools["u1"].read_only is False

    tus = [ToolUse(id="t1", name="u1", input={}), ToolUse(id="t2", name="u2", input={})]
    t0 = time.perf_counter()
    await _agent(reg, tus).run("干活")
    assert time.perf_counter() - t0 >= 0.2


@pytest.mark.asyncio
async def test_parallel_can_be_turned_off():
    reg = _registry({"read_a": 0.1, "read_b": 0.1})
    tus = [ToolUse(id="t1", name="read_a", input={}),
           ToolUse(id="t2", name="read_b", input={})]
    t0 = time.perf_counter()
    await _agent(reg, tus, parallel=False).run("干活")
    assert time.perf_counter() - t0 >= 0.2


@pytest.mark.asyncio
async def test_a_tool_error_still_produces_a_result():
    """工具内部抛异常 → registry 已经把它转成 is_error 的结果，一个都不会少。"""
    reg = _registry()
    reg.register(ToolSpec(
        name="read_boom", description="炸",
        parameters={"type": "object", "properties": {}},
        fn=lambda: (_ for _ in ()).throw(RuntimeError("底层挂了")),
        read_only=True,
    ))
    tus = [ToolUse(id="t1", name="read_a", input={}),
           ToolUse(id="t2", name="read_boom", input={}),
           ToolUse(id="t3", name="read_c", input={})]
    ag = _agent(reg, tus)
    await ag.run("干活")
    assert [e.tool_use_id for e in ag.ctx.entries if e.tool_use_id] == ["t1", "t2", "t3"]


@pytest.mark.asyncio
async def test_a_contradictory_marking_does_not_orphan_a_tool_use():
    """⚠️ 上一版这条测试是**假绿**，而它暴露的问题比测试本身更值得记。

    原来的写法是让工具内部抛 RuntimeError，然后断言三个 tool_use 都拿到结果。
    变异测试显示：把并行路径里那段异常处理整个删掉，它**依然全绿**。

    原因是 `registry.call` 已经把工具内部的异常转成了 `is_error` 的正常返回值，
    所以那段异常处理根本没被走到——**我写了一段自己以为在保护什么的死代码**。

    真正能走到它的是 `registry.call` 会**重新抛出**的那两个：
    `ConfirmationRequired` 和 `ToolNotAllowed`。前者在这里尤其现实：
    一个同时标了 `read_only=True` 和 `requires_confirmation=True` 的工具
    是**自相矛盾的标注**（要确认说明它有副作用），而并行路径必须扛得住
    别人标错——少一个 `tool_result`，下一次请求会被 API 直接拒绝
    （"tool_use ids were found without tool_result blocks"），
    而这时上下文已经写坏了，连续跑都续不了。
    """
    reg = _registry()
    reg.register(ToolSpec(
        name="read_weird", description="矛盾的标注",
        parameters={"type": "object", "properties": {}},
        fn=lambda: "不该被执行",
        read_only=True, requires_confirmation=True,     # ← 自相矛盾
    ))
    tus = [ToolUse(id="t1", name="read_a", input={}),
           ToolUse(id="t2", name="read_weird", input={}),
           ToolUse(id="t3", name="read_c", input={})]
    ag = _agent(reg, tus)
    await ag.run("干活")
    ids = [e.tool_use_id for e in ag.ctx.entries if e.tool_use_id]
    assert ids == ["t1", "t2", "t3"], f"有 tool_use 变成了孤儿：{ids}"
