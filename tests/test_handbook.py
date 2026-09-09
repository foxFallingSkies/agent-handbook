"""参考实现的测试。

每个测试对应手册里的一条论断——**如果论断是对的，测试就该通过**。
这是本书和一般技术书的区别：书里的话是可执行的。
"""

from __future__ import annotations

import json

import pytest

from handbook import LLM, Agent, Budget, ScriptedTransport, Trace
from handbook.context import ContextManager, Entry
from handbook.errors import ConfirmationRequired, OutputFormatError, ToolError
from handbook.loop import digest_of
from handbook.tools.shop import Shop, build_registry
from handbook.transport import Response, ToolUse, Usage
from handbook import tokens as tk

CUSTOMER = "cust_7f3a91e2"


# ---------------------------------------------------------------- 第 2 章


def test_token_estimate_reflects_format_cost():
    """§2.3.3：同样的信息，不同格式的 token 数不同。

    这是"选格式的判据是模型好不好写"那条论断的成本一面。
    """
    data = {"order": "order-0803", "status": "delivered", "total": 1899}
    sizes = tk.format_comparison(data)
    assert sizes["json_indented"] > sizes["json_compact"]
    assert sizes["plain_lines"] < sizes["json_indented"]


def test_cjk_costs_more_tokens_than_latin():
    """中文的 token 密度高于英文——这会直接改变中文场景的成本估算。"""
    assert tk.estimate("保修状态确认") > tk.estimate("warranty ok")


@pytest.mark.asyncio
async def test_structured_output_repair_chain():
    """§2.3.2：校验失败时回传**具体的**校验错误，模型据此自修。"""
    from pydantic import BaseModel

    class Ticket(BaseModel):
        order: str
        urgent: bool

    transport = ScriptedTransport(
        [
            Response(text='{"order": "order-0803", "urgent": "maybe"}', # 类型错，且 pydantic 不会强转
                     stop_reason="end_turn", usage=Usage(output_tokens=20)),
            Response(text='{"order": "order-0803", "urgent": true}',    # 修好了
                     stop_reason="end_turn", usage=Usage(output_tokens=20)),
        ]
    )
    llm = LLM(transport)
    obj = await llm.structured(Ticket, system="s", messages=[{"role": "user", "content": "x"}])
    assert obj.urgent is True

    # 第二次请求里必须带上**字段级**的错误说明，而不是"格式错了"
    repair_msg = transport.requests[1].messages[-1]["content"]
    assert "urgent" in repair_msg
    assert "格式错了" not in repair_msg


@pytest.mark.asyncio
async def test_structured_output_fails_loudly():
    """§2.3.2：自修耗尽后**显式失败**，不能静默返回默认值。"""
    from pydantic import BaseModel

    class Ticket(BaseModel):
        order: str

    bad = Response(text="{}", stop_reason="end_turn", usage=Usage(output_tokens=5))
    llm = LLM(ScriptedTransport([bad, bad, bad]), max_repairs=2)
    with pytest.raises(OutputFormatError):
        await llm.structured(Ticket, system="s",
                             messages=[{"role": "user", "content": "x"}])


@pytest.mark.asyncio
async def test_stable_prefix_hits_cache():
    """§2.5：前缀不变则命中缓存；前缀里塞进动态内容则命中率归零。"""
    script = [Response(text="ok", stop_reason="end_turn",
                       usage=Usage(output_tokens=5)) for _ in range(6)]

    stable = LLM(ScriptedTransport(script[:3]))
    big_system = "你是客服。" * 400          # 超过最小可缓存长度
    for i in range(3):
        await stable.call(system=big_system,
                          messages=[{"role": "user", "content": f"q{i}"}])
    assert stable.cost_report()["cache_hit_rate"] > 0.5

    churning = LLM(ScriptedTransport(script[3:]))
    for i in range(3):
        # 只多了一个时间戳——这一个改动就让整段前缀作废
        await churning.call(system=f"[时间 2026-09-09 10:0{i}:00]" + big_system,
                            messages=[{"role": "user", "content": f"q{i}"}])
    assert churning.cost_report()["cache_hit_rate"] == 0.0


# ---------------------------------------------------------------- 第 3 章


def test_rolling_fold_is_continuous_not_threshold():
    """§3.3：压缩是持续的纪律，不是"快满了才做"。

    上下文占用远低于任何阈值时，滚动折叠也应该已经在工作。
    """
    ctx = ContextManager(window=200_000, fold_after_steps=2)
    for step in range(1, 6):
        ctx.step = step
        ctx.append(Entry(role="user", content="x" * 2000, kind="observation",
                         step=step, tool_use_id=f"t{step}",
                         retrieval_key=f"workspace/r{step}.txt"))
    assert ctx.usage_ratio() < 0.05                  # 远没满
    assert any(e.folded for e in ctx.entries)        # 但已经折叠过了
    assert any(ev["type"] == "rolling_fold" for ev in ctx.events)


def test_errors_are_never_folded():
    """§3.10：失败记录是"此路不通"的证据，擦掉它模型会重犯。"""
    ctx = ContextManager(fold_after_steps=1)
    ctx.step = 1
    ctx.append(Entry(role="user", content="调用失败：参数非法", kind="error",
                     step=1, tool_use_id="t1"))
    for s in range(2, 6):
        ctx.step = s
        ctx.append(Entry(role="user", content="ok", kind="observation",
                         step=s, tool_use_id=f"t{s}", consumed=True))
    err = [e for e in ctx.entries if e.kind == "error"][0]
    assert not err.folded


def test_every_tool_use_gets_a_tool_result():
    """API 硬要求：每个 tool_use 都必须配一个 tool_result，失败也要。"""
    ctx = ContextManager()
    ctx.step = 1
    ctx.append(Entry(role="user", content="任务", kind="task", step=0))
    ctx.append(Entry(role="assistant", content="", kind="action", step=1,
                     blocks=[{"type": "tool_use", "id": "t1",
                              "name": "f", "input": {}}]))
    ctx.append(Entry(role="user", content="出错了", kind="error", step=1,
                     tool_use_id="t1"))

    msgs = ctx.to_messages()
    uses = [b for m in msgs if isinstance(m["content"], list)
            for b in m["content"] if b.get("type") == "tool_use"]
    results = [b for m in msgs if isinstance(m["content"], list)
               for b in m["content"] if b.get("type") == "tool_result"]
    assert len(uses) == len(results) == 1
    assert results[0]["is_error"] is True


# ---------------------------------------------------------------- 第 4 章


def test_tool_errors_are_written_for_the_model():
    """§4.5：错误消息是 prompt——必须具体、可行动。"""
    shop = Shop()
    reg, _ = build_registry(shop, CUSTOMER)
    out = reg.call("search_orders", {"since": "上个月"})
    assert "YYYY-MM-DD" in out          # 说清期望格式
    assert "2026-09-09" in out          # 给出今天的日期，让它能自己算
    assert "Error" not in out


def test_unknown_param_is_rejected_with_the_valid_list():
    shop = Shop()
    reg, _ = build_registry(shop, CUSTOMER)
    out = reg.call("search_orders", {"month": "2026-08"})
    assert "month" in out and "since" in out and "keyword" in out


def test_scope_is_enforced_in_the_tool_not_the_prompt():
    """§4.10 补充：身份是闭包捕获的，模型没有渠道去查别人的数据。"""
    shop = Shop()
    reg, ids = build_registry(shop, CUSTOMER)
    # 别人的订单根本没有被登记进这个客户的 ID 映射表
    with pytest.raises(ToolError):
        ids.resolve("order-0901")
    listed = reg.call("search_orders", {})
    assert "别人的订单" not in listed


def test_semantic_ids_not_uuids():
    """§4.3：对模型暴露语义化短 ID，降低幻觉。"""
    shop = Shop()
    reg, _ = build_registry(shop, CUSTOMER)
    out = reg.call("search_orders", {})
    assert "order-0803" in out
    assert "ord_20260803_a17c" not in out


def test_high_risk_tool_requires_confirmation():
    shop = Shop()
    reg, _ = build_registry(shop, CUSTOMER)
    with pytest.raises(ConfirmationRequired) as ei:
        reg.call("create_repair_ticket",
                 {"order": "order-0803", "symptom": "坏了", "warranty_claim": True})
    assert shop.tickets == []                       # 没确认就没有副作用
    # 预览必须是**原始动作**，不是摘要（OWASP ASI09）
    assert "order-0803" in ei.value.preview
    assert "warranty_claim" in ei.value.preview


def test_confirmation_args_are_not_swallowed_by_exception_args():
    """回归测试：`args` 是 BaseException 的保留属性。

    曾经写成 `self.args = args`，Python 静默把 dict 转成 tuple(keys)，
    导致确认哈希对不上、人工批准无声失效。**这个 bug 不报错。**
    """
    shop = Shop()
    reg, _ = build_registry(shop, CUSTOMER)
    args = {"order": "order-0803", "symptom": "坏了", "warranty_claim": True}
    with pytest.raises(ConfirmationRequired) as ei:
        reg.call("create_repair_ticket", args)
    assert ei.value.tool_args == args               # 值还在，不是键名列表
    assert isinstance(ei.value.tool_args, dict)


def test_diagnose_flags_repeated_identical_calls():
    """§4.9：重复调用 → 参数设计有问题。把诊断规则做成代码。"""
    shop = Shop()
    reg, _ = build_registry(shop, CUSTOMER)
    for _ in range(3):
        reg.call("list_coupons", {})
    assert any("重复调用" in d for d in reg.diagnose())


# ---------------------------------------------------------------- 第 6 章


@pytest.mark.asyncio
async def test_thrash_gate_stops_identical_repeated_calls():
    """§6：工具抖动闸——同工具同参数连续 N 次即判定卡死。"""
    from handbook.errors import NoProgress

    spin = [
        Response(tool_uses=[ToolUse(id=f"t{i}", name="list_coupons", input={})],
                 stop_reason="tool_use", usage=Usage(output_tokens=10))
        for i in range(5)
    ]
    shop = Shop()
    reg, _ = build_registry(shop, CUSTOMER)
    agent = Agent(llm=LLM(ScriptedTransport(spin)), registry=reg, system="s",
                  budget=Budget(max_steps=10, same_call_limit=3))
    with pytest.raises(NoProgress):
        await agent.run("查优惠券")


@pytest.mark.asyncio
async def test_end_to_end_email_is_verified_against_the_environment():
    """全书的验收标准：那封邮件能被跑完，且**环境状态**是对的。"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from examples._scripted import SCRIPT
    from examples.ch01_email import EMAIL, SYSTEM, verify_outcome

    shop = Shop()
    reg, _ = build_registry(shop, CUSTOMER)
    agent = Agent(llm=LLM(ScriptedTransport(SCRIPT)), registry=reg, system=SYSTEM,
                  budget=Budget(max_steps=15), verify=verify_outcome(shop))

    tr = Trace("test")
    out = await agent.run(EMAIL, trace=tr)
    assert out.status == "needs_confirmation"          # 高风险动作停下来了
    assert shop.tickets == []                          # 停下时没有副作用

    agent.confirm(digest_of(out.pending.tool_name, out.pending.tool_args))
    out = await agent.run(trace=tr)

    assert out.status == "done"
    assert out.verified is True, out.verify_note
    assert len(shop.tickets) == 1
    assert shop.tickets[0]["order_id"] == "ord_20260803_a17c"
    assert shop.tickets[0]["warranty_claim"] is True
    # 回复必须说清优惠券不能抵维修费
    assert "AUTUMN-88" in out.final_text
