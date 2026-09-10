"""参考实现的测试。

每个测试对应手册里的一条论断——**如果论断是对的，测试就该通过**。
这是本书和一般技术书的区别：书里的话是可执行的。
"""

from __future__ import annotations

import json

import asyncio
from pathlib import Path
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
    # ⚠️ 关键：错误条目也**带钥匙**。
    # 否则它不被折叠只是因为缺钥匙，这条断言对任何实现都成立——
    # 一个恒真的断言不是测试，是装饰。带上钥匙，唯一还能挡住它的
    # 就只剩 kind == "error" 这一条规则本身。
    ctx.append(Entry(role="user", content="调用失败：参数非法", kind="error",
                     step=1, tool_use_id="t1",
                     retrieval_key="workspace/err1.txt"))
    for s in range(2, 6):
        ctx.step = s
        ctx.append(Entry(role="user", content="ok" * 500, kind="observation",
                         step=s, tool_use_id=f"t{s}",
                         retrieval_key=f"workspace/r{s}.txt"))
    err = [e for e in ctx.entries if e.kind == "error"][0]
    assert not err.folded
    # 同龄、同样带钥匙的 observation 确实被折了——说明折叠机制本身在跑
    assert any(e.folded for e in ctx.entries if e.kind == "observation")


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


def test_pairing_survives_emergency_compaction():
    """配对不变量必须在**压缩之后**依然成立。

    上一条测试只走了不压缩的路径——而压缩恰恰是最容易把
    tool_use 和它的 tool_result 拆散的地方（一个被折进摘要，
    另一个留在原地）。真实的 400 错误几乎都发生在这里。
    """
    from handbook.context import MessageInvariantError

    ctx = ContextManager(window=4000, emergency_at=0.5)
    ctx.append(Entry(role="user", content="任务", kind="task", step=0))
    for step in range(1, 12):
        ctx.step = step
        ctx.append(Entry(role="assistant", content="", kind="action", step=step,
                         blocks=[{"type": "tool_use", "id": f"t{step}",
                                  "name": "f", "input": {"i": step}}]))
        ctx.append(Entry(role="user", content="x" * 800, kind="observation",
                         step=step, tool_use_id=f"t{step}"))

    assert any(ev["type"] == "compaction" for ev in ctx.events), \
        "这个用例本该触发压缩，没触发就等于什么都没测"

    msgs = ctx.to_messages()                       # 内部会做不变量检查
    uses = {b["id"] for m in msgs if isinstance(m["content"], list)
            for b in m["content"] if b.get("type") == "tool_use"}
    results = {b["tool_use_id"] for m in msgs if isinstance(m["content"], list)
               for b in m["content"] if b.get("type") == "tool_result"}
    assert results <= uses, f"出现了孤儿 tool_result：{results - uses}"
    # 角色必须交替
    roles = [m["role"] for m in msgs]
    assert all(a != b for a, b in zip(roles, roles[1:])), roles


def test_invariant_check_actually_rejects_a_broken_context():
    """不变量检查本身要有牙齿：喂一个坏上下文，它必须拒绝。"""
    from handbook.context import MessageInvariantError

    ctx = ContextManager()
    ctx.step = 1
    ctx.append(Entry(role="user", content="任务", kind="task", step=0))
    # 一个没有对应 tool_use 的 tool_result
    ctx.append(Entry(role="user", content="结果", kind="observation", step=1,
                     tool_use_id="不存在的id"))
    with pytest.raises(MessageInvariantError):
        ctx.to_messages()


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
    """§6：工具抖动闸——同工具同参数连续 N 次即判定卡死。

    ⚠️ 闸门是**返回** Outcome(status="stopped")，不是抛异常。
    抛异常会让调用方拿不到 trace、也无法从这个状态续跑，
    而且上下文里会留下一个没有 tool_result 的孤儿 tool_use。
    """
    spin = [
        Response(tool_uses=[ToolUse(id=f"t{i}", name="list_coupons", input={})],
                 stop_reason="tool_use", usage=Usage(output_tokens=10))
        for i in range(5)
    ]
    shop = Shop()
    reg, _ = build_registry(shop, CUSTOMER)
    agent = Agent(llm=LLM(ScriptedTransport(spin)), registry=reg, system="s",
                  budget=Budget(max_steps=10, same_call_limit=3))
    out = await agent.run("查优惠券")
    assert out.status == "stopped"
    assert "抖动" in out.stop_reason and "list_coupons" in out.stop_reason
    assert out.trace is not None                      # trace 关上了
    # 孤儿 tool_use 已被补上 tool_result——这个状态是可以继续跑的
    msgs = agent.ctx.to_messages()                    # 不抛 MessageInvariantError
    assert msgs


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

    # ⚠️ 这里**不**断言 final_text 里有 "AUTUMN-88"。
    # 离线模式下 final_text 整段来自 _scripted.py，断言它等于断言
    # 「我写的字符串里有我写的字符串」——恒真，且会在真跑时随口径变化而误红。
    # 能离线验的只有轨迹和环境状态，所以就只验这两样：
    names = [sp.name for sp in tr.spans
             if sp.kind == "tool_call" and sp.attrs.get("executed")]
    assert "list_coupons" in names, "回复优惠券问题前必须先去查优惠券，不能凭空作答"
    assert "get_policy" in names, "保修口径必须来自政策，不能自己编"
    assert names.index("check_warranty") < names.index("create_repair_ticket"), \
        "保修判定必须发生在开工单之前，否则工单上的 warranty_claim 是猜的"


# ------------------------------------------------- 声明了就得有人用

def test_oversized_tool_result_is_truncated_with_a_way_out():
    """§4.4：单次工具返回有硬上限，且截断提示要说明怎么缩小范围。

    ⚠️ `max_response_tokens` 曾经是个只在构造函数里出现、
    再也没被读过的字段。声明一个上限却不执行它，比没有上限更糟——
    因为看代码的人会以为有人在守着。
    """
    from handbook.tools import ToolRegistry, ToolSpec

    reg = ToolRegistry(max_response_tokens=50)
    reg.register(ToolSpec(
        name="dump_everything", description="返回一整张表",
        parameters={"type": "object", "properties": {}},
        fn=lambda: "行" * 5000,
    ))
    out = reg.call("dump_everything", {})
    assert not out.is_error                       # 截断不是错误
    assert len(out.text) < 5000
    assert "截断" in out.text
    assert "缩小查询范围" in out.text              # 给出下一步，不是只说"太长了"


def test_non_idempotent_tools_are_marked_unsafe_to_retry():
    """§4.6：重试一个非幂等工具 = 多一份副作用。"""
    shop = Shop()
    reg, _ = build_registry(shop, CUSTOMER)
    assert reg.is_retry_safe("search_orders") is True
    assert reg.is_retry_safe("create_repair_ticket") is False   # 重试就多一张工单
    assert reg.is_retry_safe("不存在的工具") is False


def test_cache_breakpoint_limit_is_checked_locally():
    """§2.5：缓存断点最多 4 个，超了 API 直接 400。

    离线模式永远不会暴露这个错——它只会在第一次真实调用时炸。
    所以在本地数一遍。
    """
    from handbook.errors import BadRequestError
    from handbook.transport import Request, _build_tools  # noqa: F401

    tools = [{"name": f"t{i}", "cache_control": {"type": "ephemeral"}}
             for i in range(4)]
    req = Request(model="m", system="s", tools=tools, messages=[],
                  cache_system=True, cache_tools=False)
    with pytest.raises(BadRequestError):
        _build_tools(req)                          # 4 个工具断点 + 1 个 system = 5


def test_scripted_transport_does_not_mutate_the_shared_script():
    """测试夹具里的共享可变状态：第二个测试会拿到第一个测试的用量。

    这类污染最阴险的地方是它看起来像「代码有随机性」，
    而实际上只是测试顺序变了。
    """
    from handbook.transport import Request

    script = [Response(text="ok", stop_reason="end_turn", usage=Usage(output_tokens=5))]
    before = script[0].usage.input_tokens

    async def _run():
        for _ in range(2):
            t = ScriptedTransport(list(script))
            await t.send(Request(model="m", system="s" * 4000, tools=[],
                                 messages=[{"role": "user", "content": "hi"}]))
    asyncio.run(_run())
    assert script[0].usage.input_tokens == before


def test_notes_and_recitation_are_usable_api():
    """§3.7 外置记事本 / §3.9 复述待办——两个第 3 章反复讲的技术。"""
    ctx = ContextManager(workspace="workspace/_test_notes")
    n = ctx.note("客户声称的延保承诺已在 2026-08-06 的对话里核实")
    assert n.retrieval_key                          # 记事本必须留得下钥匙
    assert Path(n.retrieval_key).exists()

    r = ctx.recite([("查订单", True), ("查保修", True), ("开工单", False)])
    assert "开工单" in r.content
    # 复述的意义在于把还没做的事**重新推到上下文末尾**
    assert ctx.entries[-1] is r


def test_capability_can_be_narrowed_after_reading_untrusted_content():
    """§11：读了不可信内容之后收窄可用工具集。

    遮蔽而不是删除——工具定义留在上下文里，KV-cache 前缀才不会作废。
    """
    shop = Shop()
    reg, _ = build_registry(shop, CUSTOMER)
    agent = Agent(llm=LLM(ScriptedTransport([])), registry=reg, system="s")

    agent.allow_only({"search_orders", "get_policy"})
    out = reg.call("create_repair_ticket",
                   {"order": "order-0803", "symptom": "x", "warranty_claim": True},
                   allowed=agent._allowed)
    assert out.is_error
    assert "不可用" in out.text
    assert "search_orders" in out.text              # 告诉它现在能用什么
    assert shop.tickets == []
    # 定义仍在，前缀没变
    assert "create_repair_ticket" in [t["name"] for t in reg.api_tools()]


def test_eval_runner_actually_calls_the_combinatorial_estimators():
    """⚠️ 这条测试的存在，是因为同一个 bug 犯过两次。

    第一次：`pass^k` 写成 `passed / n`（平均成功率）。
    第二次：改成调 `pass_pow_k()` 的那次编辑**静默失败了**——替换的锚点没匹配上，
    旧代码原样留着，而我在书里、台账里都写了"已修复"。

    两次都没有任何测试变红，因为没有任何测试碰过那一行。
    它最后是靠一次变异测试被抓出来的。

    所以这条断言直接钉住"报表里那个数是不是组合估计量"：
    n=3、c=2 时，平均成功率是 0.67，而 pass^3 是 0.00。
    """
    import inspect
    from evals import run as R

    # ① 两个估计量本身是对的
    assert R.pass_pow_k(3, 2, 3) == 0.0          # 三次里对两次 → 三次全对的概率是 0
    assert abs(R.pass_at_k(3, 2, 3) - 1.0) < 1e-9
    assert R.pass_pow_k(10, 8, 1) == 0.8         # k=1 时才等于 c/n

    # ② main() 真的在调它们，而不是被同名局部变量遮蔽
    # ⚠️ 要剥掉注释再查。第一版断言没剥，而我写的那句注释里恰好含有
    # 被禁的那个字符串——于是断言在**正确的代码上**也红。
    # 「检查器把自己的说明文字当成了被检查的内容」是同一类错的第三个形态。
    src = "\n".join(l.split("#")[0] for l in inspect.getsource(R.main).splitlines())
    assert "pass_pow_k(n, c" in src, "报表没有调用 pass_pow_k()——它又被遮蔽了"
    assert "pass_at_k(n, c" in src, "报表没有调用 pass_at_k()"
    assert "passed / n" not in src, "报表里还在用平均成功率冒充 pass^k"
