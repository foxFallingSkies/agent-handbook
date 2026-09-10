"""跑通手册第 1 章 §1.1 那封最难的邮件。

    python -m examples.ch01_email            # 离线（脚本化响应）
    python -m examples.ch01_email --live     # 真实调用（需要 ANTHROPIC_API_KEY）

这个脚本是全书的验收标准：**如果它跑不通，前四章就没有立住。**
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from handbook import LLM, Agent, Budget, ScriptedTransport, Trace  # noqa: E402
from handbook.errors import ConfirmationRequired  # noqa: E402
from handbook.tools.shop import Shop, build_registry  # noqa: E402

EMAIL = (
    "我上个月买的那个东西坏了，当时你们客服说过可以延保的，现在怎么算？"
    "另外我记得还有张优惠券没用，能一起处理吗？"
)

SYSTEM = """你是一家电商公司的客服 agent，负责处理客户来信。

<工作方式>
- 客户的描述通常是模糊的（"上个月那个东西"），你需要**自己查证**，不要猜。
- 客户声称"你们之前说过……"时，**去历史对话记录里核实**，不要直接采信也不要直接否认。
- 在向客户承诺任何处理方案之前，**先查政策**确认口径。
- 需要产生真实副作用的动作（如创建维修工单），确认清楚再调用。
</工作方式>

<回复要求>
- 用中文，称呼客户为"您"。
- 把客户问的每一件事都回答到，分条说明。
- 涉及日期、单号、金额时给出具体值，不要含糊。
- 如果有你无法确定的地方（例如客户说的"那个东西"可能指多笔订单中的哪一笔），
  在回复里明确指出并请客户确认。
</回复要求>"""


def verify_outcome(shop: Shop):
    """环境状态的验证器。

    第 1 章：**终止不能只靠模型说"我做完了"。**
    这里检查的是 shop 里真的多了一张工单，而不是 agent 声称开了一张。
    """

    def _verify(final_text: str) -> tuple[bool, str]:
        if not shop.tickets:
            return False, "没有创建任何维修工单"
        t = shop.tickets[-1]
        if t["order_id"] != "ord_20260803_a17c":
            return False, f"工单开在了错误的订单上：{t['order_id']}"
        if not t["warranty_claim"]:
            return False, "工单没有按保修处理，但该商品在保"
        if "AUTUMN-88" in final_text and "不能" not in final_text and "不可" not in final_text:
            return False, "回复里提到了优惠券但没说清它不能抵扣维修费"
        return True, f"工单 {t['id']} 已创建，订单与保修判定均正确"

    return _verify


async def main(live: bool) -> int:
    shop = Shop()
    customer_id = "cust_7f3a91e2"
    registry, _ids = build_registry(shop, customer_id)

    if live:
        from handbook import AnthropicTransport

        transport = AnthropicTransport()
        print("模式：LIVE（真实 API 调用）\n")
    else:
        from examples._scripted import SCRIPT

        transport = ScriptedTransport(SCRIPT)
        print("模式：OFFLINE（脚本化响应；token 数为本地估算，非实测）\n")

    llm = LLM(transport)
    agent = Agent(
        llm=llm,
        registry=registry,
        system=SYSTEM,
        budget=Budget(max_steps=15),
        verify=verify_outcome(shop),
    )

    trace = Trace("处理客户来信")
    outcome = await agent.run(EMAIL, trace=trace)

    # ---- 人在环上：高风险动作停下来等人 ----
    if outcome.status == "needs_confirmation":
        c: ConfirmationRequired = outcome.pending          # type: ignore[assignment]
        print("=" * 68)
        print("⏸  需要人工确认")
        print("=" * 68)
        # ⚠️ 展示的是**原始动作**，不是 agent 写的摘要。
        # OWASP ASI09 Human-Agent Trust Exploitation（2026 定稿版）。
        print(c.preview)
        print("=" * 68)
        print("[演示：自动批准]\n")

        from handbook.loop import digest_of

        agent.confirm(digest_of(c.tool_name, c.tool_args))
        outcome = await agent.run(trace=trace)             # 重入，不重发任务

    # ---- 结果 ----
    print("=" * 68)
    print("最终答复")
    print("=" * 68)
    print(outcome.final_text or "(无)")

    print("\n" + "=" * 68)
    print("执行轨迹")
    print("=" * 68)
    print(trace.render_tree())

    s = trace.summary()
    cost = llm.cost_report()
    ctx = agent.ctx.stats()

    print("\n" + "=" * 68)
    print("这一次运行的账")
    print("=" * 68)
    print(f"步数              {s['steps']}")
    print(f"模型调用          {s['llm_calls']}")
    print(f"工具调用          {s['tool_calls']}  {s['tool_breakdown']}")
    print(f"工具错误          {s['tool_errors']}")
    print(f"输入 token        {cost['input_tokens']:,}")
    print(f"输出 token        {cost['output_tokens']:,}")
    print(f"输入/输出比       {cost['io_ratio']}:1   (Manus 公开的生产值约 100:1)")
    print(f"缓存命中率        {cost['cache_hit_rate']:.1%}")
    print(f"成本              ${cost['total_cost_usd']:.6f}"
          + ("   ← 估算值" if not live else ""))
    print(f"上下文占用        {ctx['used_tokens']:,} tokens "
          f"({ctx['usage_ratio']:.1%} of window)")
    print(f"已折叠条目        {ctx['folded']} / {ctx['entries']}")

    print("\n" + "=" * 68)
    print("环境验证（不是问 agent，是查数据库）")
    print("=" * 68)
    print(f"结论: {'✅ 通过' if outcome.verified else '❌ 未通过'}   {outcome.verify_note}")
    print(f"工单表: {shop.tickets}")

    diag = registry.diagnose()
    if diag:
        print("\n工具诊断：")
        for d in diag:
            print("  -", d)

    out = trace.save("traces/ch01_email.json")
    print(f"\n完整 trace 已写入 {out}")

    return 0 if outcome.verified else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="真实 API 调用")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main(args.live)))
