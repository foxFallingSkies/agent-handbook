"""最小评测 harness。手册第 8 章的可执行版本。

用法：
    python -m evals.run              # 离线（脚本化响应）
    python -m evals.run --k 5        # 每条任务跑 5 次，报 pass@k / pass^k
    python -m evals.run --live       # 真实调用

它只做四件事，但这四件是评测的骨架：
    1. 从 YAML 读 golden set
    2. 跑 k 次试验（agent 是非确定性的，一次说明不了任何事）
    3. **验证环境状态**，而不是 agent 的自述
    4. 同时报 pass@k 和 pass^k
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from math import comb
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from handbook import LLM, Agent, Budget, ScriptedTransport, Trace  # noqa: E402
from handbook.errors import ConfirmationRequired  # noqa: E402
from handbook.loop import digest_of  # noqa: E402
from handbook.tools.shop import Shop, build_registry  # noqa: E402

CUSTOMER = "cust_7f3a91e2"
CASES = Path(__file__).parent / "cases.yaml"


# ---------------------------------------------------------------------------
# 工具层评测：不需要模型
# ---------------------------------------------------------------------------

def run_tool_case(case: dict) -> tuple[bool, str]:
    shop = Shop()
    reg, _ = build_registry(shop, CUSTOMER)

    for step in case["steps"]:
        expect_raise = step.get("assert_raises")
        try:
            res = reg.call(step["call"], step.get("args", {}))
        except ConfirmationRequired:
            if expect_raise != "ConfirmationRequired":
                return False, "意外地要求了人工确认"
            if step.get("assert_no_side_effect") and shop.tickets:
                return False, "未确认却已经产生了副作用"
            continue
        if expect_raise:
            return False, f"期望抛出 {expect_raise}，但正常返回了"

        if step.get("assert_is_error") and not res.is_error:
            return False, "期望是一次错误返回，但成功了"
        for sub in step.get("assert_contains", []):
            if sub not in res.text:
                return False, f"返回里缺少 {sub!r}"
        for sub in step.get("assert_not_contains", []):
            if sub in res.text:
                return False, f"返回里出现了不该有的 {sub!r}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Agent 层评测
# ---------------------------------------------------------------------------

async def run_agent_case(case: dict, live: bool) -> tuple[bool, str]:
    from examples.ch01_email import SYSTEM

    shop = Shop()
    reg, _ = build_registry(shop, CUSTOMER)

    if live:
        from handbook import AnthropicTransport
        transport = AnthropicTransport()
    else:
        from examples._scripted import SCRIPT
        transport = ScriptedTransport(list(SCRIPT))

    agent = Agent(llm=LLM(transport), registry=reg, system=SYSTEM,
                  budget=Budget(max_steps=15))
    tr = Trace(case["id"])
    called: list[str] = []

    out = await agent.run(case["input"].strip(), trace=tr)
    if out.status == "needs_confirmation":
        agent.confirm(digest_of(out.pending.tool_name, out.pending.tool_args))
        out = await agent.run(trace=tr)

    # ⚠️ 只统计**真的执行了**的工具调用。
    # 被人工确认闸拦下的调用也会开 span，如果一并算进来，
    # must_not_call 这条专门用来测安全的断言就永远分不清
    # 「闸门守住了」和「闸门破了」——那等于这个维度失效。
    called = [s.name for s in tr.spans
              if s.kind == "tool_call" and s.attrs.get("executed")]
    exp = case["expect"]

    # ---- 关键必经动作（不查顺序，也不查次数） ----
    for name in exp.get("must_call", []):
        if name not in called:
            return False, f"没有调用必需的工具 {name}"
    for name in exp.get("must_not_call", []):
        if name in called:
            return False, f"调用了禁止的工具 {name}"

    # ---- 结果状态：查环境，不查 agent 的自述 ----
    oc = exp.get("outcome", {})
    if "tickets_count" in oc and len(shop.tickets) != oc["tickets_count"]:
        return False, f"工单数应为 {oc['tickets_count']}，实际 {len(shop.tickets)}"
    if shop.tickets:
        t = shop.tickets[-1]
        if "ticket_order_id" in oc and t["order_id"] != oc["ticket_order_id"]:
            return False, f"工单开在了错误的订单上：{t['order_id']}"
        if ("ticket_warranty_claim" in oc
                and t["warranty_claim"] != oc["ticket_warranty_claim"]):
            return False, "工单的保修判定不对"

    for sub in exp.get("must_mention", []):
        if sub not in out.final_text:
            return False, f"回复里没有提到 {sub}"
    return True, "ok"


# ---------------------------------------------------------------------------


def pass_at_k(n: int, c: int, k: int) -> float:
    """k 次里**至少一次**成功的概率（无偏估计）。

    = 1 - C(n-c, k) / C(n, k)   —— 即「k 次全部抽到失败」的补集
    """
    if n < k or k <= 0:
        return float("nan")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_pow_k(n: int, c: int, k: int) -> float:
    """k 次**全部**成功的概率（无偏估计）。

    = C(c, k) / C(n, k)

    ⚠️ 不是 c / n。c/n 是**平均成功率**，也就是插章 A 明令不要用的那个数。
    两者只在 k = 1 时相等——而初版恰好只跑过 k=1，所以这个错误被掩盖了。
    直观差别：c/n = 1/2 时，平均成功率是 0.50，而 pass^2 是 0.00。
    """
    if n < k or k <= 0:
        return float("nan")
    if c < k:
        return 0.0
    return comb(c, k) / comb(n, k)


async def main(k: int, live: bool) -> int:
    cases = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    print(f"golden set: {len(cases)} 条   trials per case: k={k}   "
          f"mode: {'LIVE' if live else 'OFFLINE'}\n")

    rows = []
    for case in cases:
        results: list[bool] = []
        notes: list[str] = []
        trials = 1 if case.get("kind") == "tool" else k   # 工具层是确定性的
        for _ in range(trials):
            if case.get("kind") == "tool":
                ok, note = run_tool_case(case)
            else:
                ok, note = await run_agent_case(case, live)
            results.append(ok)
            if not ok:
                notes.append(note)

        n = len(results)
        c = sum(results)
        # ⚠️ 必须调上面那两个函数。这里一度是局部变量 `pass_pow_k = passed / n`，
        # 它把同名函数遮蔽掉了 —— 表头写着 pass^k，打出来的是平均成功率。
        # 两个正确的估计量成了死代码，而且**没有任何测试会红**：
        # 这个 bug 是靠一次变异测试才被抓出来的，不是靠跑测试。
        kk = min(k, n)                            # 工具层 n=1，k 取不到那么大
        rows.append((case["id"], c, n,
                     pass_at_k(n, c, kk), pass_pow_k(n, c, kk), notes[:1]))

    if not rows:
        print("golden set 为空")
        return 1
    width = max(len(r[0]) for r in rows)
    print(f"{'case'.ljust(width)}  通过/试次  pass@k  pass^k  备注")
    print("-" * (width + 40))
    for cid, passed, n, pak, ppk, notes in rows:
        flag = "✅" if passed == n else ("⚠️ " if passed else "❌")
        note = notes[0] if notes else ""
        print(f"{cid.ljust(width)}  {passed}/{n}       "
              f"{pak:.2f}    {ppk:.2f}   {flag} {note}")

    all_pass = all(r[1] == r[2] for r in rows)
    print("\n" + ("全部通过" if all_pass else "存在失败用例"))
    if not live:
        print("⚠️  离线模式下模型响应是脚本化的，pass^k 恒为 1.0——"
              "它验证的是**管道**，不是模型。真实的 pass^k 要用 --live 跑。")
    return 0 if all_pass else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=1, help="每条 agent 用例跑几次")
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(main(args.k, args.live)))
