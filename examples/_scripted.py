"""离线演示用的模型响应脚本。

⚠️ **这些是手写的替身，不是真实 API 返回。**

它们存在的唯一理由：让没有 API key 的读者也能把整条链路跑起来——
工具层、上下文管理、闸门、trace、成本记账全部**真实执行**，
只有"模型说了什么"是预设的。

因此：
- **动作序列是真的**（它就是这个任务的合理解法）
- **token 数是估算的，不是实测的**，标记为 synthetic
- 想要真实数字，用 `--live`（需要 ANTHROPIC_API_KEY），
  或用 RecordingTransport 录一次到 fixtures/

真实运行时这段脚本会被整个绕开。
"""

from __future__ import annotations

from handbook.transport import Response, ToolUse, Usage

SYNTHETIC = True


def _usage(inp: int, out: int, cached: int = 0) -> Usage:
    return Usage(
        model="claude-sonnet-4-5-20250929(synthetic)",
        input_tokens=inp,
        output_tokens=out,
        cache_read_input_tokens=cached,
        latency_ms=0.0,
    )


def _tool(step: int, name: str, **kwargs) -> Response:
    return Response(
        text="",
        tool_uses=[ToolUse(id=f"toolu_{step:02d}", name=name, input=kwargs)],
        stop_reason="tool_use",
        usage=_usage(inp=0, out=90, cached=0),
    )


# 这条轨迹就是这封邮件的合理解法：
# 先定位是哪一单 → 核实延保承诺 → 确认保修状态 → 查优惠券 → 查政策 →
# 想开工单（触发人工确认）→ 确认后开单 → 给出答复
SCRIPT: list[Response] = [
    # 1. 客户说"上个月"，先把上个月的订单都拉出来
    _tool(1, "search_orders", since="2026-08-01"),

    # 2. 三笔订单里有两件小家电。客户说"坏了"+"客服说过延保"，
    #    先去对话记录里核实这个承诺是否存在
    _tool(2, "search_conversations", keyword="延保"),

    # 3. 承诺指向 CM-820（8月3日那单）。确认它的保修状态
    _tool(3, "check_warranty", order="order-0803"),

    # 4. 客户还问了优惠券
    _tool(4, "list_coupons"),

    # 5. 优惠券能不能抵维修费？查政策，别凭印象承诺
    _tool(5, "get_policy", topic="coupon"),

    # 6. 结论清楚了，要开维修工单——高风险动作，触发人工确认
    _tool(6, "create_repair_ticket",
          order="order-0803", symptom="咖啡机故障，客户反馈已损坏",
          warranty_claim=True),

    # 7. 人工确认后重入，工单开出来了，给客户最终答复
    Response(
        text=(
            "陈女士您好，帮您核实完了，分三件事回复：\n\n"
            "**1. 保修**：您 8 月 3 日购买的半自动咖啡机 CM-820（8 月 6 日签收）"
            "标准保修 12 个月，另有客服 A 在 8 月 6 日为您登记的 12 个月延长保修"
            "（登记号 EXT-2026-0806-77），两者合并计算，**保修有效期至 2028 年 8 月 6 日**，"
            "目前仍在保修期内。您记得的延保承诺属实，我们这边有记录。\n\n"
            "**2. 维修**：已为您创建维修工单 TCK-0001，按保修免费处理，"
            "稍后会安排上门取件，检测周期 3-5 个工作日。若检测判定为人为损坏"
            "（如进液、拆机），我们会先出报价单，经您确认后再维修，不会直接产生费用。\n\n"
            "**3. 优惠券**：您名下的 AUTUMN-88（满 300 减 88）有效期至 9 月 30 日，"
            "尚未使用。但按优惠券规则，**它只能用于新订单抵扣，不能抵扣维修费用**——"
            "维修工单不属于订单。这张券您可以在 9 月 30 日前用于购买咖啡豆等商品。\n\n"
            "另外您 8 月还有两笔订单（8/12 咖啡豆、8/25 厨房秤），如果坏的不是咖啡机，"
            "麻烦告诉我一声，我重新为您处理。"
        ),
        tool_uses=[],
        stop_reason="end_turn",
        usage=_usage(inp=0, out=420, cached=0),
    ),
]
