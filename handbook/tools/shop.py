"""电商客服 agent 的工具集。手册第 1 章 §1.1 那封邮件用的就是这一套。

设计上刻意体现第 4 章的几条原则，读代码时可以对照：

- 按工作流切分，不按 CRUD 切：没有 list_orders + get_order + list_items，
  只有一个 search_orders，它一次返回 agent 判断「是哪一单」所需的全部信息。
- 语义化标识符：对模型暴露 order-0803 而不是 ord_20260803_a17c。
- 鉴权在工具层：每个工具都绑定 customer_id，模型无法查到别人的数据。
  这是第 4 章漏掉的一块——风险分级管的是「能不能做」，scope 管的是「对谁做」。
- 错误消息即 prompt：所有 ToolError 都带 hint。
- 高风险动作要确认：创建工单会真的产生副作用，requires_confirmation=True。
"""

from __future__ import annotations

import json
from calendar import monthrange
from datetime import date
from pathlib import Path
from typing import Any

from ..errors import ToolError
from . import SemanticIdMap, ToolRegistry, ToolSpec, truncate_items

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "shop.json"


class Shop:
    """一个虚构的电商后端。

    它同时扮演两个角色：工具的数据源，以及评测时的环境状态检查点——
    第 9 章要验证的 outcome 不是 agent 说了什么，而是这里的状态变成了什么。
    """

    def __init__(self, path: Path = DATA_PATH):
        self.data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        self.today = date.fromisoformat(self.data["today"])
        self.tickets: list[dict] = []   # 运行时副作用记在这里，供评测检查

    def orders_of(self, customer_id: str) -> list[dict]:
        return [o for o in self.data["orders"] if o["customer_id"] == customer_id]

    def conversations_of(self, customer_id: str) -> list[dict]:
        return [c for c in self.data["conversations"]
                if c["customer_id"] == customer_id]

    def coupons_of(self, customer_id: str) -> list[dict]:
        return [c for c in self.data["coupons"] if c["customer_id"] == customer_id]

    def warranty_of(self, customer_id: str) -> list[dict]:
        return [w for w in self.data["warranty_registrations"]
                if w["customer_id"] == customer_id]


# 工具描述单独定义。它们是写给模型看的 prompt，值得和主 prompt 一样的投入，
# 放在这里也便于第 9 章拿它们做 A/B 对照。
DESC_SEARCH_ORDERS = (
    "搜索当前客户的订单。返回订单编号、下单与签收日期、状态、金额，"
    "以及每件商品的名称、品类和标准保修月数——足以判断「是哪一单」，"
    "不需要再调别的工具取详情。\n"
    "since 用于限定「从哪天之后下的单」，格式 YYYY-MM-DD；"
    "keyword 会在商品名称里做包含匹配。两者都可省略。"
)

DESC_SEARCH_CONV = (
    "在当前客户的历史客服对话记录里做关键词搜索，返回命中的完整对话。"
    "用于核实客户声称的「你们之前说过……」。"
    "keyword 建议用具体名词，例如：延保、保修、换新。"
)

DESC_CHECK_WARRANTY = (
    "查询某一笔订单的保修状态。会把标准保修和已登记的延长保修合并计算，"
    "返回是否仍在保、保修到期日，以及延保登记记录（含登记人和登记时间）。\n"
    "order 参数用 search_orders 返回的订单编号，例如 order-0803。"
)

DESC_LIST_COUPONS = (
    "列出当前客户名下的全部优惠券，含面额、最低消费、有效期、是否已使用、"
    "是否已过期，以及适用品类。"
)

DESC_GET_POLICY = (
    "查询公司政策原文。可查的主题：warranty（保修）、coupon（优惠券）、"
    "repair（维修流程）。\n"
    "在向客户承诺任何处理方案之前，应当先查政策确认口径。"
)

DESC_CREATE_TICKET = (
    "为某一笔订单创建维修工单并安排上门取件。这会产生真实的副作用，"
    "只在确认了保修状态、且客户明确希望维修时才调用。\n"
    "warranty_claim 表示是否按保修处理（免费）；若为 false 则走付费维修报价流程。"
)



def _find_order(shop, order_id: str, customer_id: str, verb: str = "访问该订单") -> dict:
    """按 id 找订单，找不到就给一条**能照着改**的错误。

    ⚠️ 原来这里是 `next(x for x in ... )`。订单号写错时它抛 StopIteration——
    一个既不会被 ToolResult 捕获、也不告诉模型任何信息的异常。
    工具层的每一条失败路径都要落在 ToolError 上，否则第 4 章那句
    「错误消息是 prompt」就只对成功路径成立。
    """
    o = next((x for x in shop.data["orders"] if x["id"] == order_id), None)
    if o is None:
        mine = [x["id"] for x in shop.data["orders"]
                if x["customer_id"] == customer_id]
        raise ToolError(
            f"没有编号为 {order_id!r} 的订单。",
            hint=f"该客户名下的订单：{', '.join(mine) or '无'}。"
                 f"可以先用 search_orders 确认订单号。",
        )
    if o["customer_id"] != customer_id:              # 纵深防御
        raise ToolError(f"无权{verb}。", hint="只能操作当前客户名下的订单。")
    return o


def _add_months(start: date, months: int) -> date:
    """从 start 往后推 months 个月，落在月末时向下取该月最后一天。

    ⚠️ 原来是 `min(delivered.day, 28)`。那会把 8 月 31 日签收的订单
    算成次年 8 月 28 日到期——**悄悄少赔三天保修**。
    没人会发现，直到有个客户在第 29 天来报修。
    """
    y = start.year + (start.month - 1 + months) // 12
    m = (start.month - 1 + months) % 12 + 1
    return date(y, m, min(start.day, monthrange(y, m)[1]))


def build_registry(shop: Shop, customer_id: str) -> tuple[ToolRegistry, SemanticIdMap]:
    """为某一位客户构造工具集。

    注意 customer_id 是闭包捕获的，不是工具参数。
    如果把它做成参数，模型就有可能传别人的 ID——那不是 prompt 注入，
    是工具契约设计错误。能从上下文确定的身份，永远不要交给模型传。
    """
    reg = ToolRegistry()
    ids = SemanticIdMap("order")

    # 预登记，让语义 ID 稳定且可读：order-0803 / order-0812 / order-0825
    for o in sorted(shop.orders_of(customer_id), key=lambda x: x["placed_at"]):
        ids.shorten(o["id"], o["placed_at"].replace("-", "")[4:])

    # ------------------------------------------------------------------

    def search_orders(since: str | None = None, keyword: str | None = None) -> str:
        rows = shop.orders_of(customer_id)
        if since:
            try:
                cutoff = date.fromisoformat(since)
            except ValueError:
                raise ToolError(
                    f"参数 since 的格式应为 YYYY-MM-DD，收到 {since!r}。",
                    hint=(f"今天是 {shop.today.isoformat()}。若要查上个月，"
                          f"用 {shop.today.replace(day=1).isoformat()} 之前的日期。"),
                )
            rows = [r for r in rows if date.fromisoformat(r["placed_at"]) >= cutoff]
        if keyword:
            kw = keyword.lower()
            rows = [r for r in rows
                    if any(kw in it["name"].lower() for it in r["items"])]
        if not rows:
            all_orders = shop.orders_of(customer_id)
            earliest = min((o["placed_at"] for o in all_orders), default="-")
            return (f"没有匹配的订单。\n该客户共有 {len(all_orders)} 笔订单，"
                    f"最早 {earliest}。可以放宽 since 或去掉 keyword 再试。")

        rows, note = truncate_items(
            sorted(rows, key=lambda x: x["placed_at"], reverse=True),
            page_size=10,
            filter_hint="keyword=咖啡机",
        )
        out = [
            {
                "order": ids.shorten(r["id"]),
                "placed_at": r["placed_at"],
                "delivered_at": r["delivered_at"],
                "status": r["status"],
                "total_cny": r["total_cny"],
                "items": [
                    {"name": it["name"], "category": it["category"],
                     "warranty_months": it["warranty_months"]}
                    for it in r["items"]
                ],
            }
            for r in rows
        ]
        body = json.dumps(out, ensure_ascii=False, sort_keys=True)
        return body + (f"\n{note}" if note else "")

    def search_conversations(keyword: str) -> str:
        kw = keyword.lower()
        hits = [c for c in shop.conversations_of(customer_id)
                if kw in c["transcript"].lower()]
        if not hits:
            dates = [c["date"] for c in shop.conversations_of(customer_id)]
            return (f"没有包含 {keyword!r} 的历史对话。\n"
                    f"该客户共有 {len(dates)} 次对话记录，"
                    f"日期：{', '.join(dates) or '无'}。"
                    f"可以换一个关键词，例如：保修、延保、换新。")
        return json.dumps(
            [{"date": c["date"], "agent": c["agent"], "transcript": c["transcript"]}
             for c in hits],
            ensure_ascii=False, sort_keys=True,
        )

    def check_warranty(order: str) -> str:
        real = ids.resolve(order)
        o = _find_order(shop, real, customer_id)

        # 没签收就没有保修起算点。这不是"没保修"，是"还没开始"——
        # 两者对客户的答复完全不同，所以不能糊成一个 False。
        if not o.get("delivered_at"):
            raise ToolError(
                f"订单 {order} 尚未签收（当前状态：{o['status']}），保修期还没有起算。",
                hint="保修从签收日算起。请先确认订单是否已送达，"
                     "或者告知客户保修将在签收后开始。",
            )

        exts = [w for w in shop.warranty_of(customer_id)
                if w["order_id"] == real and w["status"] == "active"]
        extra = sum(w["extra_months"] for w in exts)
        delivered = date.fromisoformat(o["delivered_at"])

        # ⚠️ 逐件算，不是只算 items[0]。
        # 一张订单里既可能有 12 个月保修的机器，也可能有 0 个月的耗材；
        # 只看第一件会把整单的保修判定押在数组顺序上——
        # 而数组顺序不是业务事实。
        lines = []
        for item in o["items"]:
            base = item["warranty_months"]
            if base == 0:
                lines.append({"sku": item["sku"], "item": item["name"],
                              "covered": False, "reason": "该品类不提供保修"})
                continue
            end = _add_months(delivered, base + extra)
            lines.append({
                "sku": item["sku"], "item": item["name"],
                "covered": shop.today <= end,
                "base_months": base, "extra_months": extra,
                "covered_until": end.isoformat(),
            })

        return json.dumps(
            {
                "order": order,
                "delivered_at": o["delivered_at"],
                "items": lines,
                "any_covered": any(x["covered"] for x in lines),
                "extension_records": [
                    {"id": w["id"], "registered_at": w["registered_at"],
                     "registered_by": w["registered_by"]} for w in exts
                ],
            },
            ensure_ascii=False, sort_keys=True,
        )

    def list_coupons() -> str:
        rows = [
            {
                "code": c["code"],
                "value_cny": c["value_cny"],
                "min_spend_cny": c["min_spend_cny"],
                "valid_until": c["valid_until"],
                "used": c["used"],
                "expired": date.fromisoformat(c["valid_until"]) < shop.today,
                "applicable_to": c["applicable_to"],
                "note": c["note"],
            }
            for c in shop.coupons_of(customer_id)
        ]
        if not rows:
            return "该客户名下没有优惠券。"
        return json.dumps(rows, ensure_ascii=False, sort_keys=True)

    def read_stashed(path: str) -> str:
        """把折叠到磁盘上的原文读回来。

        ⚠️ 这个工具是「压缩要留钥匙」这条原则的**另一半**。
        只写钥匙不给取回工具，那把钥匙就配不上任何一把锁——
        折叠仍然是有损的，只是损得比较体面。
        """
        from pathlib import Path as _P
        f = _P(path)
        # 路径必须落在 workspace 内：path 来自模型，是可控输入
        root = _P("workspace").resolve()
        try:
            resolved = f.resolve()
            resolved.relative_to(root)
        except (ValueError, OSError):
            raise ToolError(
                f"路径 {path!r} 不在允许的工作目录内。",
                hint="只能读取折叠记录里给出的、位于 workspace/ 下的路径。",
            )
        if not resolved.exists():
            raise ToolError(
                f"文件 {path!r} 不存在。",
                hint="请核对折叠记录里给出的路径；如果它已被清理，就重新调用原来那个工具。",
            )
        return resolved.read_text(encoding="utf-8")

    def get_policy(topic: str) -> str:
        p = shop.data["policies"].get(topic)
        if p is None:
            raise ToolError(
                f"没有名为 {topic!r} 的政策条目。",
                hint=f"可查询的政策：{', '.join(sorted(shop.data['policies']))}。",
            )
        return json.dumps(p, ensure_ascii=False, sort_keys=True)

    def create_repair_ticket(order: str, symptom: str, warranty_claim: bool) -> str:
        real = ids.resolve(order)
        o = _find_order(shop, real, customer_id,
                        verb="对该订单创建工单")
        ticket = {
            "id": f"TCK-{len(shop.tickets) + 1:04d}",
            "order_id": real,
            "customer_id": customer_id,
            "symptom": symptom,
            "warranty_claim": warranty_claim,
            "created_at": shop.today.isoformat(),
        }
        shop.tickets.append(ticket)      # ← 真实副作用，第 9 章检查的就是它
        return json.dumps(
            {"created": True, "ticket_id": ticket["id"],
             "next": "已安排上门取件，检测周期 3-5 个工作日。"},
            ensure_ascii=False, sort_keys=True,
        )

    # ------------------------------------------------------------------

    reg.register(ToolSpec(
        name="search_orders",
        description=DESC_SEARCH_ORDERS,
        parameters={
            "type": "object",
            "properties": {
                "since": {
                    "type": "string",
                    "description": "只看这一天（含）之后下的订单，格式 YYYY-MM-DD",
                },
                "keyword": {
                    "type": "string",
                    "description": "商品名称关键词，例如：咖啡机",
                },
            },
            "required": [],
        },
        fn=search_orders, risk="low", scope="customer",
    ))

    reg.register(ToolSpec(
        name="search_conversations",
        description=DESC_SEARCH_CONV,
        parameters={
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "搜索关键词"},
            },
            "required": ["keyword"],
        },
        fn=search_conversations, risk="low", scope="customer",
    ))

    reg.register(ToolSpec(
        name="check_warranty",
        description=DESC_CHECK_WARRANTY,
        parameters={
            "type": "object",
            "properties": {
                "order": {"type": "string",
                          "description": "订单编号，形如 order-0803"},
            },
            "required": ["order"],
        },
        fn=check_warranty, risk="low", scope="customer",
    ))

    reg.register(ToolSpec(
        name="list_coupons",
        description=DESC_LIST_COUPONS,
        parameters={"type": "object", "properties": {}, "required": []},
        fn=list_coupons, risk="low", scope="customer",
    ))

    reg.register(ToolSpec(
        name="read_stashed",
        description=(
            "读回一段被折叠到磁盘上的工具结果。当你在历史里看到"
            "「第 N 步的工具结果已折叠。原文在 <路径>」这样的记录，"
            "而你现在又需要那份原文时，用这个工具把它取回来。\n"
            "path 参数就填那条记录里给出的路径。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string",
                         "description": "折叠记录里给出的文件路径"},
            },
            "required": ["path"],
        },
        fn=read_stashed, risk="low", scope="public",
    ))

    reg.register(ToolSpec(
        name="get_policy",
        description=DESC_GET_POLICY,
        parameters={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "enum": ["warranty", "coupon", "repair"],
                    "description": "政策主题",
                },
            },
            "required": ["topic"],
        },
        fn=get_policy, risk="low", scope="public",
    ))

    reg.register(ToolSpec(
        name="create_repair_ticket",
        description=DESC_CREATE_TICKET,
        parameters={
            "type": "object",
            "properties": {
                "order": {"type": "string",
                          "description": "订单编号，形如 order-0803"},
                "symptom": {"type": "string", "description": "客户描述的故障现象"},
                "warranty_claim": {"type": "boolean",
                                   "description": "是否按保修免费处理"},
            },
            "required": ["order", "symptom", "warranty_claim"],
        },
        fn=create_repair_ticket,
        risk="high",
        idempotent=False,          # 调两次会开两张工单
        reversible=False,
        requires_confirmation=True,
        scope="customer",
    ))

    return reg, ids
