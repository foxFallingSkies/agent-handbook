"""上下文管理。手册第 3 章。

第 3 章说 context rot 是"性能梯度而不是硬悬崖"，因此**压缩是持续的纪律，
不是快满了才做的应急措施**。

早期版本的实现和这句话是矛盾的：它只有一个 `compact_at=0.90` 的全局阈值，
那正是"快满了才做"。这一版把两者分开：

- **滚动折叠（rolling fold）** —— 主要机制，每一步都在做。
  一条观察结果只要够老、且已被消费或带着"取回的钥匙"，就立刻折叠。
- **阈值压缩（compaction）** —— 兜底机制，明确标为应急。
  滚动折叠压不住时才触发（例如一次工具返回了三万 token）。

第二个机制存在的理由是诚实的：**滚动折叠不能保证一定压得住**，
因为单条内容的大小没有上限。兜底不是理念妥协，是工程必需。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from . import tokens as tk

Kind = Literal["task", "action", "observation", "error", "note", "summary"]


@dataclass
class Entry:
    """上下文里的一条记录。"""

    role: str                      # user / assistant
    content: str
    kind: Kind
    step: int
    tool_use_id: str | None = None
    retrieval_key: str | None = None   # 取回的钥匙：URL / 文件路径 / 查询语句
    folded: bool = False
    # 结构化内容块。assistant 轮里的 tool_use 必须原样回传给 API，
    # 不能压成字符串——否则模型无法把 tool_result 和它的调用对上。
    blocks: list[dict] | None = None
    _tokens: int | None = None

    @property
    def tokens(self) -> int:
        if self._tokens is None:
            self._tokens = tk.estimate(self.content)
        return self._tokens

    def fold(self, note: str) -> None:
        """折叠成一行摘要。因为有钥匙，这是**可恢复的**，所以无损。"""
        self.content = note
        self.folded = True
        self._tokens = None


class MessageInvariantError(Exception):
    """构造出的 messages 违反了 API 的硬约束。

    这个异常存在的意义是**在本地失败，而不是让 API 去发现**。
    真实 API 对这两条会直接返回 400，而离线的脚本化响应不会校验——
    也就是说，不主动检查的话，这类 bug 只在第一次真实调用时才炸。
    """


def _enforce_invariants(msgs: list[dict]) -> list[dict]:
    """把 API 的两条硬约束变成本地的断言。

    ① 每个 tool_result 必须能对应到前面某个 tool_use。
    ② 角色必须交替（连续同角色要合并）。
    """
    # ② 合并连续同角色
    merged: list[dict] = []
    for m in msgs:
        if merged and merged[-1]["role"] == m["role"]:
            prev, cur = merged[-1]["content"], m["content"]
            prev = prev if isinstance(prev, list) else [{"type": "text", "text": prev}]
            cur = cur if isinstance(cur, list) else [{"type": "text", "text": cur}]
            merged[-1] = {"role": m["role"], "content": prev + cur}
        else:
            merged.append(dict(m))

    # ① 配对检查
    seen_use: set[str] = set()
    for m in merged:
        content = m["content"]
        if not isinstance(content, list):
            continue
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_use":
                seen_use.add(b["id"])
            elif b.get("type") == "tool_result":
                tid = b.get("tool_use_id")
                if tid not in seen_use:
                    raise MessageInvariantError(
                        f"孤儿 tool_result：{tid} 没有对应的 tool_use。\n"
                        f"最常见的成因是压缩把承载 tool_use 的 assistant 轮丢掉了，"
                        f"却留下了它的结果——压缩的单位必须是「整轮」而不是单条记录。"
                    )
    return merged


def default_summarizer(entries: list[Entry]) -> str:
    """兜底压缩用的确定性摘要器。

    刻意不调用模型：兜底路径本身可能是在成本或延迟已经出问题时触发的，
    此时再发一次请求会让情况更糟。需要更好的摘要时，
    传一个自己的 summarizer 进来（通常是一次 FAST 档的模型调用）。
    """
    by_kind: dict[str, int] = {}
    keys: list[str] = []
    for e in entries:
        by_kind[e.kind] = by_kind.get(e.kind, 0) + 1
        if e.retrieval_key:
            keys.append(e.retrieval_key)

    parts = [f"[已压缩 {len(entries)} 条历史：" +
             "，".join(f"{k}×{v}" for k, v in sorted(by_kind.items())) + "]"]
    if keys:
        parts.append("可取回的来源：" + "；".join(keys[:20]))
    return "\n".join(parts)


class ContextManager:
    """管理一次 agent 运行的上下文。

    它不持有 system 和 tools —— 那两者是稳定前缀，属于 llm.Request，
    放在这里会诱使你去修改它们，从而破坏 KV-cache（第 2 章 §2.5）。
    """

    def __init__(
        self,
        *,
        window: int = 200_000,
        fold: bool = True,
        fold_after_steps: int = 2,
        emergency_at: float = 0.85,
        summarize: Callable[[list[Entry]], str] = default_summarizer,
        workspace: str | Path = "workspace",
    ):
        self.window = window
        # ⚠️ 折叠可关，是因为第 3 章 §3.1 那条判断需要读者自己跑一遍才能信：
        # 折叠省 token，但它改写历史中段，会把 KV-cache 的前缀从那一点作废。
        # 两个开关都在，那张对比表才是可复现的，而不是「作者说的」。
        self.fold_enabled = fold
        self.fold_after_steps = fold_after_steps
        self.emergency_at = emergency_at
        self.summarize = summarize

        self.entries: list[Entry] = []
        self.step = 0
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

        # 埋点：这些数据是验证本章策略是否有效的唯一依据（第 9 章会消费它）
        self.events: list[dict] = []

    # ------------------------------------------------------------------

    def append(self, e: Entry) -> Entry:
        self.entries.append(e)
        self._rolling_fold()                 # 主要机制：每一步都做
        if self.usage_ratio() >= self.emergency_at:
            self._emergency_compact()        # 兜底机制
        return e

    def used_tokens(self) -> int:
        return sum(e.tokens for e in self.entries)

    def usage_ratio(self) -> float:
        return self.used_tokens() / self.window

    # ------------------------------------------------------------------
    # 主要机制：滚动折叠
    # ------------------------------------------------------------------

    def _rolling_fold(self) -> None:
        """把够老、且能**无损**折叠的观察结果折成一行。

        判据（第 3 章）：**这段内容还会不会影响后续决策？**

        ⚠️ 这一版比初版严格得多，原因是一次真实的审查：
        初版有两条折叠分支——「有钥匙」和「已消费」。而 loop 从不写 retrieval_key，
        却无条件把上一步的 observation 标成 consumed，于是**永远走第二条分支**，
        把原文替换成一行摘要且无处可取回。书里写着「因为有钥匙，所以无损」，
        而在唯一真正运行的路径上，那句话是假的。

        现在只剩一条分支：**没有钥匙就不折叠**。
        「无损」不再是一个承诺，而是一个由数据结构保证的事实。

        顺带删掉了 `consumed` 字段。既然折叠必须有钥匙，「后续步骤读过了没有」
        就不再影响任何判断——而它恰恰是那个只有 loop 会写、ContextManager
        却拿它做决策的字段。这类跨层的隐式约定正是上面那个 bug 的成因。
        """
        if not self.fold_enabled:
            return
        folded_tokens = 0
        for e in self.entries:
            if e.folded or e.kind != "observation":
                continue
            if self.step - e.step < self.fold_after_steps:
                continue
            # 唯一的折叠条件：有取回的钥匙
            if not e.retrieval_key:
                continue
            before = e.tokens
            e.fold(
                f"[第 {e.step} 步的工具结果已折叠。"
                f"原文在 {e.retrieval_key}，需要时用 read_stashed 取回]"
            )
            folded_tokens += before - e.tokens

        if folded_tokens:
            self.events.append(
                {"type": "rolling_fold", "step": self.step, "saved_tokens": folded_tokens}
            )

    # ------------------------------------------------------------------
    # 兜底机制：阈值压缩
    # ------------------------------------------------------------------

    def _turns(self) -> list[list[Entry]]:
        """把 entries 切成**原子轮次**：一条 assistant 动作 + 它全部的 tool_result。

        ⚠️ 压缩的单位必须是轮次，不能是单条 Entry。
        初版按 Entry 压，而 `_must_keep` 让 error 永远保留、action 可以被丢——
        两者不同步，于是产生了**没有对应 tool_use 的孤儿 tool_result**，
        真实 API 会直接以 400 拒绝。这个 bug 在离线脚本下测不出来。
        """
        turns: list[list[Entry]] = []
        cur: list[Entry] = []
        for e in self.entries:
            if e.kind in ("action", "task", "summary", "note"):
                if cur:
                    turns.append(cur)
                cur = [e]
            else:                       # observation / error 附着在上一条 action 上
                if not cur:
                    cur = []
                cur.append(e)
        if cur:
            turns.append(cur)
        return turns

    def force_compact(self) -> None:
        """无条件压一次，不看占用率。

        给一种情况用：**服务端说超窗了，而本地估算说没有**。
        本地 token 估算和真实分词器永远有出入，所以「按比例触发」这条路
        一定会有漏网的时候。这时唯一还能做的就是不问比例、直接压。
        """
        self._emergency_compact()

    def _emergency_compact(self) -> None:
        before = self.used_tokens()
        turns = self._turns()

        keep: list[list[Entry]] = []
        fold: list[Entry] = []
        for turn in turns:
            # 整轮一起判：只要这一轮里有任何一条必须保留，整轮都留下
            if any(self._must_keep(e) for e in turn):
                keep.append(turn)
            else:
                fold.extend(turn)

        if not fold:
            self.events.append(
                {"type": "compaction_stalled", "step": self.step, "tokens": before}
            )
            return

        new_summary = self.summarize(fold)
        flat = [e for turn in keep for e in turn]

        # 摘要要**合并**进已有的那条，不能每次 prepend 一条新的——
        # summary 是 must_keep，堆叠几轮之后会导致所有条目都不可折叠（compaction_stalled）
        existing = next((e for e in flat if e.kind == "summary"), None)
        if existing is not None:
            existing.content = existing.content + "\n" + new_summary
            existing._tokens = None
            self.entries = flat
        else:
            self.entries = [
                Entry(role="user", content=new_summary, kind="summary", step=self.step)
            ] + flat

        after = self.used_tokens()
        self.events.append(
            {
                "type": "compaction",
                "step": self.step,
                "before": before,
                "after": after,
                "folded_entries": len(fold),
                "folded_turns": len(turns) - len(keep),
                "ratio": round(after / max(before, 1), 3),
            }
        )

    def _must_keep(self, e: Entry) -> bool:
        """兜底压缩时哪些必须留下。

        注意 action 是**可以**被折叠的——它是 N²/2·a 里增长最快的部分，
        如果永远保留，这一层就没有解决它声称要解决的问题。
        保留的是最近几步的动作，以及所有失败。
        """
        if e.kind in ("task", "summary", "error"):
            return True
        if e.kind == "note":
            return True
        # 最近 fold_after_steps 步内的东西一律保留：模型正在用它们
        return self.step - e.step < self.fold_after_steps

    # ------------------------------------------------------------------
    # Write：外化
    # ------------------------------------------------------------------

    def note(self, text: str, filename: str = "notes.md") -> Entry:
        """把内容写到上下文之外，上下文里只留一行指针。

        ⚠️ 这个方法**自己**把条目挂进上下文，而不是返回一个让调用方挂的
        Entry。原来是后者，而全仓库没有一个调用点——也就是说
        「外化」这条第 3 章反复讲的技术，在参考实现里其实没有接线。
        写完不挂进去，等于写了个日志文件。
        """
        path = self.workspace / filename
        with path.open("a", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n")
        return self.append(Entry(
            role="user",
            content=f"[已记录到 {path}]",
            kind="note",
            step=self.step,
            retrieval_key=str(path),
        ))

    def stash(self, content: str, name: str) -> str:
        """把一段过大的内容落盘，返回取回的钥匙。

        工具返回三万 token 时走这条路：上下文里只留路径和一行摘要，
        需要时再读回来。第 3 章 §3.6。
        """
        path = self.workspace / name
        path.write_text(content, encoding="utf-8")
        return str(path)

    # ------------------------------------------------------------------
    # 复诵
    # ------------------------------------------------------------------

    def recite(self, todo: list[tuple[str, bool]]) -> Entry:
        """把全局计划重写到上下文末端——注意力最集中的位置。

        第 3 章 §3.9。这不只是进度记录，是注意力控制手段。
        """
        body = "\n".join(f"- [{'x' if done else ' '}] {item}" for item, done in todo)
        (self.workspace / "todo.md").write_text(body, encoding="utf-8")
        # 旧的复诵条目要移除，否则会累积多份计划互相打架
        self.entries = [e for e in self.entries if e.kind != "note"
                        or not e.content.startswith("当前进度")]
        # ⚠️ 必须 append。原来这里是 return 一个没挂进去的 Entry——
        # 于是 recite() 只做了上面那半句「删掉旧的进度」，
        # 调用它比不调用更糟：计划被删了，新的一份从没进过上下文。
        # 复述的全部意义就是**把计划重新放到上下文末尾**，
        # 少了这一步，剩下的都是无用功。
        return self.append(Entry(
            role="user", content=f"当前进度：\n{body}", kind="note", step=self.step
        ))

    # ------------------------------------------------------------------

    def to_messages(self) -> list[dict]:
        """渲染成 API 要的 messages。

        ⚠️ 只追加、不修改历史顺序——第 2 章 §2.5.3 规则二。
        """
        out: list[dict] = []
        pending_results: list[dict] = []

        def flush() -> None:
            if pending_results:
                out.append({"role": "user", "content": list(pending_results)})
                pending_results.clear()

        for e in self.entries:
            # 工具失败也必须以 tool_result 的形式回去——每个 tool_use 都要配对，
            # 否则 API 会拒绝这次请求。失败只是 is_error=True 的那种结果。
            if e.tool_use_id and e.kind in ("observation", "error"):
                # 同一轮里的多个 tool_result 必须合并进一条 user 消息
                pending_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": e.tool_use_id,
                        "content": e.content,
                        **({"is_error": True} if e.kind == "error" else {}),
                    }
                )
                continue
            flush()
            if e.blocks is not None:
                out.append({"role": e.role, "content": e.blocks})
            else:
                out.append({"role": e.role, "content": e.content})
        flush()

        # API 要求首条消息是 user
        while out and out[0]["role"] != "user":
            out.pop(0)

        return _enforce_invariants(out)

    def stats(self) -> dict:
        return {
            "entries": len(self.entries),
            "folded": sum(1 for e in self.entries if e.folded),
            "used_tokens": self.used_tokens(),
            "usage_ratio": round(self.usage_ratio(), 4),
            "events": self.events,
        }
