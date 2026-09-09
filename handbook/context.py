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
    consumed: bool = False         # 是否已被后续步骤消费
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
        fold_after_steps: int = 2,
        emergency_at: float = 0.85,
        summarize: Callable[[list[Entry]], str] = default_summarizer,
        workspace: str | Path = "workspace",
    ):
        self.window = window
        self.fold_after_steps = fold_after_steps
        self.emergency_at = emergency_at
        self.summarize = summarize

        self.entries: list[Entry] = []
        self.step = 0
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

        # 埋点：这些数据是验证本章策略是否有效的唯一依据（第 10 章会消费它）
        self.events: list[dict] = []

    # ------------------------------------------------------------------

    def append(self, e: Entry) -> None:
        self.entries.append(e)
        self._rolling_fold()                 # 主要机制：每一步都做
        if self.usage_ratio() >= self.emergency_at:
            self._emergency_compact()        # 兜底机制

    def used_tokens(self) -> int:
        return sum(e.tokens for e in self.entries)

    def usage_ratio(self) -> float:
        return self.used_tokens() / self.window

    # ------------------------------------------------------------------
    # 主要机制：滚动折叠
    # ------------------------------------------------------------------

    def _rolling_fold(self) -> None:
        """把够老、且能安全折叠的观察结果折成一行。

        判据（第 3 章）：**这段内容还会不会影响后续决策？**
        - 失败记录  → 永远保留。它把一片动作空间标成了"此路不通"。
        - 带钥匙的  → 可折叠，因为随时能取回，折叠是无损的。
        - 已消费的  → 可折叠，它的结论已经体现在后续动作里。
        - 其余      → 保留。
        """
        folded_tokens = 0
        for e in self.entries:
            if e.folded or e.kind != "observation":
                continue
            if self.step - e.step < self.fold_after_steps:
                continue
            if e.kind == "error":
                continue
            before = e.tokens
            if e.retrieval_key:
                e.fold(f"[已折叠，可用 {e.retrieval_key} 取回原文]")
            elif e.consumed:
                e.fold(f"[已折叠：第 {e.step} 步的工具结果，已被后续步骤使用]")
            else:
                continue
            folded_tokens += before - e.tokens

        if folded_tokens:
            self.events.append(
                {"type": "rolling_fold", "step": self.step, "saved_tokens": folded_tokens}
            )

    # ------------------------------------------------------------------
    # 兜底机制：阈值压缩
    # ------------------------------------------------------------------

    def _emergency_compact(self) -> None:
        before = self.used_tokens()

        keep: list[Entry] = []
        fold: list[Entry] = []
        for e in self.entries:
            (keep if self._must_keep(e) else fold).append(e)

        if not fold:
            # 压不动了：全部都是必须保留的。这是一个需要被看见的信号，
            # 而不是静默地继续跑到溢出。
            self.events.append(
                {"type": "compaction_stalled", "step": self.step, "tokens": before}
            )
            return

        summary = Entry(
            role="user", content=self.summarize(fold), kind="summary", step=self.step
        )
        self.entries = [summary] + keep
        after = self.used_tokens()
        self.events.append(
            {
                "type": "compaction",
                "step": self.step,
                "before": before,
                "after": after,
                "folded_entries": len(fold),
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
        """把内容写到上下文之外，上下文里只留一行指针。"""
        path = self.workspace / filename
        with path.open("a", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n")
        return Entry(
            role="user",
            content=f"[已记录到 {path}]",
            kind="note",
            step=self.step,
            retrieval_key=str(path),
        )

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
        return Entry(
            role="user", content=f"当前进度：\n{body}", kind="note", step=self.step
        )

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
        return out

    def stats(self) -> dict:
        return {
            "entries": len(self.entries),
            "folded": sum(1 for e in self.entries if e.folded),
            "used_tokens": self.used_tokens(),
            "usage_ratio": round(self.usage_ratio(), 4),
            "events": self.events,
        }
