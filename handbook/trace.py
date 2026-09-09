"""最小可用的 trace。手册第 10 章会把它接到 OpenTelemetry 上，
但在那之前，前面几章就已经需要"看得见发生了什么"了。

这是评审给的一条意见：一本书如果反复说"这事只能靠测/靠看数据"，
就不能把观测能力推迟到第 10 章。

数据模型是一棵 span 树：

    run
    ├── step 1
    │   ├── llm_call
    │   └── tool_call
    ├── step 2
    │   └── ...
    ├── fold / compaction      ← 上下文事件也是 span
    └── ...
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Span:
    name: str
    kind: str                     # run / step / llm_call / tool_call / context
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_id: str | None = None
    start: float = field(default_factory=time.perf_counter)
    end: float | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def duration_ms(self) -> float:
        return ((self.end or time.perf_counter()) - self.start) * 1000

    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind,
            "duration_ms": round(self.duration_ms, 1),
            "attrs": self.attrs,
            "error": self.error,
        }


class Trace:
    """一次 agent 运行的完整记录。"""

    def __init__(self, name: str):
        self.spans: list[Span] = []
        self.root = self._open(name, "run", None)

    # ------------------------------------------------------------------

    def _open(self, name: str, kind: str, parent: Span | None) -> Span:
        s = Span(name=name, kind=kind, parent_id=parent.span_id if parent else None)
        self.spans.append(s)
        return s

    def span(self, name: str, kind: str, parent: Span | None = None) -> "_SpanCtx":
        return _SpanCtx(self, name, kind, parent or self.root)

    def close(self) -> None:
        for s in self.spans:
            if s.end is None:
                s.end = time.perf_counter()

    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """把一次运行压成一屏能看完的几个数。"""
        self.close()
        llm = [s for s in self.spans if s.kind == "llm_call"]
        tools = [s for s in self.spans if s.kind == "tool_call"]
        ctx = [s for s in self.spans if s.kind == "context"]

        tool_counts: dict[str, int] = {}
        tool_errors = 0
        for s in tools:
            tool_counts[s.name] = tool_counts.get(s.name, 0) + 1
            if s.error:
                tool_errors += 1

        return {
            "steps": len([s for s in self.spans if s.kind == "step"]),
            "llm_calls": len(llm),
            "tool_calls": len(tools),
            "tool_breakdown": tool_counts,
            "tool_errors": tool_errors,
            "context_events": [s.attrs | {"event": s.name} for s in ctx],
            "wall_ms": round(self.root.duration_ms, 1),
            "model_ms": round(sum(s.duration_ms for s in llm), 1),
            "tool_ms": round(sum(s.duration_ms for s in tools), 1),
        }

    def save(self, path: str | Path) -> Path:
        self.close()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {"summary": self.summary(), "spans": [s.to_dict() for s in self.spans]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return p

    def render_tree(self) -> str:
        """把 span 树画成文本。这是读者第一次"看见 agent 做了什么"的地方。"""
        by_parent: dict[str | None, list[Span]] = {}
        for s in self.spans:
            by_parent.setdefault(s.parent_id, []).append(s)

        lines: list[str] = []

        def walk(span: Span, depth: int) -> None:
            pad = "  " * depth
            mark = "✗" if span.error else " "
            extra = ""
            if span.kind == "llm_call":
                a = span.attrs
                extra = (
                    f"  [in={a.get('input_tokens', 0)} "
                    f"cached={a.get('cache_read', 0)} "
                    f"out={a.get('output_tokens', 0)}]"
                )
            elif span.kind == "tool_call":
                extra = f"  [{a_short(span.attrs.get('args', {}))}]"
            elif span.kind == "context":
                extra = f"  [{span.attrs}]"
            lines.append(
                f"{pad}{mark} {span.name} ({span.duration_ms:.0f}ms){extra}"
            )
            for child in by_parent.get(span.span_id, []):
                walk(child, depth + 1)

        walk(self.root, 0)
        return "\n".join(lines)


def a_short(args: dict, limit: int = 60) -> str:
    s = json.dumps(args, ensure_ascii=False, sort_keys=True)
    return s if len(s) <= limit else s[: limit - 1] + "…"


class _SpanCtx:
    def __init__(self, trace: Trace, name: str, kind: str, parent: Span):
        self.trace, self.name, self.kind, self.parent = trace, name, kind, parent
        self.span: Span | None = None

    def __enter__(self) -> Span:
        self.span = self.trace._open(self.name, self.kind, self.parent)
        return self.span

    def __exit__(self, exc_type, exc, tb) -> bool:
        assert self.span is not None
        self.span.end = time.perf_counter()
        if exc is not None:
            self.span.error = f"{exc_type.__name__}: {exc}"
        return False
