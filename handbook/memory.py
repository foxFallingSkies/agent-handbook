"""跨会话记忆。手册第 7 章。

这个模块实现的是 Anthropic `memory_20250818` 工具的**服务端**——
也就是那个"客户端执行"里的客户端。模型只发出请求，真正动文件的是这里。

    tools=[{"type": "memory_20250818", "name": "memory"}]

六个命令：view / create / str_replace / insert / delete / rename。
返回值格式照抄官方文档，理由见 §7.11：**返回值本身就是给模型的 prompt**，
它是照着这个格式训练的，自己发明一个更整洁的格式只会让解析变差。

📎 https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool

⚠️ 没有实现的：§7.6 的三因子取回（需要 embedding，检索层还没进仓库）、
自动冲突检测。不写没跑过的代码。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ToolError

# 记忆根目录在模型眼里的名字。它是一个**前缀**，不是真实路径——
# 真实位置由 MemoryStore(root=...) 决定，可以是每用户一个目录、
# 也可以是别的存储后端。
PREFIX = "/memories"

# 单文件上限。官方安全建议里的第二条：给大小设上限。
# 没有上限的记忆文件迟早会长到一次 view 就吃掉半个上下文窗口。
MAX_FILE_BYTES = 64 * 1024
MAX_TOTAL_BYTES = 4 * 1024 * 1024

# 超过这个长度的文本视图会被截断。这个数字来自官方文档对 view 的描述。
VIEW_TRUNCATE_CHARS = 16_000


class MemoryError(ToolError):
    """记忆操作失败。

    继承 ToolError 而不是自成一系，是为了让它走第 4 章那条既定路径：
    **工具层的失败变成给模型看的文本，而不是抛给调用栈**。
    """


# ---------------------------------------------------------------------------


@dataclass
class MemoryStore:
    """文件系统后端的记忆存储。

    换后端（数据库、对象存储、加密文件）只需要替换 _read/_write/_list，
    命令层和校验层都不用动——这也是官方把 /memories 定义成前缀而不是
    真实路径的原因。
    """

    root: Path
    max_file_bytes: int = MAX_FILE_BYTES
    max_total_bytes: int = MAX_TOTAL_BYTES

    # 访问时间。§7.4「按访问时间淘汰」用，不是按写入时间——
    # 按写入时间会淘汰掉「写了一次、之后每次都用」的记忆，
    # 那恰好是最有价值的一类。
    _atime: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 唯一的安全边界
    # ------------------------------------------------------------------

    def _resolve(self, path: str) -> Path:
        """把模型给的路径解析成真实路径，并确认它没跑出记忆根目录。

        ⚠️ 这个函数是整个模块唯一的安全边界。它必须在**每一条命令**的
        开头被调用，没有例外——六个命令里漏掉任何一个，前面五个的校验
        就都白做了。`rename` 有两个路径参数，两个都要过。

        用 resolve() + relative_to() 而不是检查字符串里有没有 ".."：
        字符串匹配挡不住符号链接，也挡不住 URL 编码。
        **用字符串匹配做路径安全是一个反复被攻破的模式。**
        """
        # ⚠️ 必须带斜杠比。只判 startswith(PREFIX) 的话，
        # "/memoriesX/a.txt" 会被切成 rel="X/a.txt" 落回根目录内，静默放行。
        ok = isinstance(path, str) and (path == PREFIX or path.startswith(PREFIX + "/"))
        if not ok:
            raise MemoryError(
                f"路径必须以 {PREFIX} 开头，收到的是 {path!r}。",
                hint=f"所有记忆文件都在 {PREFIX} 下，例如 {PREFIX}/notes.md。",
            )
        rel = path[len(PREFIX):].lstrip("/")
        resolved = (self.root / rel).resolve()
        try:
            resolved.relative_to(self.root.resolve())
        except ValueError:
            raise MemoryError(
                f"路径 {path!r} 越出了记忆目录。",
                hint="不要使用 `..`，也不要给绝对路径。",
            ) from None
        return resolved

    def _display(self, p: Path) -> str:
        """真实路径 → 模型看到的路径。反方向的 _resolve。"""
        rel = p.resolve().relative_to(self.root.resolve())
        return PREFIX if str(rel) == "." else f"{PREFIX}/{rel}"

    def _touch(self, p: Path) -> None:
        self._atime[str(p)] = time.time()

    def _total_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())

    # ------------------------------------------------------------------
    # 六个命令
    # ------------------------------------------------------------------

    def view(self, path: str, view_range: list[int] | None = None) -> str:
        p = self._resolve(path)
        if not p.exists():
            raise MemoryError(f"The path {path} does not exist. "
                              f"Please provide a valid path.")

        if p.is_dir():
            lines = [f"{_human_size(p)}\t{self._display(p)}"]
            for f in sorted(p.rglob("*")):
                if f.name.startswith(".") or "node_modules" in f.parts:
                    continue
                if len(f.resolve().relative_to(p.resolve()).parts) > 2:
                    continue                       # 文档说只到 2 层
                lines.append(f"{_human_size(f)}\t{self._display(f)}")
            return (f"Here're the files and directories up to 2 levels deep in "
                    f"{path}, excluding hidden items and node_modules:\n"
                    + "\n".join(lines))

        self._touch(p)
        text = p.read_text(encoding="utf-8")
        rows = text.split("\n")
        if len(rows) > 999_999:
            raise MemoryError(f"File {path} exceeds maximum line limit "
                              f"of 999,999 lines.")

        start, end = 1, len(rows)
        if view_range:
            start = max(1, view_range[0])
            end = len(rows) if view_range[1] == -1 else min(len(rows), view_range[1])

        body = "\n".join(f"{i:>6}\t{rows[i - 1]}" for i in range(start, end + 1))
        if len(body) > VIEW_TRUNCATE_CHARS:
            body = (body[:VIEW_TRUNCATE_CHARS]
                    + f"\n[已截断。用 view_range 分段读取剩余部分。]")
        return f"Here's the content of {path} with line numbers:\n{body}"

    def create(self, path: str, file_text: str) -> str:
        """创建或**覆盖**一个记忆文件。

        📌 官方文档说模型的工具描述里写的是 "creates or overwrites"，
        所以会收到对已存在路径的 create。这不是模型出错——
        接口本来就宁可让它整个重写一个文件，也不想让记忆库靠追加长大（§7.4）。
        """
        p = self._resolve(path)
        self._check_size(path, file_text)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(file_text, encoding="utf-8")
        self._touch(p)
        return f"File created successfully at: {path}"

    def str_replace(self, path: str, old_str: str, new_str: str = "") -> str:
        p = self._resolve(path)
        if not p.is_file():
            raise MemoryError(f"Error: The path {path} does not exist. "
                              f"Please provide a valid path.")
        text = p.read_text(encoding="utf-8")
        # ⚠️ 必须对**整段文本**数，不能按行判断"包含"。
        # 按行判断有两个静默后果：多行 old_str 永远匹配不上（而官方语义是
        # verbatim 子串）；同一行里出现两次会被当成"唯一"，默默改掉第一处——
        # 而这正是下面那段代码声称要防的事。
        n_hits = text.count(old_str)
        hits = [text[:m].count("\n") + 1
                for m in _find_all(text, old_str)] if n_hits else []
        if not hits:
            raise MemoryError(
                f"No replacement was performed, old_str `{old_str}` "
                f"did not appear verbatim in {path}."
            )
        if n_hits > 1:
            # 多处匹配必须拒绝，不能改第一处。
            # 改第一处等于让模型以为它改的是它想改的那处——静默的错。
            raise MemoryError(
                f"No replacement was performed. Multiple occurrences of "
                f"old_str `{old_str}` in lines: {hits}. Please ensure it is unique"
            )
        new = text.replace(old_str, new_str, 1)
        self._check_size(path, new)
        p.write_text(new, encoding="utf-8")
        self._touch(p)
        return "The memory file has been edited."

    def insert(self, path: str, insert_line: int, insert_text: str) -> str:
        p = self._resolve(path)
        if not p.is_file():
            raise MemoryError(f"Error: The path {path} does not exist")
        rows = p.read_text(encoding="utf-8").split("\n")
        if insert_line < 0 or insert_line > len(rows):
            raise MemoryError(
                f"Error: Invalid `insert_line` parameter: {insert_line}. "
                f"It should be within the range of lines of the file: "
                f"[0, {len(rows)}]"
            )
        rows.insert(insert_line, insert_text.rstrip("\n"))
        new = "\n".join(rows)
        self._check_size(path, new)
        p.write_text(new, encoding="utf-8")
        self._touch(p)
        return f"The file {path} has been edited."

    def delete(self, path: str) -> str:
        p = self._resolve(path)
        if p.resolve() == self.root.resolve():
            # 文档说模型的工具描述里就写了它不能删记忆根目录。
            # 但那是"告诉它别做"，这里是"让它做不到"——第 4 章的防呆。
            raise MemoryError(f"不能删除记忆根目录 {PREFIX}。",
                              hint="只能删除其中的文件或子目录。")
        if not p.exists():
            raise MemoryError(f"Error: The path {path} does not exist")
        if p.is_dir():
            import shutil
            shutil.rmtree(p)
        else:
            p.unlink()
        self._atime.pop(str(p), None)
        return f"Successfully deleted {path}"

    def rename(self, old_path: str, new_path: str) -> str:
        """⚠️ 两个路径参数都要过 _resolve。

        只校验 old_path 是一个很容易犯的错，而它足够把任意文件
        搬到记忆目录外面去。测试里对着两个参数各试了一遍。
        """
        src = self._resolve(old_path)
        dst = self._resolve(new_path)
        if src.resolve() == self.root.resolve():
            raise MemoryError(f"不能重命名记忆根目录 {PREFIX}。")
        if not src.exists():
            raise MemoryError(f"Error: The path {old_path} does not exist")
        if dst.exists():
            raise MemoryError(f"Error: The destination {new_path} already exists")
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        self._atime[str(dst)] = self._atime.pop(str(src), time.time())
        return f"Successfully renamed {old_path} to {new_path}"

    # ------------------------------------------------------------------

    def _check_size(self, path: str, text: str) -> None:
        n = len(text.encode("utf-8"))
        if n > self.max_file_bytes:
            raise MemoryError(
                f"记忆文件 {path} 超过单文件上限（{n} > {self.max_file_bytes} 字节）。",
                hint="把它拆成几个按主题分的文件，或者删掉已经不成立的条目。",
            )
        if self._total_bytes() + n > self.max_total_bytes:
            raise MemoryError(
                f"记忆总量将超过上限 {self.max_total_bytes} 字节。",
                hint="先用 view 看一遍 /memories，删掉过期的文件。",
            )

    # ------------------------------------------------------------------
    # 淘汰
    # ------------------------------------------------------------------

    def evict_unused(self, older_than_seconds: float) -> list[str]:
        """删掉太久没被**读**过的记忆文件。§7.4 的第三条对策。

        没有 atime 记录的文件按"从未被读过"处理——它们是最该走的：
        写进去之后一次都没用上，说明当初就不该写。
        """
        cutoff = time.time() - older_than_seconds
        gone = []
        for f in sorted(self.root.rglob("*")):
            if not f.is_file():
                continue
            if self._atime.get(str(f), 0.0) < cutoff:
                gone.append(self._display(f))
                f.unlink()
        return gone


# ---------------------------------------------------------------------------


def dispatch(store: MemoryStore, cmd: dict) -> str:
    """把一个 memory tool_use 的 input 派发到对应命令。

    直接接在 Anthropic 的 `memory_20250818` 工具后面：
    模型发来 {"command": "view", "path": "/memories"}，这里执行并返回文本。
    """
    name = cmd.get("command")
    handlers = {
        "view": lambda: store.view(cmd["path"], cmd.get("view_range")),
        "create": lambda: store.create(cmd["path"], cmd["file_text"]),
        "str_replace": lambda: store.str_replace(
            cmd["path"], cmd["old_str"], cmd.get("new_str", "")),
        "insert": lambda: store.insert(
            cmd["path"], cmd["insert_line"], cmd["insert_text"]),
        "delete": lambda: store.delete(cmd["path"]),
        "rename": lambda: store.rename(cmd["old_path"], cmd["new_path"]),
    }
    if name not in handlers:
        raise MemoryError(
            f"未知的 memory 命令 {name!r}。",
            hint=f"可用命令：{', '.join(sorted(handlers))}。",
        )
    try:
        return handlers[name]()
    except KeyError as e:
        # 缺参数也要变成给模型看的文本，而不是 KeyError（第 4 章）
        raise MemoryError(
            f"命令 {name} 缺少参数 {e.args[0]!r}。",
            hint=f"完整参数见官方文档的 Tool commands 一节。",
        ) from None


def api_tool() -> dict:
    """Anthropic 提供的工具，配置就这一行——不需要你写 schema。"""
    return {"type": "memory_20250818", "name": "memory"}


# ---------------------------------------------------------------------------


def _find_all(text: str, sub: str):
    """返回 sub 在 text 里全部出现位置。给 str_replace 报行号用。"""
    i = text.find(sub)
    while i != -1:
        yield i
        i = text.find(sub, i + 1)


def _human_size(p: Path) -> str:
    n = p.stat().st_size if p.is_file() else sum(
        f.stat().st_size for f in p.rglob("*") if f.is_file()
    )
    for unit in ("", "K", "M", "G"):
        if n < 1024 or unit == "G":
            return f"{n:.1f}{unit}" if unit else f"{n}"
        n /= 1024
    return f"{n:.1f}G"
