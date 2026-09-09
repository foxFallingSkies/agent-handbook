"""记忆层的测试。手册第 7 章。

重点在 §7.9：记忆是**持久化的**攻击面，一次成功的注入影响此后每一次运行，
而排查时在当次输入里看不到污染源。所以路径校验的测试写在最前面，
而且是对着六个命令的每一个路径参数逐个试的。
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from handbook.memory import (
    PREFIX,
    MemoryError,
    MemoryStore,
    api_tool,
    dispatch,
)


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(root=tmp_path / "mem")
    s.create(f"{PREFIX}/ok.txt", "hello\nworld\n")
    return s


# ------------------------------------------------------------------ 安全


def test_path_traversal_is_rejected_by_every_command(store: MemoryStore):
    """§7.9：六个命令都要挡，漏一个前面五个就白做了。

    ⚠️ 最后两行是重点：rename 有**两个**路径参数。只校验 old_path 是一个
    很容易犯的错，而它足够把任意文件搬到记忆目录外面去。
    """
    evil = f"{PREFIX}/../../etc/passwd"
    calls = [
        ("view", lambda: store.view(evil)),
        ("create", lambda: store.create(evil, "x")),
        ("str_replace", lambda: store.str_replace(evil, "a", "b")),
        ("insert", lambda: store.insert(evil, 0, "x")),
        ("delete", lambda: store.delete(evil)),
        ("rename/old", lambda: store.rename(evil, f"{PREFIX}/x.txt")),
        ("rename/new", lambda: store.rename(f"{PREFIX}/ok.txt", evil)),
    ]
    for name, call in calls:
        with pytest.raises(MemoryError, match="越出|必须以"):
            call()
        assert not Path("/etc/passwd_moved").exists(), f"{name} 放行了越权路径"


def test_absolute_and_relative_paths_outside_prefix_are_rejected(store: MemoryStore):
    """不以 /memories 开头的一律拒绝——包括看起来无害的相对路径。"""
    for bad in ["/etc/passwd", "notes.md", "", "/memoriesX/a.txt", None]:
        with pytest.raises(MemoryError):
            store.view(bad)                        # type: ignore[arg-type]


def test_cannot_delete_or_rename_the_memory_root(store: MemoryStore):
    """把「告诉它别做」变成「让它做不到」——第 4 章的防呆。"""
    with pytest.raises(MemoryError):
        store.delete(PREFIX)
    with pytest.raises(MemoryError):
        store.rename(PREFIX, f"{PREFIX}/moved")


# ------------------------------------------------------------------ 命令语义


def test_view_directory_matches_the_documented_format(store: MemoryStore):
    """§7.11：返回值格式照抄文档，因为模型是照着它训练的。"""
    out = store.view(PREFIX)
    assert out.startswith("Here're the files and directories up to 2 levels deep in")
    assert "\t" in out                             # size 和 path 之间是制表符
    assert f"{PREFIX}/ok.txt" in out


def test_view_file_has_six_wide_right_aligned_line_numbers(store: MemoryStore):
    out = store.view(f"{PREFIX}/ok.txt")
    assert out.startswith(f"Here's the content of {PREFIX}/ok.txt with line numbers:")
    assert "     1\thello" in out                  # 6 宽右对齐 + 制表符
    assert "     2\tworld" in out


def test_view_range_supports_open_ended(store: MemoryStore):
    store.create(f"{PREFIX}/n.txt", "\n".join(str(i) for i in range(1, 11)))
    assert "     3\t3" in store.view(f"{PREFIX}/n.txt", [3, -1])
    assert "     1\t1" not in store.view(f"{PREFIX}/n.txt", [3, -1])


def test_create_overwrites_because_the_interface_says_so(store: MemoryStore):
    """§7.4：接口宁可整个重写，也不想让记忆库靠追加长大。"""
    store.create(f"{PREFIX}/ok.txt", "replaced\n")
    assert "replaced" in store.view(f"{PREFIX}/ok.txt")
    assert "hello" not in store.view(f"{PREFIX}/ok.txt")


def test_str_replace_refuses_ambiguous_matches(store: MemoryStore):
    """多处匹配必须拒绝，不能默默改第一处。

    改第一处等于让模型以为它改的是它想改的那处——一个静默的错，
    而记忆层的静默错会一直活下去（§7.4 难题四）。
    """
    store.create(f"{PREFIX}/dup.txt", "color: blue\nbg color: blue\n")
    with pytest.raises(MemoryError, match="Multiple occurrences"):
        store.str_replace(f"{PREFIX}/dup.txt", "color: blue", "color: green")
    assert store.view(f"{PREFIX}/dup.txt").count("blue") == 2   # 一个字没改


def test_str_replace_with_empty_new_str_deletes(store: MemoryStore):
    store.str_replace(f"{PREFIX}/ok.txt", "hello")
    assert "hello" not in store.view(f"{PREFIX}/ok.txt")


def test_insert_line_zero_goes_to_the_top(store: MemoryStore):
    store.insert(f"{PREFIX}/ok.txt", 0, "first")
    assert "     1\tfirst" in store.view(f"{PREFIX}/ok.txt")


def test_insert_out_of_range_says_the_valid_range(store: MemoryStore):
    """错误消息是 prompt：要给出合法区间，模型才能自己改（第 4 章 §4.5）。"""
    with pytest.raises(MemoryError, match=r"\[0, \d+\]"):
        store.insert(f"{PREFIX}/ok.txt", 999, "x")


def test_rename_refuses_to_overwrite(store: MemoryStore):
    store.create(f"{PREFIX}/b.txt", "b")
    with pytest.raises(MemoryError, match="already exists"):
        store.rename(f"{PREFIX}/ok.txt", f"{PREFIX}/b.txt")


# ------------------------------------------------------------------ 容量与淘汰


def test_file_size_cap_is_enforced_with_an_actionable_hint(tmp_path: Path):
    """§7.9 安全建议二：给大小设上限。"""
    s = MemoryStore(root=tmp_path / "m", max_file_bytes=100)
    with pytest.raises(MemoryError) as e:
        s.create(f"{PREFIX}/big.txt", "x" * 200)
    # ⚠️ 断言 to_model() 而不是 str(e)：hint 只在前者里。
    # 模型看到的是 to_model()，所以「错误消息是不是可行动的」这个问题
    # 只能对着 to_model() 问——对着 str() 问会漏掉整个 hint。
    assert "拆成" in e.value.to_model()


def test_eviction_is_by_access_time_not_write_time(tmp_path: Path):
    """§7.4：按访问时间淘汰。

    ⚠️ 这条断言是这个模块最容易写反的地方。按写入时间淘汰会删掉
    「写了一次、之后每次都用」的记忆——那恰好是最有价值的一类。
    """
    s = MemoryStore(root=tmp_path / "m")
    s.create(f"{PREFIX}/old_but_used.txt", "still needed")
    s.create(f"{PREFIX}/never_read.txt", "written once, never used")
    time.sleep(0.05)

    s.view(f"{PREFIX}/old_but_used.txt")           # 只读这一个

    gone = s.evict_unused(older_than_seconds=0.02)
    assert f"{PREFIX}/never_read.txt" in gone
    assert f"{PREFIX}/old_but_used.txt" not in gone
    # 两个文件是同时写的，所以只有「按访问时间」才能得到这个结果
    assert "still needed" in s.view(f"{PREFIX}/old_but_used.txt")


# ------------------------------------------------------------------ 派发层


def test_dispatch_routes_all_six_commands(store: MemoryStore):
    assert api_tool() == {"type": "memory_20250818", "name": "memory"}
    assert "ok.txt" in dispatch(store, {"command": "view", "path": PREFIX})
    dispatch(store, {"command": "create", "path": f"{PREFIX}/d.txt",
                     "file_text": "a\nb\n"})
    dispatch(store, {"command": "str_replace", "path": f"{PREFIX}/d.txt",
                     "old_str": "a", "new_str": "A"})
    dispatch(store, {"command": "insert", "path": f"{PREFIX}/d.txt",
                     "insert_line": 0, "insert_text": "top"})
    dispatch(store, {"command": "rename", "old_path": f"{PREFIX}/d.txt",
                     "new_path": f"{PREFIX}/e.txt"})
    assert "Successfully deleted" in dispatch(
        store, {"command": "delete", "path": f"{PREFIX}/e.txt"})


def test_missing_parameter_becomes_a_prompt_not_a_keyerror(store: MemoryStore):
    """缺参数是工具层的失败，必须变成模型能读的文本（第 4 章）。"""
    with pytest.raises(MemoryError) as e:
        dispatch(store, {"command": "create", "path": f"{PREFIX}/x.txt"})
    assert "缺少参数" in e.value.to_model()


def test_unknown_command_lists_the_valid_ones(store: MemoryStore):
    with pytest.raises(MemoryError) as e:
        dispatch(store, {"command": "append", "path": PREFIX})
    assert "view" in e.value.to_model()            # 把可用命令列出来
