"""无人值守的可靠性骨架。手册第 11 章。

这个模块存在的理由，是第 11 章那句话：**无人值守系统的头号敌人不是崩溃，
是静默失败。** 下面五件东西全都在处理「不报错但出事」的那些路径。

    single_instance   同一台机器只跑一个        §11.7
    Lease             跨机器只有一个持有者      §11.7
    Ledger            同一件事只入队一次        §11.8
    DeadLetter        失败之后还能重放          §11.9
    Heartbeat         卡住了要有人知道          §11.10

⚠️ 一条贯穿全模块的设计取向：**在两种错里选可恢复的那种。**
先写台账再投队列 → 最坏是丢一次任务（可以重试补）；
反过来 → 最坏是重复执行（扣两次款补不回来）。
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# §11.7 单例：同一台机器
# ---------------------------------------------------------------------------


@contextmanager
def single_instance(lock_path: str | Path) -> Iterator[None]:
    """同一台机器上只允许一个实例在跑。跑不到锁就抛 AlreadyRunning。

    ⚠️ 用 flock，不要用「写一个 pid 文件再检查」。后者有两个经典问题：
      ① 检查和写入之间存在竞态，两个进程可能都通过检查；
      ② 进程被 kill -9 之后 pid 文件留在原地，下次启动误判成「已在运行」，
         而这个误判**不会报错**——它表现为任务从此再也不跑了。
    flock 由内核维护，进程一死锁自动释放。

    ⚠️ 文件对象必须在整个临界区内保持被引用。
    写成 `fcntl.flock(open(p, "w"), ...)` 的话，open() 的返回值没人引用，
    对象被 GC，锁跟着释放——**而代码看起来完全正常**。
    这里用 with 语句持有它，就是为了让这件事没法写错。
    """
    p = Path(lock_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    f = p.open("w")
    try:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as e:
            raise AlreadyRunning(f"另一个实例正持有 {p}") from e
        f.write(f"{os.getpid()}\n")
        f.flush()
        yield
    finally:
        try:
            fcntl.flock(f, fcntl.LOCK_UN)
        finally:
            f.close()


class AlreadyRunning(RuntimeError):
    """已经有一个实例在跑。这不是故障，是单例机制在起作用。"""


# ---------------------------------------------------------------------------
# §11.7 租约：跨机器
# ---------------------------------------------------------------------------


@dataclass
class Lease:
    """带过期时间的持有权。跨机器时代替 flock。

    ⚠️ ttl 要按「最长一步」定，不是按平均。agent 的单步耗时方差极大——
    一次工具调用可能 200 毫秒，也可能是一个跑了两分钟的检索。
    ttl 比最长一步短，你会在任务跑到一半时被别人抢走。

    ⚠️ 更要紧的一条：**租约保证「同时只有一个持有者」，不保证
    「被抢走的那个会立刻停下」。** 老实例可能正卡在一次网络调用里，
    等它回过神来它已经不是持有者了——但它可能已经写了。
    所以租约必须配合写入侧的幂等（Ledger / 幂等键），不能单独用。

    生产环境里这个 store 应该换成数据库或 Redis 的原子操作；
    这里用文件是为了让这一层的**语义**能被读懂和被测。
    """

    path: Path
    owner: str
    ttl: float = 90.0

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- 读 --

    def _read(self) -> dict | None:
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None            # 写到一半崩了，当作没有租约

    def holder(self, now: float | None = None) -> str | None:
        """当前有效持有者。过期的不算。"""
        rec = self._read()
        if not rec:
            return None
        if (now or time.time()) >= rec.get("expires_at", 0):
            return None
        return rec.get("owner")

    # -- 写 --

    def acquire(self, now: float | None = None) -> bool:
        """尝试拿到租约。已被别人有效持有则返回 False。"""
        t = now or time.time()
        cur = self.holder(t)
        if cur is not None and cur != self.owner:
            return False
        self._write(t)
        return True

    def renew(self, now: float | None = None) -> bool:
        """续期。⚠️ 只有仍然是持有者才能续——已经被抢走就必须停手。

        这个返回值是给调用方用来**主动退出**的：续期失败意味着
        你已经不是持有者了，此刻还在跑的任何写操作都要停。
        """
        t = now or time.time()
        if self.holder(t) != self.owner:
            return False
        self._write(t)
        return True

    def release(self) -> None:
        if self.holder() == self.owner:
            self.path.unlink(missing_ok=True)

    def _write(self, t: float) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            {"owner": self.owner, "expires_at": t + self.ttl},
            ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)      # 原子替换，避免读到写了一半的内容


# ---------------------------------------------------------------------------
# §11.8 只入队一次
# ---------------------------------------------------------------------------


def dedup_key(*parts: Any) -> str:
    """算一个稳定的去重键。

    ⚠️ 用 hashlib，不要用内置 hash()。后者对字符串加了随机盐
    （PYTHONHASHSEED），**跨进程不一致**——而去重键恰好要跨进程比对。
    这个 bug 的表现是：重启之后所有任务都被当成新的，全部重跑一遍。

    ⚠️ parts 里不要放「当前时间戳」或自增序号。判据和幂等键一样：
    **同一件事重来时算出来必须相同。**
    """
    raw = "\x1f".join(
        json.dumps(p, sort_keys=True, ensure_ascii=False)
        if not isinstance(p, str) else p
        for p in parts
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


@dataclass
class Ledger:
    """只追加的去重台账。同一个 key 只放行一次。

    绝大多数队列的语义是 **at-least-once**——Google 的 ADK 文档说得很直白：
    "Tools in an agent are run at least once, and may run more than once
    when resuming."。所以入口处必须有这么一层。
    """

    path: Path
    retain_seconds: float = 7 * 24 * 3600      # 超出这个窗口的重复本来也防不住

    _seen: set[str] = field(default_factory=set, init=False)
    _loaded: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        """启动时把台账读进内存。

        ⚠️ 不要每次入队都全量扫文件。台账长到几十万行时，
        那是一个隐性的性能炸弹——而且它只在跑了很久之后才出现。
        """
        if self._loaded:
            return
        if self.path.exists():
            cutoff = time.time() - self.retain_seconds
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue    # 崩溃时写了半行，跳过
                    if rec.get("at", 0) >= cutoff:
                        self._seen.add(rec["key"])
        self._loaded = True

    def seen(self, key: str) -> bool:
        self._load()
        return key in self._seen

    def record_once(self, key: str, note: str = "") -> bool:
        """登记一次。返回 True 表示这是首次，False 表示之前登记过。

        ⚠️ 调用方的顺序不能反：**先 record_once，再做那件事**。
        反过来的话「做成了但登记失败」会导致下次重做；
        而「登记了但没做成」只会丢一次，可以靠重试补。
        在两种错里选可恢复的那种。
        """
        self._load()
        if key in self._seen:
            return False
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "at": time.time(), "note": note},
                               ensure_ascii=False) + "\n")
            f.flush()
            # ⚠️ fsync 不能省。不 fsync 的话那一行只在页缓存里，
            # 而无人值守系统最常见的死法恰好是整机断电/重启。
            os.fsync(f.fileno())
        self._seen.add(key)
        return True

    def compact(self) -> int:
        """丢掉过期记录，返回丢了多少条。

        🔑 任何「只追加」的可靠性机制，都要在设计的时候就想好它怎么变小。
        否则它会从一个保护机制变成一个故障源。
        """
        if not self.path.exists():
            return 0
        cutoff = time.time() - self.retain_seconds
        kept, dropped = [], 0
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("at", 0) >= cutoff:
                    kept.append(line.rstrip("\n"))
                else:
                    dropped += 1
        tmp = self.path.with_suffix(".compact")
        tmp.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        tmp.replace(self.path)
        self._seen = {json.loads(l)["key"] for l in kept}
        return dropped


# ---------------------------------------------------------------------------
# §11.9 死信
# ---------------------------------------------------------------------------


@dataclass
class DeadLetter:
    """重试耗尽之后的去处。

    ⚠️ 死信队列不是「错误日志的另一个名字」。它的关键性质是：
    **里面的每一条都还能被重新处理。** 所以它存的不是错误消息，
    是足够重放这次任务的全部输入。
    """

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def put(self, task_id: str, reason: str, *,
            payload: dict, trace_id: str = "", done_keys: list[str] | None = None) -> None:
        """把一个失败的任务连同**重放它所需要的一切**存下来。

        done_keys 是已经执行过的副作用动作的幂等键——没有它，
        重放时你不知道「开工单」那一步到底做没做（§11.4）。
        """
        rec = {
            "task_id": task_id,
            "reason": reason,
            "payload": payload,
            "trace_id": trace_id,
            "done_keys": done_keys or [],
            "at": time.time(),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def items(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out


# ---------------------------------------------------------------------------
# §11.10 活性
# ---------------------------------------------------------------------------


@dataclass
class Heartbeat:
    """带进度的心跳。

    ⚠️ 只报时间的心跳没有用。一个卡在网络调用里的进程，心跳线程照样在跳——
    你需要的是**进度推进**的证据，不是**进程存活**的证据。

    所以 beat() 必须带上 step；is_stalled() 判的是「step 有没有变」，
    而不是「有没有心跳」。
    """

    path: Path
    stall_after: float = 300.0     # ⚠️ 按单步的 P99 定，不是按平均

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def beat(self, step: int, note: str = "", now: float | None = None) -> None:
        t = now or time.time()
        prev = self._read()
        # step 没变 → 不刷新 progressed_at，让停滞可见
        progressed = t if (prev is None or prev.get("step") != step) \
            else prev.get("progressed_at", t)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            {"step": step, "note": note, "beat_at": t, "progressed_at": progressed},
            ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def _read(self) -> dict | None:
        if not self.path.exists():
            return None
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def is_stalled(self, now: float | None = None) -> bool:
        """卡住了没有。判据是**进度**多久没动，不是心跳多久没来。"""
        rec = self._read()
        if rec is None:
            return False           # 还没开始，不算卡住
        return (now or time.time()) - rec.get("progressed_at", 0) > self.stall_after

    def status(self) -> dict | None:
        return self._read()
