"""可靠性骨架的测试。手册第 11 章。

⚠️ 这一整个文件的写法遵守第 8 章 §8.8 那条判据：
**把被测的那条规则删掉，测试会变红吗？**

所以每条测试针对的都是一个**具体的退化实现**——
比如「用 hash() 而不是 hashlib」「不判 step 只判时间」「先投队列再写台账」，
而不是泛泛地测「功能能用」。
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import pytest

from handbook.reliability import (
    AlreadyRunning,
    DeadLetter,
    Heartbeat,
    Ledger,
    Lease,
    dedup_key,
    single_instance,
)


# ------------------------------------------------------------------ 单例


def test_second_instance_is_refused(tmp_path: Path):
    lock = tmp_path / "run.lock"
    with single_instance(lock):
        with pytest.raises(AlreadyRunning):
            with single_instance(lock):
                pass


def test_lock_is_released_when_the_block_exits(tmp_path: Path):
    """离开临界区后必须能再拿到。

    ⚠️ 这条防的是「文件对象被 GC 掉锁提前释放」的反面——
    如果实现写成 fcntl.flock(open(p,"w"), ...)，锁会在临界区**内部**
    就被释放，于是这条测试仍然通过，但 test_second_instance_is_refused
    会红。两条要一起看。
    """
    lock = tmp_path / "run.lock"
    with single_instance(lock):
        pass
    with single_instance(lock):
        pass          # 拿不到就会抛


def _child_tries_lock(lock_path: str, q):
    """在真正的另一个进程里试锁——flock 是进程级的，同进程内测不准。"""
    try:
        with single_instance(lock_path):
            q.put("got")
    except AlreadyRunning:
        q.put("refused")


def test_lock_is_cross_process(tmp_path: Path):
    """⚠️ flock 的语义是进程间的，必须真开一个子进程来测。"""
    lock = tmp_path / "x.lock"
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    with single_instance(lock):
        p = ctx.Process(target=_child_tries_lock, args=(str(lock), q))
        p.start()
        p.join(15)
        assert q.get(timeout=5) == "refused"


# ------------------------------------------------------------------ 租约


def test_lease_blocks_a_second_owner_until_it_expires(tmp_path: Path):
    f = tmp_path / "lease.json"
    a = Lease(f, owner="A", ttl=10)
    b = Lease(f, owner="B", ttl=10)
    t0 = 1000.0

    assert a.acquire(now=t0) is True
    assert b.acquire(now=t0 + 1) is False        # A 还持有
    assert b.acquire(now=t0 + 11) is True        # 过期了，B 可以接管


def test_renew_fails_once_the_lease_was_taken_over(tmp_path: Path):
    """⚠️ 这条是租约机制里最要紧的一条。

    续期失败是调用方**主动停手**的信号。如果 renew() 无脑刷新，
    老实例会以为自己还是持有者，继续往下写——而那正是租约要防的事。
    """
    f = tmp_path / "lease.json"
    a = Lease(f, owner="A", ttl=10)
    b = Lease(f, owner="B", ttl=10)
    t0 = 1000.0

    a.acquire(now=t0)
    b.acquire(now=t0 + 11)                       # A 过期，B 接管
    assert a.renew(now=t0 + 12) is False         # A 必须知道自己出局了
    assert b.renew(now=t0 + 12) is True


def test_lease_survives_a_half_written_file(tmp_path: Path):
    """写到一半崩了 → 当作没有租约，而不是抛异常卡死整个系统。"""
    f = tmp_path / "lease.json"
    f.write_text('{"owner": "A", "expi', encoding="utf-8")
    assert Lease(f, owner="B", ttl=10).acquire() is True


# ------------------------------------------------------------------ 去重键


def test_dedup_key_is_stable_across_processes():
    """⚠️ 必须用 hashlib，不能用内置 hash()。

    hash() 对字符串加了随机盐（PYTHONHASHSEED），跨进程不一致——
    表现是重启之后所有任务都被当成新的，全部重跑一遍。
    这里在**另一个解释器进程**里算同一个 key 来钉死它。
    """
    import subprocess
    import sys

    here = dedup_key("task", {"order": "A-1", "n": 3})
    code = (
        "import sys; sys.path.insert(0, %r);"
        "from handbook.reliability import dedup_key;"
        "print(dedup_key('task', {'order': 'A-1', 'n': 3}))"
        % str(Path(__file__).resolve().parents[1])
    )
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == here


def test_dedup_key_ignores_dict_ordering():
    """键顺序变了不该影响去重——和 KV-cache 那条是同一个要求。"""
    assert dedup_key("t", {"a": 1, "b": 2}) == dedup_key("t", {"b": 2, "a": 1})


def test_different_things_get_different_keys():
    assert dedup_key("t", {"a": 1}) != dedup_key("t", {"a": 2})


# ------------------------------------------------------------------ 台账


def test_same_key_is_only_recorded_once(tmp_path: Path):
    led = Ledger(tmp_path / "ledger.jsonl")
    assert led.record_once("k1") is True
    assert led.record_once("k1") is False
    assert led.record_once("k2") is True


def test_ledger_survives_a_restart(tmp_path: Path):
    """重启之后还认得出重复（新对象重新读盘）。"""
    path = tmp_path / "ledger.jsonl"
    assert Ledger(path).record_once("k1") is True
    assert Ledger(path).record_once("k1") is False      # 全新对象，重新读盘


def _write_then_die(path: str, key: str):
    """写一条，然后不做任何用户态清理就结束进程。"""
    Ledger(path).record_once(key)
    os._exit(0)


def test_ledger_entry_survives_a_process_that_never_cleans_up(tmp_path: Path):
    """进程直接 os._exit()（跳过 atexit / 缓冲区 flush / GC），记录仍在。

    ⚠️ **这条测不到 fsync，我试过了。**

    我原本想用它钉住 fsync：写完立刻 os._exit()，只有落盘的才留下来。
    但 os._exit() 只跳过**用户态**的清理——数据已经交给内核了，
    页缓存是内核维护的，进程死掉它照样在。把 fsync 甚至 flush 整个删掉，
    这条测试依然全绿。

    真要区分「写进了页缓存」和「写进了磁盘」，得断电或者丢内核缓存，
    而那在单元测试里做不到。

    📌 所以这条测试的真实覆盖范围是：**用户态缓冲区里的数据不会丢**。
    fsync 那一行在代码里是有理由的（防的是整机断电），
    但它**没有测试保护**——这一点写在这里，而不是假装测到了。
    第 8 章 §8.8 那条判据的另一面：**知道自己没测到什么，和测到一样重要。**
    """
    path = tmp_path / "ledger.jsonl"
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_write_then_die, args=(str(path), "k-crash"))
    proc.start()
    proc.join(15)

    assert path.exists(), "台账文件根本没建"
    assert Ledger(path).seen("k-crash") is True


def test_ledger_survives_a_torn_line(tmp_path: Path):
    """崩溃时写了半行 → 跳过它，不要让整个台账读不出来。"""
    path = tmp_path / "ledger.jsonl"
    Ledger(path).record_once("good")
    with path.open("a", encoding="utf-8") as f:
        f.write('{"key": "half-writ')                   # 断电
    led = Ledger(path)
    assert led.seen("good") is True
    assert led.record_once("new") is True


def test_compact_drops_old_records_and_forgets_them(tmp_path: Path):
    path = tmp_path / "ledger.jsonl"
    led = Ledger(path, retain_seconds=1.0)
    led.record_once("old")
    time.sleep(1.1)
    led.record_once("fresh")

    assert led.compact() == 1
    assert led.seen("fresh") is True
    # 过期的已经忘掉了——重复投递最晚会在保留窗口内到达，超出的本来也防不住
    assert led.seen("old") is False


def test_ledger_does_not_rescan_the_file_on_every_call(tmp_path: Path):
    """⚠️ 全量扫文件是个隐性的性能炸弹，只在跑了很久之后才出现。

    这里用「删掉文件之后仍然认得出已见过的 key」来间接证明它在用内存索引。
    """
    path = tmp_path / "ledger.jsonl"
    led = Ledger(path)
    led.record_once("k")
    path.unlink()
    assert led.seen("k") is True


# ------------------------------------------------------------------ 死信


def test_dead_letter_stores_enough_to_replay(tmp_path: Path):
    """⚠️ 死信不是错误日志。判据：里面的东西够不够重放这次任务。"""
    dl = DeadLetter(tmp_path / "dead.jsonl")
    dl.put("task-1", "工具连续三次 503",
           payload={"email": "咖啡机坏了"},
           trace_id="tr-9",
           done_keys=["k-ticket-created"])
    (rec,) = dl.items()
    assert rec["payload"] == {"email": "咖啡机坏了"}
    assert rec["trace_id"] == "tr-9"
    # 已执行动作的幂等键必须在——否则重放时不知道工单开没开
    assert rec["done_keys"] == ["k-ticket-created"]


# ------------------------------------------------------------------ 活性


def test_heartbeat_detects_a_stall_even_while_still_beating(tmp_path: Path):
    """⚠️ 这条是这个文件里最重要的一条。

    一个卡在网络调用里的进程，心跳线程照样在跳。
    只判「有没有心跳」的实现会认为一切正常——而它正卡着。
    判据必须是**进度有没有推进**。
    """
    hb = Heartbeat(tmp_path / "hb.json", stall_after=60)
    t0 = 1000.0
    hb.beat(step=3, now=t0)
    for k in range(1, 20):                    # 一直在跳，但 step 不动
        hb.beat(step=3, now=t0 + k * 10)
    assert hb.is_stalled(now=t0 + 200) is True


def test_heartbeat_is_not_stalled_while_steps_advance(tmp_path: Path):
    hb = Heartbeat(tmp_path / "hb.json", stall_after=60)
    t0 = 1000.0
    for k in range(6):
        hb.beat(step=k, now=t0 + k * 30)      # 每 30 秒推进一步
    assert hb.is_stalled(now=t0 + 6 * 30) is False


def test_never_started_is_not_reported_as_stalled(tmp_path: Path):
    assert Heartbeat(tmp_path / "none.json").is_stalled() is False
