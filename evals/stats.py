"""评测结果的统计工具。手册第 8 章 §8.9。

    python -m evals.stats

这个模块回答三个在面试和复盘里都躲不掉的问题：

    1. 我的 golden set 要多少条才够？        → wilson()
    2. 一条用例要跑几次？                    → repeats_for()
    3. 从 0.80 到 0.85，是真变好了还是抖动？ → mcnemar()

⚠️ 它刻意不依赖 scipy/numpy。三个函数加起来不到一百行，
而**一个你看不懂的统计库，会让你把"显著"当成一个可以引用的结论**——
第 8 章从头到尾都在说，指标要能被追问到底。
"""

from __future__ import annotations

import math
from typing import Sequence

__all__ = ["wilson", "repeats_for", "mcnemar", "min_detectable_gap"]


# ---------------------------------------------------------------- 区间


def wilson(c: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """成功 c 次 / 共 n 次的 Wilson 得分区间（默认 95%）。

    ⚠️ **不要用教科书那个 `p ± z·sqrt(p(1-p)/n)`（Wald 区间）。**

    agent 评测恰好总在 p 很高的区间工作（0.85、0.9、0.95），而 Wald 在那里
    会给出**越界**的结果：20 条里对 19 条，Wald 区间是 [0.854, 1.046]。
    一个上界大于 1 的置信区间不是保守，是错的——而它看起来非常正常，
    你会把它抄进汇报里。

    Wilson 在同样的数据上给 [0.764, 0.991]，永远落在 [0,1] 内，
    而且在 c=n 时仍然有意义：20 条全对，Wald 给 [1.000, 1.000]——
    **宽度为零**，也就是"100% 正确，误差 ±0"；Wilson 给 [0.839, 1.000]。
    这两个区间在同一份数据上，一个是废话，一个是结论。
    """
    if n <= 0:
        raise ValueError("n 必须大于 0")
    if not 0 <= c <= n:
        raise ValueError(f"c={c} 不在 [0, {n}] 内")
    p = c / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z / d * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def min_detectable_gap(n: int, p: float = 0.85, z: float = 1.96) -> float:
    """在 n 条用例上，pass 率至少要差多少，两个区间才不重叠。

    📌 这是"20 条够不够"的可执行答案。它给的是一个**下限**：
    比这个差距小的改进，你这套评测集看不见——不是它不存在，是你测不出来。
    """
    lo, hi = wilson(round(p * n), n, z)
    return hi - lo


# ---------------------------------------------------------------- 重复次数


def repeats_for(k: int, margin: int = 2) -> int:
    """要报 pass^k / pass@k，一条用例至少跑几次。

    ⚠️ 硬下限是 n ≥ k：组合估计量 C(c,k)/C(n,k) 在 n < k 时无定义。
    但贴着下限跑毫无意义——n = k 时 pass^k 只有 0 和 1 两个取值，
    它不是一个估计，是一次抽样。

    这里给 k + margin，是一个工程上的折中而不是一个统计结论。
    真正的判据是下面那句：**跑到 wilson 区间窄到你能做决定为止。**
    """
    if k < 1:
        raise ValueError("k 必须 ≥ 1")
    return k + margin


# ---------------------------------------------------------------- 配对比较


def mcnemar(before: Sequence[bool], after: Sequence[bool]) -> dict:
    """同一批用例，改动前后各跑一遍，判断差异是不是噪声。

    ⚠️ **不要用两个独立比例的检验（卡方 / 双样本 z）。**

    golden set 是**同一批用例**，前后两次的结果是配对的。当成两组独立样本，
    你丢掉了配对信息，结果是把一个真实的改进判成"不显著"——
    然后你回滚了一个正确的改动。

    🔑 McNemar 只看**不一致的那些用例**：
      b = 改动前对、改动后错   （变坏的）
      c = 改动前错、改动后对   （变好的）

    两次都对、两次都错的用例**不携带任何信息**——它们对"这次改动有没有用"
    这个问题一言不发。这和 §7.11 里"常数因子不参与排序"是同一个道理。

    📌 所以 100 条用例里如果只有 3 条结果变了，你的有效样本量是 3，不是 100。
    这一条比 p 值本身有用得多。
    """
    if len(before) != len(after):
        raise ValueError(f"两次的用例数不一致：{len(before)} vs {len(after)}")
    if not before:
        raise ValueError("空的结果集")

    b = sum(1 for x, y in zip(before, after) if x and not y)   # 变坏
    c = sum(1 for x, y in zip(before, after) if y and not x)   # 变好
    n_disc = b + c

    if n_disc == 0:
        p_value = 1.0
    else:
        # 精确二项检验，H0: 变好和变坏各占一半
        lo = min(b, c)
        tail = sum(math.comb(n_disc, i) for i in range(lo + 1)) / (2 ** n_disc)
        p_value = min(1.0, 2 * tail)

    return {
        "n": len(before),
        "worse": b,
        "better": c,
        "discordant": n_disc,
        "p_value": p_value,
        # ⚠️ 不叫 "significant"。0.05 是一条惯例，不是一条自然律，
        # 而给它起名叫「显著」会让下游代码把它当成一个事实。
        "p_below_05": p_value < 0.05,
    }


# ---------------------------------------------------------------- 演示


def main() -> int:
    print("① 你的 golden set 有多宽的误差棒？（pass 率按 0.85 算，95% 区间）\n")
    print(f"{'用例数':>8}{'区间':>22}{'宽度':>10}   能看见多大的改进")
    print("-" * 66)
    for n in (10, 20, 30, 50, 100, 200, 500):
        lo, hi = wilson(round(0.85 * n), n)
        w = hi - lo
        verdict = ("⚠️ 只能看见天差地别" if w > 0.30 else
                   "大改动看得见" if w > 0.15 else
                   "中等改动看得见" if w > 0.09 else "细节改动也看得见")
        print(f"{n:>8}   [{lo:.3f}, {hi:.3f}]{w:>10.3f}   {verdict}")

    print("\n📌 第 8 章 §8.4 说「至少 20 条」。上面这张表是那句话的代价："
          f"\n   20 条时区间宽 {min_detectable_gap(20):.2f}——"
          "0.85 和 0.75 在这套评测集上是分不开的。\n")

    print("② Wald 区间在高 pass 率下会越界（所以本模块用 Wilson）\n")
    for c, n in ((19, 20), (20, 20), (48, 50)):
        p = c / n
        half = 1.96 * math.sqrt(p * (1 - p) / n)
        lo, hi = wilson(c, n)
        flag = "  ← 上界 > 1，不合法" if p + half > 1.0 else ""
        print(f"  {c}/{n}:  Wald [{p - half:.3f}, {p + half:.3f}]{flag}")
        print(f"          Wilson [{lo:.3f}, {hi:.3f}]")

    print("\n③ 改动前后比较：只有结果变了的用例携带信息\n")
    demos = [
        ("100 条里 3 条变好、0 条变坏", [True] * 97 + [False] * 3, [True] * 100),
        ("100 条里 8 条变好、1 条变坏",
         [True] * 91 + [False] * 8 + [True], [True] * 99 + [False]),
        ("100 条里 5 条变好、4 条变坏",
         [True] * 91 + [False] * 5 + [True] * 4, [True] * 96 + [False] * 4),
    ]
    for name, a, b in demos:
        r = mcnemar(a, b)
        print(f"  {name}")
        print(f"     有效样本量 {r['discordant']}（不是 {r['n']}）  "
              f"p = {r['p_value']:.4f}  {'可以认为真的变好了' if r['p_below_05'] else '⚠️ 分不出来，别急着下结论'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
