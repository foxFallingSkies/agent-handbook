"""评测统计工具的测试。手册第 8 章 §8.9。"""

from __future__ import annotations

import math

import pytest

from evals.stats import mcnemar, min_detectable_gap, repeats_for, wilson


# ------------------------------------------------------------------ 区间


def test_wilson_never_leaves_the_unit_interval():
    """⚠️ 这条是不用 Wald 区间的唯一理由。

    Wald 在 p 接近 1 时上界会超过 1（19/20 时是 1.046），
    而 agent 评测恰好总在那个区间工作。一个上界 > 1 的置信区间不是保守，
    是错的——而它看起来完全正常，会被原样抄进汇报。
    """
    for c, n in ((0, 5), (19, 20), (20, 20), (48, 50), (1, 3), (99, 100)):
        lo, hi = wilson(c, n)
        assert 0.0 <= lo <= hi <= 1.0, f"{c}/{n} → [{lo}, {hi}]"


def test_wilson_has_nonzero_width_at_the_extremes():
    """⚠️ 全对不等于「100%，误差 ±0」。

    Wald 在 c=n 时宽度**恰好是 0**——20 条全对会被报成一个确定的 1.0。
    这是本模块存在的第二个理由。
    """
    lo, hi = wilson(20, 20)
    assert hi - lo > 0.1, f"20/20 的区间宽度只有 {hi - lo}"
    assert lo < 0.9          # 20 条全对，下界不该高到让人误以为很确定

    # 对照：Wald 在这里是退化的
    p = 1.0
    wald_half = 1.96 * math.sqrt(p * (1 - p) / 20)
    assert wald_half == 0.0


def test_interval_narrows_as_n_grows():
    widths = [wilson(round(0.85 * n), n)[1] - wilson(round(0.85 * n), n)[0]
              for n in (10, 20, 50, 100, 200, 500)]
    assert all(a > b for a, b in zip(widths, widths[1:])), widths


def test_twenty_cases_cannot_separate_085_from_075():
    """📌 把第 8 章那句「至少 20 条」的代价钉住。

    20 条时区间宽 0.3 出头，所以 0.85 和 0.75 在这套评测集上是**分不开的**。
    这条测试的作用是：如果哪天有人把书里的门槛改小，它会红。
    """
    assert min_detectable_gap(20) > 0.25
    assert min_detectable_gap(200) < 0.12


def test_wilson_rejects_impossible_inputs():
    with pytest.raises(ValueError):
        wilson(3, 0)
    with pytest.raises(ValueError):
        wilson(5, 3)


# ------------------------------------------------------------------ 重复次数


def test_repeats_must_exceed_k_not_merely_reach_it():
    """⚠️ 上一版这条写的是 `assert repeats_for(k) >= k`——**它是假绿**。

    `return k` 也满足 `>= k`，于是这条测试对「贴着下限跑」一言不发。
    变异测试当场抓到：把 `k + margin` 改成 `k`，13 条测试全绿。

    真正要钉住的不是那个不等号，是它背后的理由：**n = k 时
    pass^k = C(c,k)/C(n,k) 只有 0 和 1 两个取值**，它不是一个估计，
    是一次抽样。所以下面直接去测估计量本身的退化。
    """
    from evals.run import pass_pow_k

    for k in (2, 3, 5):
        n = k
        vals = {pass_pow_k(n, c, k) for c in range(n + 1)}
        assert vals == {0.0, 1.0}, f"n=k={k} 时居然有中间值：{sorted(vals)}"

        n2 = repeats_for(k)
        assert n2 > k
        vals2 = {pass_pow_k(n2, c, k) for c in range(n2 + 1)}
        assert vals2 - {0.0, 1.0}, f"n={n2} 时仍然只有 0/1，margin 没起作用"


# ------------------------------------------------------------------ 配对比较


def test_only_discordant_pairs_carry_information():
    """🔑 McNemar 的核心：两次都对、两次都错的用例一言不发。

    ⚠️ 这条同时钉住一个常见误解——「我有 100 条用例所以样本量是 100」。
    如果只有 3 条结果变了，**有效样本量就是 3**。
    """
    before = [True] * 97 + [False] * 3
    after = [True] * 100
    r = mcnemar(before, after)
    assert r["n"] == 100
    assert r["discordant"] == 3
    assert r["better"] == 3 and r["worse"] == 0

    # 把 97 条「两次都对」的用例扩成 997 条，结论不能变
    r2 = mcnemar([True] * 997 + [False] * 3, [True] * 1000)
    assert r2["p_value"] == pytest.approx(r["p_value"])


def test_three_wins_out_of_a_hundred_is_not_enough():
    """📌 反直觉但重要：100 条里 3 条变好、0 条变坏，**判不出来**（p=0.25）。

    多数人会认为"只赢不输"就是改进了。它可能是，但这份数据证明不了。
    """
    r = mcnemar([True] * 97 + [False] * 3, [True] * 100)
    assert not r["p_below_05"]
    assert r["p_value"] == pytest.approx(0.25)


def test_a_real_improvement_is_detected():
    r = mcnemar([True] * 91 + [False] * 8 + [True],
                [True] * 99 + [False])
    assert r["better"] == 8 and r["worse"] == 1
    assert r["p_below_05"]


def test_equal_wins_and_losses_is_a_wash():
    """5 条变好、4 条变坏 → 什么都没发生。总分可能涨了 1 个百分点。"""
    r = mcnemar([True] * 91 + [False] * 5 + [True] * 4,
                [True] * 96 + [False] * 4)
    assert r["better"] == 5 and r["worse"] == 4
    assert not r["p_below_05"]


def test_mcnemar_is_symmetric_in_direction():
    """变好 8 / 变坏 1，和变好 1 / 变坏 8，p 值必须相同——它只判「是不是噪声」。

    ⚠️ 方向要看 better/worse 两个计数，不能看 p 值。
    把 p 值当成"变好了"的证据，是这个检验最容易被误用的方式。
    """
    a = mcnemar([True] * 91 + [False] * 8 + [True], [True] * 99 + [False])
    b = mcnemar([True] * 99 + [False], [True] * 91 + [False] * 8 + [True])
    assert a["p_value"] == pytest.approx(b["p_value"])
    assert a["better"] == b["worse"]


def test_no_change_at_all():
    r = mcnemar([True, False, True], [True, False, True])
    assert r["discordant"] == 0 and r["p_value"] == 1.0


def test_mismatched_lengths_are_loud():
    with pytest.raises(ValueError, match="用例数不一致"):
        mcnemar([True, False], [True])
