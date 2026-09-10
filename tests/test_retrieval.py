"""检索层的测试。手册第 6 章。

⚠️ 同样遵守 §8.8：每条测试针对一个**具体的退化实现**，
而不是泛泛地测「能检索到东西」。
"""

from __future__ import annotations

import pytest

from handbook.retrieval import (
    BM25Retriever,
    Doc,
    VectorRetriever,
    evaluate,
    load_corpus,
    load_golden,
    mrr,
    recall_at_k,
    rrf_fuse,
    tokenize,
)


# ------------------------------------------------------------------ 分词


def test_identifiers_survive_tokenization():
    """⚠️ 这条是 §6.4「精确匹配是 BM25 的主场」的前提。

    如果分词把 `ord_20260803_a17c` 切成 ord / 20260803 / a17c，
    那么查订单号就退化成查「ord」——所有订单都命中，等于没查。
    """
    t = tokenize("订单 ord_20260803_a17c 和工单 TCK-0001")
    assert "ord_20260803_a17c" in t
    assert "tck-0001" in t


def test_cjk_is_bigrammed_not_split_into_chars():
    """中文用字符 bigram：「保修政策」→ 保修/修政/政策。

    ⚠️ 退化成单字会让 IDF 失效（单字到处都是），
    退化成整串会让「保修」这个查询对不上「延长保修」。
    """
    t = tokenize("保修政策")
    assert "保修" in t and "政策" in t
    assert "保修政策" not in t          # 不是整串
    assert "保" not in t                 # 也不是单字


# ------------------------------------------------------------------ BM25


def test_idf_is_never_negative():
    """⚠️ 教科书 IDF 在 df > N/2 时会变成负数。

    后果很隐蔽：一个出现在多数文档里的词，会把**包含它的**文档往后推。
    表现是「越相关排得越靠后」，而没人会怀疑到 IDF 头上。
    这里造一个词出现在 4/5 篇里的语料来钉死它。
    """
    docs = [Doc(f"d{i}", "标题", "保修 " + "xyz " * i) for i in range(5)]
    r = BM25Retriever(docs)
    assert all(v >= 0 for v in r._idf.values()), \
        {w: v for w, v in r._idf.items() if v < 0}


def test_bm25_finds_an_exact_identifier():
    """§6.4 的核心断言，做成一条可执行的测试。"""
    docs = load_corpus()
    hits = BM25Retriever(docs).search("ord_20260803_a17c 是什么意思", k=3)
    assert hits, "订单号查询一条都没召回"
    assert hits[0].doc_id == "faq_ordid"


def test_hit_carries_the_key_to_fetch_it_back():
    """§3.6：折叠可以，但必须留下取回的钥匙。检索结果同理。"""
    docs = load_corpus()
    h = BM25Retriever(docs).search("保修", k=1)[0]
    assert h.source.startswith("kb://")
    assert h.rank_by == {"bm25": 1}


# ------------------------------------------------------------------ RRF


def test_rrf_uses_rank_only_not_score():
    """⚠️ 这条是 RRF 存在的**全部理由**（§6.5）。

    两路的分数量纲差 1000 倍，但排名相同。
    只用排名的实现，两个文档的融合得分必须相等；
    一旦有人"顺手"把 score 也加进来，A 会被巨大的分数推到前面。
    """
    from handbook.retrieval import Hit

    run1 = [Hit("A", "a", score=9999.0, source="x", rank_by={"r1": 1}),
            Hit("B", "b", score=9998.0, source="x", rank_by={"r1": 2})]
    run2 = [Hit("B", "b", score=0.02, source="x", rank_by={"r2": 1}),
            Hit("A", "a", score=0.01, source="x", rank_by={"r2": 2})]

    fused = rrf_fuse([run1, run2], k=2)
    assert {h.doc_id for h in fused} == {"A", "B"}
    a, b = (h for h in sorted(fused, key=lambda h: h.doc_id))
    assert a.score == pytest.approx(b.score), \
        f"A={a.score} B={b.score}：排名对称的两个文档得分不等，说明分数漏进来了"


def test_rrf_keeps_both_runs_rankings_for_debugging():
    """rank_by 要保住每一路给的排名——排查融合问题时它是唯一的线索。"""
    from handbook.retrieval import Hit

    fused = rrf_fuse([[Hit("A", "a", 1.0, "x", {"bm25": 1})],
                      [Hit("A", "a", 1.0, "x", {"vector": 7})]], k=1)
    assert fused[0].rank_by == {"bm25": 1, "vector": 7}


def test_rrf_k_is_keyword_only():
    """⚠️ 两个 k 混淆是这个函数最容易犯的错，所以 rrf_k 只能按关键字传。"""
    from handbook.retrieval import Hit

    run = [Hit("A", "a", 1.0, "x", {"r": 1})]
    with pytest.raises(TypeError):
        rrf_fuse([run], 5, 60)          # type: ignore[misc]


# ------------------------------------------------------------------ 指标


def test_recall_denominator_is_the_relevant_set_not_k():
    """⚠️ 写成「命中数 / k」是常见错误。

    只有 1 篇相关文档的用例，正确答案是 1.0；除以 k=5 会得到 0.2，
    于是你的天花板看起来比实际低——然后你去优化一个不存在的问题。
    """
    assert recall_at_k(["a", "b", "c", "d", "e"], ["a"], k=5) == 1.0
    assert recall_at_k(["a", "b"], ["a", "z"], k=5) == 0.5
    assert recall_at_k(["x"], ["a"], k=5) == 0.0


def test_recall_respects_the_cutoff():
    """第 6 个位置上的命中，在 recall@5 里不算。"""
    assert recall_at_k(["x", "x", "x", "x", "x", "a"], ["a"], k=5) == 0.0
    assert recall_at_k(["x", "x", "x", "x", "x", "a"], ["a"], k=6) == 1.0


def test_mrr_is_the_reciprocal_of_the_first_hit():
    assert mrr(["a", "b"], ["a"]) == 1.0
    assert mrr(["x", "a"], ["a"]) == 0.5
    assert mrr(["x", "y"], ["a"]) == 0.0


# ------------------------------------------------------------------ 端到端


def test_the_ceiling_is_reported_with_its_misses():
    """§6.9：评测要给出**漏掉的是哪几条**，不能只给一个总分。

    ⚠️ 一个只返回百分比的评测，没法告诉你下一步改什么——
    而「下一步改什么」才是评测存在的理由（§8.3）。
    """
    docs = load_corpus()
    m = evaluate(BM25Retriever(docs), load_golden(), k=5)
    assert 0.0 < m["recall_at_k"] < 1.0, "语料/金标退化了，天花板不该是 0 或 1"
    assert m["misses"], "有 recall<1 的用例，misses 却是空的"
    assert all("want" in r and "got" in r for r in m["misses"])


def test_bag_of_ngrams_vector_is_not_semantic():
    """⚠️ 这条测试断言的是一个**局限**，不是一个功能。

    仓库自带的 embed 是词袋，不是语义模型。所以它和 BM25 一样，
    在「东西不想要了能退吗」vs「七天无理由退货」这种**没有共同词**的
    查询上会一起失败。

    📌 这正是 §6.5 混合检索的前提被抽掉之后的样子：
    混合有用的前提是**两路的弱点互补**；两路信号相同时，融合没有信息可加。
    把这条钉住，是为了防止有人看到 `VectorRetriever` 这个名字，
    以为仓库里已经有语义检索了。
    """
    docs = load_corpus()
    q = "东西不想要了能退吗"
    bm = [h.doc_id for h in BM25Retriever(docs).search(q, 5)]
    ve = [h.doc_id for h in VectorRetriever(docs).search(q, 5)]
    assert "pol_return" not in bm
    assert "pol_return" not in ve, "词袋向量检索居然命中了语义匹配——那它不是词袋"


# ------------------------------------------------------------------ 三因子取回


def _mem(now):
    from handbook.retrieval import Recallable
    return [
        # 三年前的记录，对"咖啡机型号"极度贴题——但已经过期了
        Recallable("old", "客户的咖啡机型号是 X100", importance=6.0,
                   last_access=now - 3600 * 24 * 1000),
        # 上周的记录，同样提到型号，而且是现在成立的那条
        Recallable("new", "客户的咖啡机型号换成了 X200", importance=6.0,
                   last_access=now - 3600 * 24 * 7),
    ]


def test_recency_rescues_the_stale_but_relevant_record():
    """§7.6 的核心论断，做成可执行的：

    ⚠️ 纯相似度会让一条三年前、非常贴题但**已经过期**的记录，
    排在那条仍然成立的记录前面。recency 因子存在的全部意义就是压住这个。

    这里两条记录的文本几乎一样（BM25 分数接近），importance 也相同，
    **唯一的变量是上次访问时间**——所以排序结果只能由 recency 决定。
    """
    import time

    from handbook.retrieval import bm25_relevance, three_factor_rank

    now = time.time()
    ranked = three_factor_rank("咖啡机型号", _mem(now), now=now,
                               relevance=bm25_relevance(), k=2)
    assert ranked[0][0].item_id == "new"


def test_recency_uses_last_access_not_a_constant():
    """⚠️ 防的是「recency 因子写了但没起作用」这个假绿。

    把 last_access 换成常数（或者干脆忽略它），上面那条测试会红，
    这条也会红——两条一起才说明这个因子真的参与了排序。
    """
    import time

    from handbook.retrieval import bm25_relevance, three_factor_rank

    now = time.time()
    items = _mem(now)
    got = {it.item_id: p["recency"] for it, p in
           three_factor_rank("咖啡机型号", items, now=now,
                             relevance=bm25_relevance(), k=2)}
    assert got["new"] > got["old"]


def test_factors_are_normalized_over_the_actual_spread():
    """⚠️ 归一化必须按这一批候选的实际分布，不能按理论量程。

    按理论量程（importance / 10）的话，(5.0, 5.1) 这两条的差别是 0.01，
    永远压不过另外两个因子——于是你调 weights 怎么调都不对，
    因为你调的不是你以为的那个东西。
    """
    from handbook.retrieval import _minmax

    assert _minmax([5.0, 5.1]) == [0.0, 1.0]
    assert _minmax([1.0, 10.0]) == [0.0, 1.0]
    # 全都相等 = 这个因子不携带排序信息，贡献 0 而不是 0.5
    assert _minmax([7.0, 7.0]) == [0.0, 0.0]


def test_weights_can_ablate_a_factor_and_the_order_actually_flips():
    """weights 是给消融用的。这条测试要求消融**真的改变结果**。

    ⚠️ 这里原本写的是 `assert a != b or True` —— 一条恒真断言，
    永远通过，什么也没测到。第 8 章 §8.4 说「负例最容易写成恒真」，
    而它当场就发生在这个文件里。留下这段注释而不是悄悄改掉。

    现在的构造让两个因子指向相反的方向：
      relevance 偏向 old（注入的分数写死，old 更高）
      recency   偏向 new（old 是三年前的）
    于是关掉哪个因子，第一名就该换人。**换不了人，就说明权重没接上。**
    """
    import time

    from handbook.retrieval import three_factor_rank

    now = time.time()
    rel = lambda q, items: [1.0 if it.item_id == "old" else 0.2 for it in items]

    only_relevance = three_factor_rank("q", _mem(now), now=now,
                                       relevance=rel, weights=(0, 0, 1), k=2)
    only_recency = three_factor_rank("q", _mem(now), now=now,
                                     relevance=rel, weights=(1, 0, 0), k=2)

    assert only_relevance[0][0].item_id == "old"
    assert only_recency[0][0].item_id == "new"

    # 每条结果都要带着三个因子的分值，否则"为什么它排前面"答不上来（§9.2）
    assert set(only_recency[0][1]) == {"recency", "importance", "relevance"}


def test_relevance_length_mismatch_is_loud():
    """⚠️ 注入式设计的代价：传错了长度，静默错位比报错难查得多。"""
    import time

    from handbook.retrieval import three_factor_rank

    with pytest.raises(ValueError, match="relevance"):
        three_factor_rank("q", _mem(time.time()), now=time.time(),
                          relevance=lambda q, items: [1.0])
