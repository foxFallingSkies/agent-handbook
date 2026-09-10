"""检索层的单独评测。手册第 6 章 §6.9。

    python -m evals.retrieval

⚠️ 这个脚本存在的理由就是那一节的论点：**检索必须单独评测。**
端到端跑一遍「答得对不对」，答错了你分不清是没找到还是找到了没用好。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from handbook.retrieval import (  # noqa: E402
    BM25Retriever, VectorRetriever, evaluate, load_corpus, load_golden, rrf_fuse,
)


class Hybrid:
    """BM25 + 向量，RRF 融合。§6.5"""

    def __init__(self, docs):
        self.a = BM25Retriever(docs)
        self.b = VectorRetriever(docs)

    def search(self, query: str, k: int):
        # ⚠️ 每一路都要多召一些再融合。只召 k 条去融合，等于让融合
        # 在两个已经被截断的列表上工作——RRF 能纠正的排序错误，
        # 大部分发生在第 k 条之后。
        wide = max(k * 3, 10)
        return rrf_fuse([self.a.search(query, wide), self.b.search(query, wide)], k)


def main() -> int:
    docs = load_corpus()
    cases = load_golden()

    setups = [
        ("BM25（关键词）", BM25Retriever(docs)),
        ("向量（⚠️ 词袋，非语义）", VectorRetriever(docs)),
        ("混合 + RRF", Hybrid(docs)),
    ]

    print(f"语料 {len(docs)} 篇 · 用例 {len(cases)} 条 · 相关文档为人工标注\n")
    print(f"{'检索器':<26}{'recall@5':>10}{'MRR':>8}   全中的用例")
    print("-" * 68)

    results = {}
    for name, r in setups:
        m = evaluate(r, cases, k=5)
        results[name] = m
        full = sum(1 for row in m["rows"] if row["recall"] >= 1.0)
        print(f"{name:<24}{m['recall_at_k']:>10.1%}{m['mrr']:>8.2f}   {full}/{len(cases)}")

    best = max(results.values(), key=lambda m: m["recall_at_k"])
    print("\n" + "=" * 68)
    print(f"最好的一路：recall@5 = {best['recall_at_k']:.1%}")
    print("=" * 68)
    print("📌 这个数就是整条链路的天花板：即使模型完美利用了给它的资料，")
    print("   没检索到的那部分它也答不出来。生成层再怎么调都够不着。\n")

    if best["misses"]:
        print("没全中的用例（这些是该先修的地方，不是模型的问题）：")
        for row in best["misses"][:6]:
            missing = [d for d in row["want"] if d not in row["got"]]
            print(f"  {row['id']}  {row['query']}")
            print(f"       漏了 {missing}   召回的是 {row['got'][:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
