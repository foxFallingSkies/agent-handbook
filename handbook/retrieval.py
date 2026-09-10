"""检索层。手册第 6 章。

这个模块**刻意没有依赖任何向量数据库**，理由在 §6.10：
本书的判据是「你的资料有没有天然的组织结构」，而演示这条判据不需要一个索引服务。

它提供的是第 6 章那几条论断的**可执行版本**：

    BM25Retriever   关键词检索。精确匹配强（§6.4）
    VectorRetriever 语义检索。⚠️ embed 函数由你注入，见下面那段说明
    rrf_fuse        倒数排名融合。只用排名，不用分数（§6.5）
    recall_at_k     整条链路的天花板（§6.9）
    mrr             第一个相关结果排第几

⚠️ **关于 VectorRetriever 里那个自带的 embed：它不是语义的。**
本仓库不引入任何模型权重，所以自带的 `bag_of_ngrams_embedding` 只是把
文本映射成词袋向量——它测得了**管道**（融合、排序、指标算得对不对），
测不了**语义**。真正的语义检索需要你传进来一个真的 embedding 函数。
这一点写在这里，而不是让 `VectorRetriever` 这个名字自己去暗示。
第 8 章 §8.8 的判据在这里同样成立：**知道自己没测到什么，和测到一样重要。**
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence

__all__ = [
    "Doc", "Hit", "Retriever", "tokenize", "BM25Retriever", "VectorRetriever",
    "bag_of_ngrams_embedding", "rrf_fuse", "recall_at_k", "mrr", "evaluate",
    "load_corpus", "load_golden",
    "Recallable", "three_factor_rank", "bm25_relevance",
]


# ---------------------------------------------------------------- 数据结构


@dataclass(frozen=True)
class Doc:
    doc_id: str
    title: str
    text: str

    @property
    def searchable(self) -> str:
        # 标题参与检索。标题里往往是最精确的那几个词（"延长保修"），
        # 而正文会把它稀释掉。
        return f"{self.title}\n{self.text}"


@dataclass
class Hit:
    """一条检索结果。字段形状由第 6 章 §6.11 的接口设计确定。"""

    doc_id: str
    text: str
    score: float
    source: str                              # ← 取回的钥匙（§3.6）
    rank_by: dict[str, int] = field(default_factory=dict)   # 各路给的排名，供调试


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[Hit]: ...


# ---------------------------------------------------------------- 分词


_CJK = r"一-鿿"
_ASCII_WORD = re.compile(r"[a-zA-Z0-9_][a-zA-Z0-9_\-.]*")
_CJK_RUN = re.compile(f"[{_CJK}]+")


def tokenize(text: str) -> list[str]:
    """中英混排的分词。

    ⚠️ **中文这里用的是字符 bigram，不是分词器。这是刻意的。**

    一个错的分词器比没有分词器更难查：它会静默地把「延长保修」切成
    「延长」「保修」还算好，切成「延」「长保」「修」就完全检索不到了，
    而你在 recall 掉下来的时候只会看到「向量检索不行」这个错误结论。

    bigram 的性质是可预测的：「延长保修」→ 延长/长保/保修，
    任何一个查询里出现「保修」都能对上。覆盖率换精度，而 BM25 的
    IDF 会把「保修」这种到处都是的 bigram 自动降权——**两个机制配合，
    正好补上不分词的那个洞**。

    ASCII 侧保留 `_` `-` `.`，因为 `ord_20260803_a17c` 和 `TCK-0001`
    这类标识符整体才有意义。§6.4 说的向量检索最弱的地方，就是这些。
    """
    out: list[str] = []
    for m in _ASCII_WORD.finditer(text):
        out.append(m.group(0).lower())
    for m in _CJK_RUN.finditer(text):
        run = m.group(0)
        if len(run) == 1:
            out.append(run)
        else:
            out.extend(run[i:i + 2] for i in range(len(run) - 1))
    return out


# ---------------------------------------------------------------- BM25


@dataclass
class BM25Retriever:
    """经典 BM25。§6.4：精确匹配强，专有名词、ID、代码符号是它的主场。"""

    docs: Sequence[Doc]
    k1: float = 1.5
    b: float = 0.75
    name: str = "bm25"

    def __post_init__(self) -> None:
        self._toks = [tokenize(d.searchable) for d in self.docs]
        self._tf = [Counter(t) for t in self._toks]
        self._len = [len(t) for t in self._toks]
        self._avglen = (sum(self._len) / len(self._len)) if self._len else 0.0
        df: Counter[str] = Counter()
        for t in self._toks:
            df.update(set(t))
        n = len(self.docs)
        # ⚠️ 用带 +1 的平滑 IDF。教科书里那个 log((N-df+0.5)/(df+0.5)) 在
        # df > N/2 时**会变成负数**——一个出现在多数文档里的词，会把包含它的
        # 文档往后推。表现是「越相关排得越靠后」，而没人会怀疑到 IDF 上。
        self._idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def search(self, query: str, k: int) -> list[Hit]:
        q = tokenize(query)
        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self._tf):
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self._len[i] / (self._avglen or 1))
                s += self._idf.get(w, 0.0) * f * (self.k1 + 1) / denom
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda x: (-x[0], self.docs[x[1]].doc_id))
        return [
            Hit(doc_id=self.docs[i].doc_id, text=self.docs[i].text, score=s,
                source=f"kb://{self.docs[i].doc_id}", rank_by={self.name: r + 1})
            for r, (s, i) in enumerate(scored[:k])
        ]


# ---------------------------------------------------------------- 向量


Embed = Callable[[Sequence[str]], list[list[float]]]


def bag_of_ngrams_embedding(texts: Sequence[str]) -> list[list[float]]:
    """⚠️ **这不是语义 embedding。** 它是词袋，用来测管道，不是测语义。

    留它在这里是为了让 `VectorRetriever` 能在没有网络、没有模型权重的情况下
    跑起来并被测试。真要做语义检索，把你的 embedding 函数传进去。

    命名上没有藏起来这件事：它叫 `bag_of_ngrams`，不叫 `default_embedding`。
    """
    vocab: dict[str, int] = {}
    rows = []
    for t in texts:
        c = Counter(tokenize(t))
        rows.append(c)
        for w in c:
            vocab.setdefault(w, len(vocab))
    out = []
    for c in rows:
        v = [0.0] * len(vocab)
        for w, n in c.items():
            v[vocab[w]] = float(n)
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / norm for x in v])
    return out


@dataclass
class VectorRetriever:
    """语义检索。embed 由调用方注入——见模块 docstring 里那段说明。"""

    docs: Sequence[Doc]
    embed: Embed = bag_of_ngrams_embedding
    name: str = "vector"

    def __post_init__(self) -> None:
        # ⚠️ 文档和查询必须用**同一个** embed 调用来编码，否则
        # bag_of_ngrams 这种依赖语料建词表的实现会得到不同维度的向量。
        # 真 embedding 模型没有这个问题，但接口要能容纳两者。
        self._texts = [d.searchable for d in self.docs]

    def search(self, query: str, k: int) -> list[Hit]:
        vecs = self.embed([query, *self._texts])
        qv, dvs = vecs[0], vecs[1:]
        scored = []
        for i, dv in enumerate(dvs):
            s = sum(a * b for a, b in zip(qv, dv))
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda x: (-x[0], self.docs[x[1]].doc_id))
        return [
            Hit(doc_id=self.docs[i].doc_id, text=self.docs[i].text, score=s,
                source=f"kb://{self.docs[i].doc_id}", rank_by={self.name: r + 1})
            for r, (s, i) in enumerate(scored[:k])
        ]


# ---------------------------------------------------------------- 融合


def rrf_fuse(runs: Iterable[list[Hit]], k: int, *, rrf_k: int = 60) -> list[Hit]:
    r"""倒数排名融合：score(d) = Σ 1/(rrf_k + rank_r(d))。§6.5

    ⚠️ **两个 k 不是同一个东西**，这是这个函数最容易出错的地方：
      - `k`      要返回几条
      - `rrf_k`  RRF 公式里的平滑常数（惯例 60）
    把它们写成一个参数，或者把 60 传进 `k`，都不会报错，只会让结果变差。
    所以这里强制 `rrf_k` 只能按关键字传。

    ⚠️ **本函数只读 rank，不读 score。** 这是 RRF 的全部意义：
    BM25 的分数和余弦相似度不在一个量纲上，加权求和既要调参又不稳。
    如果哪天有人"顺手"把 score 也加进来，这个设计就没了。
    """
    agg: dict[str, float] = defaultdict(float)
    keep: dict[str, Hit] = {}
    ranks: dict[str, dict[str, int]] = defaultdict(dict)

    for run in runs:
        for rank, h in enumerate(run, start=1):
            agg[h.doc_id] += 1.0 / (rrf_k + rank)
            keep.setdefault(h.doc_id, h)
            ranks[h.doc_id].update(h.rank_by or {})

    order = sorted(agg.items(), key=lambda x: (-x[1], x[0]))
    out = []
    for doc_id, s in order[:k]:
        h = keep[doc_id]
        out.append(Hit(doc_id=doc_id, text=h.text, score=s, source=h.source,
                       rank_by=dict(ranks[doc_id])))
    return out


# ---------------------------------------------------------------- 指标


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """§6.9：**整条链路的天花板。**

    ⚠️ 分母是「该命中的总数」，不是 k。写成 `命中数 / k` 是一个常见错误，
    它会让「只有 1 篇相关文档」的用例永远拿不到 1.0，于是你的天花板
    看起来比实际低——然后你去优化一个并不存在的问题。
    """
    if not relevant:
        return 1.0
    got = set(retrieved[:k]) & set(relevant)
    return len(got) / len(relevant)


def mrr(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """第一个相关结果排第几的倒数。没命中就是 0。"""
    rel = set(relevant)
    for i, d in enumerate(retrieved, start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def evaluate(retriever: Retriever, cases: Sequence[dict], k: int = 5) -> dict:
    """跑一遍 §6.9 那个「20 个真实问题，只看 recall@5」的最小评测。"""
    rows = []
    for c in cases:
        hits = retriever.search(c["query"], k)
        ids = [h.doc_id for h in hits]
        rows.append({
            "id": c["id"],
            "query": c["query"],
            "recall": recall_at_k(ids, c["relevant"], k),
            "mrr": mrr(ids, c["relevant"]),
            "got": ids,
            "want": list(c["relevant"]),
        })
    n = len(rows) or 1
    return {
        "k": k,
        "recall_at_k": sum(r["recall"] for r in rows) / n,
        "mrr": sum(r["mrr"] for r in rows) / n,
        "misses": [r for r in rows if r["recall"] < 1.0],
        "rows": rows,
    }


# ---------------------------------------------------------------- 语料


def load_corpus(path: str | Path = "data/kb.jsonl") -> list[Doc]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out.append(Doc(doc_id=d["doc_id"], title=d["title"], text=d["text"]))
    return out


def load_golden(path: str | Path = "data/kb_golden.json") -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["cases"]


# ---------------------------------------------------------------- 三因子取回


@dataclass
class Recallable:
    """一条可被取回的记忆。手册第 7 章 §7.6（Generative Agents）。"""

    item_id: str
    text: str
    importance: float          # 1–10，⚠️ **写入时**由模型打好，不是取回时算
    last_access: float         # epoch 秒


def _minmax(vals: Sequence[float]) -> list[float]:
    """把一组分数归一化到 [0,1]。

    ⚠️ 必须按**这一批候选的实际分布**归一化，不能按各因子的理论量程。
    importance 除以 10、relevance 用余弦原值（0~1）、recency 用衰减原值——
    三者的"1 分"含义完全不同，加起来时权重就名不副实了。
    症状是你调 weights 怎么调都不对，因为你调的不是你以为的那个东西。

    全都相等时返回全 0：一个不区分候选的因子，对排序**不携带任何信息**，
    让它贡献 0 比让它贡献 0.5 更诚实（后者只是把常数加到每个人头上）。
    """
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return [0.0] * len(vals)
    return [(v - lo) / (hi - lo) for v in vals]


def three_factor_rank(
    query: str,
    items: Sequence[Recallable],
    *,
    now: float,
    relevance: Callable[[str, Sequence[Recallable]], Sequence[float]],
    half_life_hours: float = 24.0,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    k: int = 5,
) -> list[tuple[Recallable, dict[str, float]]]:
    """recency + importance + relevance，各自归一化后加权求和。§7.6

    📌 **只有第三个因子是检索。** 第 6 章那一整章的东西，在这里占三分之一——
    这是本函数存在的理由：纯相似度会让一条三年前、非常贴题但已经过期的记录，
    排在昨天那条相关性稍低但仍然成立的记录前面。

    `relevance` 是注入的，所以这一层不绑定任何 embedding 实现
    （可以传 BM25、传真 embedding、传一个常量函数做消融）。

    返回 (记忆, 三个因子的归一化分值)，第二项是给排查用的——
    没有它，"为什么这条排前面"就答不上来（§9.2）。
    """
    if not items:
        return []

    # ⚠️ recency 用的是**上次访问时间**，不是创建时间。
    # 用创建时间会让一条被反复用到的核心事实随着日子过去慢慢沉底，
    # 而它恰恰是最该留在上面的那条。论文里这个字段就叫 last accessed。
    decay = math.log(2) / (half_life_hours * 3600.0)
    rec = [math.exp(-decay * max(0.0, now - it.last_access)) for it in items]
    imp = [float(it.importance) for it in items]
    rel = list(relevance(query, items))
    if len(rel) != len(items):
        raise ValueError(f"relevance 返回了 {len(rel)} 个分数，候选有 {len(items)} 条")

    nr, ni, nv = _minmax(rec), _minmax(imp), _minmax(rel)
    wr, wi, wv = weights

    scored = []
    for i, it in enumerate(items):
        parts = {"recency": nr[i], "importance": ni[i], "relevance": nv[i]}
        total = wr * nr[i] + wi * ni[i] + wv * nv[i]
        scored.append((total, it, parts))

    scored.sort(key=lambda x: (-x[0], x[1].item_id))
    return [(it, parts) for _, it, parts in scored[:k]]


def bm25_relevance() -> Callable[[str, Sequence[Recallable]], list[float]]:
    """把 BM25 包成 `three_factor_rank` 要的 relevance 函数。

    ⚠️ 这是**关键词**相关性，不是语义相关性。要语义就传你自己的 embedding
    版本进去——和 `VectorRetriever` 那里是同一个交代。
    """
    def _rel(query: str, items: Sequence[Recallable]) -> list[float]:
        docs = [Doc(it.item_id, "", it.text) for it in items]
        r = BM25Retriever(docs)
        by_id = {h.doc_id: h.score for h in r.search(query, k=len(items))}
        return [by_id.get(it.item_id, 0.0) for it in items]
    return _rel
