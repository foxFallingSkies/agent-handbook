"""Token 计数。

手册第 2 章的成本模型、第 3 章的上下文管理，全部建立在"一段文本有多少
token"这个量上。所以它必须先被讲清楚。

**什么是 token**
模型不按字符也不按词处理文本，而是按 token——一段由分词器（tokenizer）
切出来的子词单元。英文里一个常见单词通常是 1 个 token，罕见单词会被切成
几个；中文汉字通常是 1-2 个 token 一个字；而 JSON、代码里的标点、缩进、
转义符各自都要占 token。

这直接产生一个工程后果：**同样的信息，用不同格式表达，token 数可以差
一倍以上**。这就是第 4 章说"选格式的判据是模型好不好写"时，成本那一半的
来源。

**怎么数**
三种办法，可靠性递减：

1. **API 返回的 usage 字段** —— 唯一权威。调用发生之后才知道。
2. **官方的 count_tokens 端点** —— 调用之前就能知道，要多一次网络往返。
3. **本地估算** —— 立刻、免费、不准。

本模块提供 3，并把 1 接进 Usage 记录。**估算只用于"要不要触发压缩"这类
不需要精确的决策**；凡是涉及计费和报表的地方，一律用 API 返回的真实值。
"""

from __future__ import annotations

import re

# 经验系数。这些数字是粗糙的，来自对常见分词器行为的观察，
# 不同模型的分词器会有出入。**不要拿它算钱。**
_CJK = re.compile(r"[一-鿿぀-ヿ가-힯]")
_WORD = re.compile(r"[A-Za-z]+")
_NUM = re.compile(r"\d+")

CJK_TOKENS_PER_CHAR = 1.5      # 中日韩：一个字约 1-2 token
LATIN_CHARS_PER_TOKEN = 4.0    # 拉丁字母：约 4 个字符 1 token
OTHER_CHARS_PER_TOKEN = 3.0    # 标点、符号、空白


def estimate(text: str) -> int:
    """估算一段文本的 token 数。

    ⚠️ 这是估算，典型误差 ±20%。它存在的唯一理由是：在没有网络往返的
    情况下，给上下文管理器一个"现在大概占了多少"的信号。
    """
    if not text:
        return 0

    cjk = len(_CJK.findall(text))
    latin = sum(len(m) for m in _WORD.findall(text))
    digits = sum(len(m) for m in _NUM.findall(text))
    other = max(len(text) - cjk - latin - digits, 0)

    return int(
        cjk * CJK_TOKENS_PER_CHAR
        + latin / LATIN_CHARS_PER_TOKEN
        + digits / 2.0
        + other / OTHER_CHARS_PER_TOKEN
    ) + 1


def estimate_messages(messages: list[dict]) -> int:
    """估算一组消息的 token 数，含每条消息的固定开销。"""
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += estimate(content)
        else:
            # content 是 block 列表（tool_use / tool_result / text）
            for block in content:
                if isinstance(block, dict):
                    total += estimate(str(block.get("text", "")))
                    total += estimate(str(block.get("content", "")))
                    total += estimate(str(block.get("input", "")))
        total += 4      # 每条消息的角色标记等固定开销
    return total


def format_comparison(data: dict) -> dict[str, int]:
    """演示"同样的信息，不同格式的 token 差异"。

    手册第 2 章 §2.3.3 的论点是"格式对模型的难度不对称"；
    这个函数展示它的成本那一面。examples/ch02_format_cost.py 会用到。
    """
    import json

    as_json = json.dumps(data, ensure_ascii=False, sort_keys=True)
    as_json_indented = json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2)
    as_lines = "\n".join(f"{k}: {v}" for k, v in sorted(data.items()))

    return {
        "json_compact": estimate(as_json),
        "json_indented": estimate(as_json_indented),
        "plain_lines": estimate(as_lines),
    }
