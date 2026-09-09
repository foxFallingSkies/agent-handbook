"""异常层次。

手册第 2 章 §2.6 把模型调用的失败分成几类，每类的处理方式不同。
这里的类型划分直接对应那张表——**重试策略由异常类型决定，不由调用点的 if 决定**。
"""

from __future__ import annotations


class HandbookError(Exception):
    """本库所有异常的基类。"""


# ---------- 传输层：决定要不要重试 ----------

class TransportError(HandbookError):
    """与模型服务通信时的失败。"""


class RateLimitError(TransportError):
    """429。应当退避后重试。

    retry_after 是服务端建议的等待秒数（如果它给了的话）。
    """

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class ServerError(TransportError):
    """5xx。可以有限次重试。"""


class BadRequestError(TransportError):
    """4xx（非 429）。**不要重试**——重试只会重复失败。

    典型原因：参数非法、上下文超长、模型名写错。
    上下文超长要走压缩路径而不是重试，见 handbook/context.py。
    """


class ContextOverflowError(BadRequestError):
    """上下文超长。单独成类，因为它的处置方式和其它 4xx 不同：

    不是"修参数重试"，而是"压缩后重试"。
    """


class RetryExhausted(TransportError):
    """重试次数用尽，且没有可用的降级路径。"""


# ---------- 输出层：结构化输出的失败 ----------

class OutputFormatError(HandbookError):
    """自修链耗尽后，模型输出仍不符合 schema。

    注意这是一个**显式失败**。第 2 章 §2.3.2 的原则：
    不要在重试耗尽后静默返回空对象或默认值——那会让模型层的故障
    伪装成业务层的"用户没填"，在几层调用之后才爆炸。
    """

    def __init__(self, message: str, last_raw: str = "", attempts: int = 0):
        super().__init__(message)
        self.last_raw = last_raw       # 最后一次的原始输出，供排查
        self.attempts = attempts


class OutputTruncatedError(HandbookError):
    """输出被 max_tokens 截断。

    单独成类的理由（第 2 章 §2.6）：截断的 JSON 看起来像"格式错误"，
    如果丢进自修链，模型会试图修复一段它本来写对了、只是没写完的内容。
    正确处置是提高 max_tokens 或让它分段输出。
    """


# ---------- 工具层 ----------

class ToolError(HandbookError):
    """工具执行失败。

    ⚠️ 这个异常的 message 会**回到模型的上下文里**，所以它是一段 prompt，
    不是一条日志。写它的时候用写 prompt 的标准。见第 4 章 §4.5。
    """

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint                # 下一步该怎么做

    def to_model(self) -> str:
        """渲染成给模型看的文本。"""
        return self.message + (f"\n{self.hint}" if self.hint else "")


class ToolNotAllowed(ToolError):
    """工具存在，但在当前状态下不可用。

    对应第 4 章 §4.8 的"遮蔽而非删除"：工具定义始终留在上下文里
    （保住 KV-cache 前缀），可用性在调用点检查。
    """


class ConfirmationRequired(HandbookError):
    """高风险动作需要人工确认。

    不是错误，是控制流信号——循环会捕获它并把决定权交还给人。
    见第 4 章 §4.10。
    """

    def __init__(self, tool_name: str, tool_args: dict, preview: str):
        super().__init__(f"{tool_name} 需要人工确认")
        self.tool_name = tool_name
        # ⚠️ 字段名不能叫 args。
        # BaseException.args 是保留属性（异常参数元组），给它赋一个 dict
        # 会被静默转成 tuple(dict) —— 也就是**只剩键名**，值全丢了。
        # 这个 bug 不会报错，只会让后续的哈希/比对无声地对不上。
        # 手册第 12 章讲「静默失败比崩溃更危险」时会用这个真实案例。
        self.tool_args = tool_args
        # preview 必须是原始动作的展示，不是模型写的摘要。
        # 理由见手册第 11 章（OWASP ASI09 人机信任利用）。
        self.preview = preview


# ---------- 循环层 ----------

class LoopError(HandbookError):
    """agent 循环的终止条件。"""


class StepBudgetExceeded(LoopError):
    def __init__(self, steps: int):
        super().__init__(f"超出步数预算：已执行 {steps} 步")
        self.steps = steps


class TokenBudgetExceeded(LoopError):
    def __init__(self, tokens: int, budget: int):
        super().__init__(f"超出 token 预算：已用 {tokens}，上限 {budget}")
        self.tokens = tokens
        self.budget = budget


class NoProgress(LoopError):
    """检测到无进展：相同状态重复出现，或同一工具用相同参数连续调用。"""

    def __init__(self, reason: str, evidence: str = ""):
        super().__init__(f"检测到无进展：{reason}")
        self.reason = reason
        self.evidence = evidence
