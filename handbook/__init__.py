"""《AI Agent 工程手册》的参考实现。

每一章对应一个模块，后一章建立在前一章之上：

    第 2 章  tokens.py / errors.py / transport.py / llm.py   模型调用
    第 3 章  context.py                                       上下文工程
    第 4 章  tools/                                           工具
    第 5 章  loop.py                                          编排与控制流
    第 7 章  memory.py                                        跨会话记忆
    第 9 章  trace.py                                         可观测

跑通第 1 章那封邮件：

    python -m examples.ch01_email
"""

from .context import ContextManager, Entry
from .errors import (
    ConfirmationRequired,
    HandbookError,
    OutputFormatError,
    ToolError,
)
from .llm import FAST, SMART, LLM, Tier
from .loop import Agent, Budget, Outcome
from .memory import MemoryStore, dispatch as memory_dispatch
from .tools import ToolRegistry, ToolSpec
from .trace import Trace
from .transport import (
    AnthropicTransport,
    RecordingTransport,
    ReplayTransport,
    Request,
    Response,
    ScriptedTransport,
    ToolUse,
    Usage,
)

__all__ = [
    "Agent", "Budget", "Outcome",
    "MemoryStore", "memory_dispatch",
    "ContextManager", "Entry",
    "LLM", "Tier", "FAST", "SMART",
    "ToolRegistry", "ToolSpec",
    "Trace",
    "AnthropicTransport", "ReplayTransport", "RecordingTransport",
    "ScriptedTransport", "Request", "Response", "Usage", "ToolUse",
    "HandbookError", "ToolError", "OutputFormatError", "ConfirmationRequired",
]

__version__ = "0.1.0"
