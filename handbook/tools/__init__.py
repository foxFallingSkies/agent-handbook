"""工具层。手册第 4 章。

这里是"确定性系统"与"非确定性 agent"之间的契约面。设计上体现四件事：

1. 工具按**高价值工作流**切分，不按 CRUD 切（见 shop.py）
2. **错误消息是写给模型看的 prompt**，不是日志
3. **风险分级**与幂等/可逆标记是工具定义的一部分，不是事后补的
4. 可用性用**遮蔽**实现（allowed 集合），工具定义始终留在上下文里保住 KV-cache
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from ..errors import ConfirmationRequired, ToolError, ToolNotAllowed

Risk = Literal["low", "medium", "high"]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict                     # JSON Schema
    fn: Callable[..., Any]
    risk: Risk = "low"
    idempotent: bool = True              # 重试安全吗
    reversible: bool = True              # 做错了能撤销吗
    requires_confirmation: bool = False   # 强制人工确认（第 10 章）
    # 并行安全：这个工具有没有副作用。第 5 章 §5.9。
    # ⚠️ **默认 False（不安全），必须显式声明。** 这是 fail-closed：
    # 并行执行破坏顺序依赖时不会报错，只会给出一个看起来正常的错结果。
    # 一个「忘了标」的工具应该退化成串行（慢），而不是并行（错）。
    # ⚠️ 它不能从 risk 推出来：一个 risk="low" 的工具照样可以写日志、
    # 计数、发埋点。风险等级说的是「做错了多严重」，read_only 说的是
    # 「做了会不会改变世界」——两个正交的问题。
    read_only: bool = False
    # 确认闸**之前**跑的廉价校验。必须无副作用。
    # 存在的理由：不要请人批准一个注定会失败的动作。
    # 让人点了「同意」再看到「无权操作」，会训练出无脑点同意的习惯，
    # 而那正是确认闸唯一想防的东西。
    precheck: Callable[[dict], None] | None = None
    scope: str = "customer"              # 数据可见范围，见 §"鉴权"一节

    def to_api(self) -> dict:
        """转成 Anthropic tools 参数要的形状。

        这一步在书的初稿里被省略了——而它恰恰是新手第一次接工具调用
        100% 会卡住的地方。
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


class SemanticIdMap:
    """对模型暴露语义化短 ID，内部换回真实主键。第 4 章 §4.3。

    UUID 对模型是纯噪声：记不住，需要引用时容易编一个格式看起来对的出来。
    """

    def __init__(self, prefix: str):
        self.prefix = prefix
        self._to_real: dict[str, str] = {}
        self._to_short: dict[str, str] = {}

    def shorten(self, real_id: str, label: str | None = None) -> str:
        if real_id in self._to_short:
            return self._to_short[real_id]
        base = label or str(len(self._to_real) + 1)
        short = f"{self.prefix}-{base}"
        n = 2
        while short in self._to_real:          # 不同 real_id 撞同一个 label 时不静默覆盖
            short = f"{self.prefix}-{base}-{n}"
            n += 1
        self._to_real[short] = real_id
        self._to_short[real_id] = short
        return short

    def resolve(self, short: str) -> str:
        if short not in self._to_real:
            raise ToolError(
                f"未知的标识符 {short!r}。",
                hint=f"当前可用的标识符有：{', '.join(sorted(self._to_real)) or '（无）'}。",
            )
        return self._to_real[short]


def truncate_items(
    items: list, page_size: int, filter_hint: str
) -> tuple[list, str | None]:
    """按条数分页，并返回一段**有用的**截断提示。

    第 4 章 §4.4：不要只说 "results truncated"，要告诉它怎么缩小范围。
    """
    if len(items) <= page_size:
        return items, None
    note = (
        f"[结果被截断：共匹配 {len(items)} 条，已返回前 {page_size} 条。"
        f"建议增加过滤条件缩小范围，例如 {filter_hint}。]"
    )
    return items[:page_size], note


@dataclass
class ToolResult:
    """一次工具调用的结果。

    `text` 一定会回到模型的上下文里——**成功和失败都会**。
    `is_error` 只影响两件事：上下文里标 is_error、以及这条记录永不被折叠
    （失败是「此路不通」的证据，见第 3 章 §3.10）。
    """

    text: str
    is_error: bool = False

    def __str__(self) -> str:            # 方便直接打印与断言
        return self.text

    def __contains__(self, sub: str) -> bool:
        return sub in self.text


@dataclass
class CallStat:
    calls: int = 0
    errors: int = 0
    arg_digests: list[str] = field(default_factory=list)


class ToolRegistry:
    def __init__(self, max_response_tokens: int = 25_000):
        self.tools: dict[str, ToolSpec] = {}
        self.stats: dict[str, CallStat] = {}
        self.max_response_tokens = max_response_tokens

    def register(self, spec: ToolSpec) -> None:
        self.tools[spec.name] = spec

    def is_retry_safe(self, name: str) -> bool:
        """这个工具失败后能不能直接重试？

        `idempotent=False` 的工具重试一次就多一个副作用——
        开工单的那个工具重试三次，客户就收到三次上门取件电话。
        编排层要读这个字段，而不是对所有失败一视同仁地重试。
        """
        spec = self.tools.get(name)
        return bool(spec and spec.idempotent)

    def api_tools(self) -> list[dict]:
        """给模型的工具定义。**顺序稳定**——否则会破坏 KV-cache 前缀。"""
        return [self.tools[n].to_api() for n in sorted(self.tools)]

    # ------------------------------------------------------------------

    def call(
        self,
        name: str,
        args: dict,
        *,
        allowed: set[str] | None = None,
        confirmed: bool = False,
    ) -> "ToolResult":
        """执行一个工具。

        **任何工具层的问题都不向上抛，而是变成给模型看的文本。**
        理由是第 4 章的核心主张：错误消息是 prompt。一个抛到调用栈上层的
        异常，模型永远看不到，也就永远学不会怎么改。

        唯一的例外是 ConfirmationRequired —— 它不是错误，是控制流信号，
        必须一直冒泡到能做决定的那一层（人）。
        """
        spec = self.tools.get(name)
        if spec is None:
            return ToolResult(
                ToolError(
                    f"不存在名为 {name!r} 的工具。",
                    hint=f"可用的工具有：{', '.join(sorted(self.tools))}。",
                ).to_model(),
                is_error=True,
            )

        st = self.stats.setdefault(name, CallStat())

        try:
            # 遮蔽而非删除：定义还在上下文里，只是此刻不让调
            if allowed is not None and name not in allowed:
                raise ToolNotAllowed(
                    f"工具 {name} 在当前阶段不可用。",
                    hint=f"当前可用：{', '.join(sorted(allowed))}。",
                )

            self._validate(spec, args)

            # 语义校验先于确认闸：schema 对不代表这个动作做得成。
            if spec.precheck is not None:
                spec.precheck(args)

            if spec.requires_confirmation and not confirmed:
                raise ConfirmationRequired(
                    tool_name=name, tool_args=args,
                    preview=self._preview(spec, args),
                )

            st.calls += 1
            st.arg_digests.append(_digest(args))
            result = spec.fn(**args)

        except ConfirmationRequired:
            raise                        # 控制流，不是错误
        except ToolError as e:
            st.calls += 1
            st.errors += 1
            return ToolResult(e.to_model(), is_error=True)
        except Exception as e:           # 未预期的异常也要变成模型能读的文本
            st.errors += 1
            return ToolResult(
                ToolError(
                    f"工具 {name} 执行时发生了未预期的错误：{type(e).__name__}: {e}",
                    hint="这通常是工具实现的 bug，不是你的调用方式有问题。"
                         "请换一个方式完成任务，或如实报告失败。",
                ).to_model(),
                is_error=True,
            )

        text = result if isinstance(result, str) else json.dumps(
            result, ensure_ascii=False, sort_keys=True
        )

        # ⚠️ 最后一道兜底。工具**应该**自己分页（见 truncate_items），
        # 但总有一天会有人接一个返回整张表的新工具。
        # 一次超长的工具返回能一口气吃掉半个上下文窗口，
        # 而这种事发生在生产里、发生在半夜、发生在你没加分页的那个工具上。
        from .. import tokens as _tk
        if _tk.estimate(text) > self.max_response_tokens:
            # ⚠️ 不能用「token × 4」换算回字符——那只对拉丁文成立。
            # 同目录的 tokens.py 自己写着中文约 1.5 token/字符，
            # 于是 4 字符/token 的截断在中文内容上会超出上限约 6 倍。
            # 直接二分收敛到真实估算值以内。
            lo, hi = 0, len(text)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if _tk.estimate(text[:mid]) <= self.max_response_tokens:
                    lo = mid
                else:
                    hi = mid - 1
            keep = lo
            text = (
                text[:keep]
                + f"\n\n[结果过长已被截断：完整内容约 {_tk.estimate(text)} tokens，"
                  f"超过单次工具返回上限 {self.max_response_tokens}。"
                  f"请缩小查询范围后重试——例如加上时间区间或关键词过滤。]"
            )
        return ToolResult(text, is_error=False)

    # ------------------------------------------------------------------

    def _validate(self, spec: ToolSpec, args: dict) -> None:
        """参数校验层。第 4 章 §4.7 的"防呆"就落在这里。

        故意在**契约层**拒绝，而不是在 prompt 里请求模型别传错。
        """
        schema = spec.parameters
        props: dict = schema.get("properties", {})
        required: list = schema.get("required", [])

        missing = [k for k in required if k not in args]
        if missing:
            raise ToolError(
                f"调用 {spec.name} 缺少必填参数：{', '.join(missing)}。",
                hint=f"该工具的参数是：{_describe(props)}。",
            )

        unknown = [k for k in args if k not in props]
        if unknown:
            raise ToolError(
                f"调用 {spec.name} 时传了不认识的参数：{', '.join(unknown)}。",
                hint=f"该工具只接受：{', '.join(sorted(props))}。",
            )

        for k, v in args.items():
            expected = props[k].get("type")
            if expected and not _type_ok(v, expected):
                raise ToolError(
                    f"参数 {k} 的类型应为 {expected}，收到的是 "
                    f"{type(v).__name__}（值：{v!r}）。",
                    hint=props[k].get("description", ""),
                )
            enum = props[k].get("enum")
            if enum and v not in enum:
                raise ToolError(
                    f"参数 {k} 的值 {v!r} 不在允许的取值范围内。",
                    hint=f"允许的取值：{', '.join(map(str, enum))}。",
                )

    @staticmethod
    def _preview(spec: ToolSpec, args: dict) -> str:
        """给人看的确认预览。

        ⚠️ 这里必须展示**原始动作**，不能展示模型写的自然语言摘要——
        否则这道闸就是纸糊的：模型写"帮客户处理一下退款"，人点了同意，
        实际执行的是别的。对应 OWASP ASI15「Human Manipulation」。
        """
        return (
            f"即将执行：{spec.name}\n"
            f"风险等级：{spec.risk}｜可逆：{'是' if spec.reversible else '否'}\n"
            f"参数（原样）：\n{json.dumps(args, ensure_ascii=False, indent=2, sort_keys=True)}"
        )

    # ------------------------------------------------------------------

    def diagnose(self) -> list[str]:
        """把第 4 章 §4.9 的两条诊断规则做成代码。

        重复调用 → 参数设计有问题；反复出错 → 工具描述有问题。
        """
        out: list[str] = []
        for name, st in sorted(self.stats.items()):
            if st.calls >= 3 and len(set(st.arg_digests)) == 1:
                out.append(
                    f"{name}: 重复调用——用完全相同的参数调用了 {st.calls} 次 → 检查参数设计或返回值是否有用"
                )
            if st.calls and st.errors / st.calls > 0.2:
                out.append(
                    f"{name}: 错误率 {st.errors / st.calls:.0%}"
                    f"（{st.errors}/{st.calls}）→ 检查工具描述是否写清楚了"
                )
        return out


def _digest(args: dict) -> str:
    """确定性哈希。

    不用内置 hash()：它对字符串加了随机种子，跨进程不稳定。
    在一本刚讲完"序列化必须确定性"的书里用 hash() 会很尴尬。
    """
    return hashlib.sha256(
        json.dumps(args, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]


def _type_ok(value: Any, expected: str) -> bool:
    return {
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "array": isinstance(value, list),
        "object": isinstance(value, dict),
    }.get(expected, True)


def _describe(props: dict) -> str:
    return "；".join(
        f"{k}（{v.get('type', 'any')}）：{v.get('description', '')}"
        for k, v in sorted(props.items())
    )
