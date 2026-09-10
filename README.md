# agent-handbook

**把一个 LLM agent 从"能跑一次"做到"敢无人值守跑"，中间要补的每一层，这里都有一份能跑的最小实现。**

这是《AI Agent 工程手册》的参考实现。书里出现的每一段代码都从这个仓库剪出来，
**不允许出现 `...` 占位** —— 因为一本教读者做 agent 的书，如果自己的代码跑不起来，
它教的就只是怎么评价别人做的 agent。

---

## 30 秒看它做什么

一封真实难度的客户来信：

> 我上个月买的那个东西坏了，当时你们客服说过可以延保的，现在怎么算？
> 另外我记得还有张优惠券没用，能一起处理吗？

三个难点：「那个东西」指哪一笔订单（上个月有三笔）；「客服说过延保」是否属实（要去对话记录里核实）；
优惠券能不能抵维修费（要查政策，不能凭印象承诺）。

```bash
git clone https://github.com/foxFallingSkies/agent-handbook && cd agent-handbook
uv venv && uv pip install -e ".[dev]"
python -m examples.ch01_email          # 不需要 API key
```

输出（节选）：

```
====================================================================
⏸  需要人工确认
====================================================================
即将执行：create_repair_ticket
风险等级：high｜可逆：否
参数（原样）：
{
  "order": "order-0803",
  "symptom": "咖啡机故障，客户反馈已损坏",
  "warranty_claim": true
}

====================================================================
执行轨迹
====================================================================
  处理客户来信 (1ms)
    step 1
      llm_call  [in=80 cached=0 out=90]
      search_orders  [{"since": "2026-08-01"}]
    step 2
      llm_call  [in=321 cached=1475 out=90]
      search_conversations  [{"keyword": "延保"}]
    step 3
      llm_call  [in=412 cached=1475 out=90]
      check_warranty  [{"order": "order-0803"}]
      rolling_fold  [{'saved_tokens': 194}]
    ...
    step 6 (resumed)
      create_repair_ticket
    verify  [{'verified': True, 'note': '工单 TCK-0001 已创建，订单与保修判定均正确'}]

====================================================================
这一次运行的账
====================================================================
步数              8
工具调用          7
输入 token        12,891
输入/输出比       13.4:1   (Manus 公开的生产值约 100:1)
缓存命中率        68.7%
成本              $0.030284

====================================================================
环境验证（不是问 agent，是查数据库）
====================================================================
结论: ✅ 通过   工单 TCK-0001 已创建，订单与保修判定均正确
```

**最后一段是这个项目的重点**：验收不看 agent 说了什么，看数据库里真的多了什么。

---

## 有什么

| 模块 | 对应章节 | 关键点 |
|---|---|---|
| `tokens.py` | 第 2 章 | token 是什么、怎么数、为什么格式选择会改变成本 |
| `transport.py` | 第 2 章 | 真实 / 回放 / 录制三件套；缓存断点标在 `system` 和最后一个工具上 |
| `llm.py` | 第 2 章 | 结构化输出的自修链（回传**字段级**校验错误）、按异常类型分流的重试、模型分级路由、成本报表 |
| `context.py` | 第 3 章 | **滚动折叠**为主 + 阈值压缩兜底；带「取回钥匙」的内容折叠是无损的；失败记录永不折叠 |
| `tools/` | 第 4 章 | 错误消息是 prompt；语义化 ID；风险分级；**遮蔽而非删除**；参数校验即防呆 |
| `loop.py` | 第 5 章 | 五道闸（终止/预算/漂移/抖动/去重）；人工确认是显式重入 |
| `memory.py` | 第 7 章 | Anthropic `memory_20250818` 工具的服务端；六个命令、路径穿越防护、按**访问**时间淘汰 |
| `retrieval.py` | 第 6 章 | BM25（中文字符 bigram）· RRF 融合（**只用排名不用分数**）· recall@k 是天花板；⚠️ 自带的向量检索是词袋不是语义，写在 docstring 里 |
| `reliability.py` | 第 11 章 | 单例 flock · 租约 · append-once 去重台账 · 死信 · **带进度的心跳**（只报时间的心跳在卡死时照样跳） |
| `trace.py` | 第 9 章 | span 树，**压缩事件也是 span** |
| `evals/` | 第 8 章 | golden set（23 条 · 13 条负例）+ pass@k / pass^k 的组合估计量 + 环境状态断言 |

```bash
pytest                    # 64 个测试，每个对应书里的一条论断
python -m evals.run --k 3 # golden set
```

---

## 三个能省你半天的坑

这些不是书里抄来的，是写这个仓库时**真的踩到**的。

**1. `args` 是 `BaseException` 的保留属性。**
`ConfirmationRequired` 曾经写成 `self.args = args`（一个 dict）。Python 静默地把它转成
`tuple(dict)` —— 也就是**只剩键名，值全丢了**。不报错，只是后续的确认哈希永远对不上，
人工批准无声失效。见 `tests/test_handbook.py::test_confirmation_args_are_not_swallowed_by_exception_args`。

**2. 人工确认中断时，上下文里会留下没有配对 `tool_result` 的 `tool_use`。**
真实 API 会直接拒绝下一次请求。所以 `Agent` 必须把未执行的调用存进 `_pending_calls`，
恢复时**先把它们做完**再继续循环。

**3. 参数校验的错误如果抛到调用栈上层，模型就永远看不到，也永远学不会改。**
所以 `ToolRegistry.call` 把**所有**工具层问题都变成给模型看的文本，只有
`ConfirmationRequired` 例外——它不是错误，是必须冒泡到人那里的控制流信号。

---

## 离线与在线

不带 key 时用 `ScriptedTransport`：**工具层、上下文管理、闸门、trace、成本记账全部真实执行**，
只有「模型说了什么」是预设的。token 数由本地估算（含前缀缓存的模拟），报表里标了「估算值」。

要真实数字：

```bash
export ANTHROPIC_API_KEY=sk-...
python -m examples.ch01_email --live
python -m evals.run --live --k 5      # 真实的 pass^k
```

也可以用 `RecordingTransport` 录一次到 `fixtures/`，之后用 `ReplayTransport` 离线复现。

---

## 诚实的局限

- **fixtures 目前是空的。** 脚本化响应是手写的替身，不是录制的真实返回。
- **离线模式的 pass^k 恒为 1.0**，它验证的是管道不是模型。真实的非确定性只有 `--live` 才看得到。
- **没有流式输出。** 生产系统几乎必然需要它，第 2 章也承认这是缺口。
- **没有并发控制。** `LLM` 是 async 的，但没有令牌桶、没有多 key 轮换、没有 TPM/RPM 处理。
- **`tokens.estimate` 是估算**，典型误差 ±20%，只用于「要不要折叠」这类不需要精确的决策；
  凡是涉及计费的地方一律用 API 返回的真实 usage。
- **虚构数据。** `data/shop.json` 是为教学构造的，不含任何真实业务信息。

---

## License

MIT
