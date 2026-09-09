# fixtures

录制的真实 API 响应放这里，供 `ReplayTransport` 离线回放。

目前是空的——离线演示用的是 `examples/_scripted.py` 里手写的替身。
要录制真实响应：

```python
from handbook import AnthropicTransport, RecordingTransport
transport = RecordingTransport(AnthropicTransport(), fixture_dir="fixtures")
```

文件名是请求的指纹（sha256 前 16 位）。**请求内容变一个字符，指纹就会变**——
这正是 KV-cache 的行为，也是为什么 `Request.fingerprint()` 必须用确定性序列化。
