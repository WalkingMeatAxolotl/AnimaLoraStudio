"""Studio 业务服务（无 HTTP 入口）。

每个子模块都是「纯函数 + dataclass」风格，由 worker 或服务端调用。
不依赖 db；db 操作在调用侧完成。
"""
