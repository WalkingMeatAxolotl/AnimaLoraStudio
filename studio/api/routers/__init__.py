"""HTTP routers — PR-5 起从 studio/server.py 逐批搬过来。

每个文件 = 一个域：health / presets / browse / events_sse / ...
api/app.py 一次性 include_router 全部。
"""
