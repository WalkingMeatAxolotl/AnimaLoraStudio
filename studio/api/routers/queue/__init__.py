"""Queue endpoints（PR-6 commit 6 从 server.py 抽出，20 routes 内部拆 3 文件）。

按职责切：
    lifecycle.py  task 状态机 12 routes：list / enqueue / hold / release / reorder /
                  get / cancel / pause / resume / retry / delete
    io.py          数据导入导出 3 routes：export / import / snapshot/config
    outputs.py     训练产物 5 routes：outputs / outputs.zip / export-outputs /
                   output/{filename} / open-folder

3 个 sub-router 各自独立，api/app.py include 3 次。
"""
