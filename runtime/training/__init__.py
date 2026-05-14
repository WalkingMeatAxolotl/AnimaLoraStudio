"""runtime/training 子包：anima_train 训练代码的模块化拆分（ADR 0003）。

PR-A：把原 runtime/anima_train.py 的 53 个 def/class 按职责分到本子包。
PR-B：引入 TrainingContext + phase 拆分 main()。
PR-C：引入 4 个 plugin 子包（adapters / optimizers / schedulers / inference_samplers）。

详细设计见 docs/adr/0003-anima-train-refactor.md。
"""
