"""utils 历史目录。

新代码不要在此添加内容——training 相关请去 ``runtime/training/``，算法实现
（lycoris / 未来 T-LoRA 等）的归宿待 [[utils-full-refactor-plan-postponed]]
完整重构决定（见 memory）。

本模块**故意保持空**：早期版本在这里 eager re-export 五个子模块（dataset /
model_utils / checkpoint / comfyui_loader / optimizer_utils），触发 torchvision
链式 import，是测试基础设施的长期痛点。死代码已删除；剩余活模块仍可通过
``from utils.X import ...`` 子模块路径访问。
"""
