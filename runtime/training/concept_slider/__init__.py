"""Concept slider POC：image-pair 训练 saturation tweaker LoRA。

参考 Gandikota et al. 2023 "Concept Sliders"。POC 只跑 saturation 单轴；
未来加新轴时直接在 data.py 加 pair op，loop 无需改。

入口：anima_train.py --training_mode concept_slider
"""
