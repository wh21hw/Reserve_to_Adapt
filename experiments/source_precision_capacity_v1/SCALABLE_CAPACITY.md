# 容量推断的分块内存改造

2026-10-04。正式特征仍为 ResNet，不引入 DINO、采样或新损失。

`fit_robust_capacity` 与 `estimate_source_cost_capacity` 新增可选 `proposal_block_size`；默认 None 保持原完整矩阵路径。指定正整数后，按候选列计算距离，每个候选仍使用全部观测行，保留原收益、建簇成本、排序、argmax 与中心更新规则。source 校准建簇成本也采用相同分块方式，避免只解决 target 内存。

这只降低内存，不降低二次复杂度的候选搜索时间。VisDA target 55388 行时，一份 float64 完整距离矩阵约 22.86 GiB；256 列块约 108.18 MiB，另有特征和计算临时数组。尚未验证全量 VisDA 推断耗时可接受，不能据此宣布已跑通 VisDA。

在独立 `/content/robust-streaming-check-v1` 中完成一次针对性检查，不覆盖正在训练使用的 `/content` 模块：

- `check_robust_capacity_streaming.py`：72 行、block7（覆盖末尾不足一块），完整与分块均 K2、3 步；分配/计数相同，中心与目标轨迹在 1e-10 内一致，returncode0。
- `check_source_cost_streaming.py`：source51 行、校准8行、block3；校准成本/阈值/先验相同，参数正确传入核心；核心使用 stub，避免重复测试，returncode0。

不同矩阵乘法形状可能带来浮点舍入差异，极近的候选并列不能保证所有输入 bit-identical。已运行的 OfficeHome frozenBN 配对仍使用原完整矩阵，不中途更换实现或研究设置。
