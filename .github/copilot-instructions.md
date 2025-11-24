# Copilot / AI agent 指南 — Reserve_to_Adapt

下面的说明直接针对本仓库的结构、运行约定与常见坑，旨在帮助自动化编程代理（或新到的开发者）快速、安全、可重复地在此代码库中工作。

- 主要入口：`Reserve_to_Adapt/main.py`。这是训练脚本，包含数据加载、模型构建、训练循环与评估。
- 关键模块：
  - 特征与网络：`Reserve_to_Adapt/networks.py`（`ResNetFc`, `CLS`, 对抗网络等）
  - 数据与加载：`Reserve_to_Adapt/data.py`（`CustomDataset`, 列表文件格式）
  - 训练工具与辅助：`Reserve_to_Adapt/utilities.py`（`Accumulator`, `TrainingModeManager`, 自定义损失、调度器、优化器封装等）
  - 多域迭代器：`Reserve_to_Adapt/domain_bus.py`（`DomainBus`用于同步多个DataLoader）
  - 类心（centroids）管理：`Reserve_to_Adapt/centroid.py`（`Centroids`类用于源/目标类中心的维护）

关键设计与约定（可直接用于代码编辑建议）：

- 数据文件格式：数据列表文本（如 `data/*.txt`）每行格式为 `relative_path label`。用 `data.get_split_dataset_info(txt, data_dir)` 加载。
- Windows 注意事项：`main.py` 的默认参数使用了硬编码 Windows 绝对路径（工程作者的本地路径）。不要在改动时移除 `num_workers=0` 的注释或更改为 >0，除非你能在 Windows 上正确运行消费者进程；默认代码里多处对 `torch.cuda.is_available()` 进行了保护。
- 输出与日志：训练过程中 `main.py` 将 stdout 重定向到 `args.log_dir + '/out.txt'`。任何修改训练输出的变动（例如移除重定向或更改路径）会影响 CI/日志采集。
- 依赖与环境：推荐 Python 3.8，PyTorch 1.7.1 和 torchvision 0.8.2（见 `requirements.txt` 中精确 pin）。项目还依赖 `faiss`（用于聚类），`scikit-learn`、`scipy` 等科学库。

代码模式与常用 API（对生成代码/重构很重要）：

- 上下文累积器：`Accumulator(['name1','name2'])` 被频繁用于在训练/前向循环中收集 numpy 数组，调用样例：`ProbRecorder.updateData(globals())`，退出上下文后读取 `ProbRecorder['fs']` 等。
- 训练/评估切换：使用 `TrainingModeManager([nets], train=False)` 在一个块内临时切换模型的 `.train()` 模式。
- 优化器封装：`OptimWithSheduler` 和 `OptimizerManager` 封装了 learning rate 调度与多优化器零梯度/step 的管理。请在修改优化流程时保留这些封装或确保行为等价。
- 自定义损失：项目实现了自己的 `CrossEntropyLoss`, `BCELossForMultiClassification`, `EntropyLoss` 等（位于 `utilities.py`），代码中混用这些自实现与 PyTorch 原生函数，修改时请确认数值/shape 兼容性。
- 分类器扩展：`CLS.virt_forward` 实现了“虚拟类/扩展分类器”逻辑（把额外的 nomatch 特征并入 logits），修改分类分支逻辑时需保留该接口签名。

运行/开发快捷说明（可复制到终端）：

1) 创建环境并安装依赖（建议在虚拟环境中安装 `requirements.txt`）：

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r Reserve_to_Adapt/requirements.txt
```

2) 训练示例：`main.py` 使用 argparse 接收 `--source` `--target` `--gpu` 等。示例（替换成你的路径）：

```powershell
python Reserve_to_Adapt/main.py --source <source_txt> --target <target_txt> --gpu 0 --batch_size 64 --name run1
```

注意：默认参数里指向了作者本地的绝对路径，运行前请把 `--data_dir` 与 `--log_dir` 指向本机可写目录，或直接传入完整 `--source`/`--target` 路径。

修改/扩展时的具体建议：

- 新增数据集：在 `data/` 下添加文本列表，格式同上；不要修改 `CustomDataset` 的返回结构（返回 `(image_tensor, one_hot_label)` 在 `main.py` 中被期望）。
- 新增模型或替换 backbone：优先在 `networks.py` 中扩展 `BaseFeatureExtractor` 子类，保持 `output_num()` 方法和 `forward` 行为一致；`main.py` 通过 `ResNetFc(...).output_num()` 假定了这一约定。
- 小心全局状态：`main.py` 中使用大量 `globals()` 传入 `Accumulator.updateData` 和 `globals()` 读取变量；自动修改器请避免在局部作用域内改变这些变量名或删除被 `globals()` 引用的中间变量。

常见坑与提醒：

- 绝对路径：默认 args 使用 Windows 本地路径，CI/云端运行前务必覆盖 `--data_dir` / `--source` / `--target`。
- GPU/CPU 兼容：代码经常用 `torch.cuda.is_available()` 条件分支；当你在 CPU 上调试时，确认 `.cuda()` 的调用被正确条件化，或在修改后在 CPU 上运行一次 smoke test。
- 不要随意移除 stdout 重定向：日志被写入 `args.log_dir/out.txt`，调试时可以临时注释，但 PR 中应恢复。

如果某处信息不明确，请指出需要补充的细节（例如你要运行的 OS、目标硬件、或希望自动化的具体改动），我会据此调整/补充本文件。
