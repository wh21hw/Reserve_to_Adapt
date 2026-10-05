# 未知伪标签自举的因果对照（待确认运行）

当前状态：脚本已准备；真实control代码的build-only编译通过，prepare-only读取完整control并输出同设置命令。没有模型前向、GPU训练或新结果。用户尚未确认新增约3–4小时L4预算，不能把goal自动续行当作对该询问的回答。

## 问题与唯一改动

VisDA固定K2和估计K8均后期把几乎所有target判未知，且已知特征身份退化。检验原版未知伪标签CE是否是重要推动因素，不以关闭未知学习作为最终解决方案。

复用 `/content/imp-runs/visda-frozenbn-capacity-10e-v1/fixed2` 完整control。新arm C6/K2/Q8、seed1、source C维监督3轮、相同source-final初始化、ResNet50、encoder BN统计冻结、batch64、相同优化器与学习率、RTA10轮。

选定发布代码在warmup后原损失为 `ce + .01*virtual_ce + .3*adv_loss + entropy + ce_ep`。仅最后一项系数由1改0；预热、候选选择、未知分类头前向、BN统计更新、原版warm-end K-means头初始化、虚拟方向和预测规则均保留。不能跳过候选前向，因为那还会改变head BN及执行轨迹。发布代码没有可用的lambda2 CLI参数，所以由独立entry做一处精确损失替换，不改用户根目录训练文件。

脚本：`scripts/train_visda_unknown_ce_off_entry.py` 和 `scripts/run_visda_unknown_ce_off_colab.py`。默认prepare-only不创建run目录、不训练；确认后才能显式`--run`。输出目录独立 `/content/imp-runs/visda-unknown-ce-off-10e-v1`，不覆盖既有结果。

## 评价与解释

全部十轮OS*/UNK/HOS及final对同轮control；best仅为target-oracle描述，不能以best选新阈值或提前停训。记录loss有限、进程/OOM、普通配置日志。单seed短程机制实验，不声称统计显著或原论文VisDA复现。

若已知塌缩明显缓解而UNK下降，表明未知CE可能推动失衡，但也承担拒识学习；应进一步改造可靠结构监督，而非以删除该模块为最终方法。若仍塌缩，则继续区分entropy、选择性对抗/表示等因素，不能根据这一组简单宣称自举不是任何原因。一次系数消融不解释所有模块交互，不同时扩大网格、换seed或加新损失。

失败保留直接错误证据，不自动重跑；正常完成保存结果后，再更新unknown建模方案。当前没有任何新GPU训练等待句柄。
