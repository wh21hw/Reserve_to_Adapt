# 新未知槽数据初始化：冻结状态压力测试

## Goal

针对K2→5在final70新增三行全无赢家的observation，检验简单数据原型方向能否让新槽起步，同时不破坏source已知拒识边界。这是独立初始化探索，不是纯K-only，不更改现有训练结果或默认算法。

## Success Metric / Guard

- Metric：完整target的全头argmax中，被使用的新增行个数，越高越好，基线0；至少1才算功能激活。
- Guard：source已知样本被判unknown的比例，相比原模型不得增加超过1pp，不能用source误拒识换新槽占用。
- source仅为已监督训练过的数据，guard是机制压力检查，不当独立泛化证据。target语义标签完全不读，也不重算accuracy。
- 即便通过，也只说明一个冻结状态候选值得训练验证，不宣称OSDA性能改进、恢复真类别或修复已完成。

## Constraints

max_iterations: 1；pause_every: never；Evaluator: _(none — agent judges manually)_。已继承用户无人值守/综合判断授权，不再询问。预算0轮训练、一次缓存张量计算，CPU/子进程上限300秒，无图像前向、不拟合K/GMM/参数网格，不写改checkpoint。

固定使用当前final70/K5缓存及对应final70 checkpoint。原分类头输入是normalized bottleneck→eval BN→LeakyReLU0.2，必须先一次针对性检查重建logits与缓存一致，不能把pre-BN簇中心直接当fc方向。

保留known10与旧unknown2权重；仅将三个新增行12/13/14换成对应数据成员在实际head输入空间的均值方向，norm取保留12行平均norm。成员沿用第10轮已保存matched_assignments；用当前第70轮表示重算均值，不重估K。这个跨时段固定成员限制必须披露，不能冒充训练第10轮的初始化干净消融。

不根据target结果调norm，不添加source校准缩放或其他补救到本单次测试。若source guard失败，保留失败证据，候选不接长训练；不偷扫scale找到表面通过的配置。模型、已有报告与用户根目录修改保持不动。

## Hypothesis

冷启动随机方向可能使新增槽没有赢家，当前成员的head-space均值方向会提供正向支持；但类内共同激活或早期簇中的known模式，也可能把已知样本推向未知。测试必须同时检验这两个方向，不只看新槽占用。

## History

| Iteration | Change | Observation | Decision |
| --- | --- | --- | --- |
| 0 | final70原未知头 | 新增三行均无target赢家，source未知预测0 | 缓存已存在，不重评分模型 |
| 1 | 三个新增行用实际head-space成员均值/norm匹配 | target三个新增槽均被使用，89个样本；source误拒识97/958，增加10.1253pp | guard_violation，不接长训练，不改默认方法 |

工程状态：首次CPU运行因Drive中本批last.pt不存在，在原型计算前FileNotFoundError；不是方法测试结果。独立目录读取与T4 DriveFS日志确认用户存储quotaExceeded，大模型上传未完成。已优先暂停本测试，备份本批四模型到电脑；小型fc/BN状态将通过明确--head-state传入，缓存不重提。待备份保护后继续，不能把缺文件当guard失败。

恢复条件更新：refresh-last完整本地副本已保存，fixed2两模型已在CPU独立Drive挂载可见，三个其余本地续传确认仍在推进。CPU单次纯缓存计算不修改T4原件、不重训、不停止备份，因此与剩余备份并行恢复本诊断；不据此宣称四模型均已保护或云端已全保存。仅约20KB分类头状态直接上传CPU，避免读取未上传的refresh模型。
