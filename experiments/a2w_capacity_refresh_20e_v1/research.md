# Research: A→W 阶段性重估未知容量

## Goal

总体研究目标仍是从域适应本质出发完善未知结构建模、自适应RTA未知容量；本批不是将整个毕业设计缩小为一次A→W实验。当前可检验问题：保留原RTA学习机制，只在一个预声明阶段重估K，能否相对始终固定K改善已知/未知取舍？

用户2026-10-06选择无人值守运行，由助手综合best/final OS*、UNK、HOS解释结果，不仅挑最高best。用户要求注明预算，本批已明确两arm各20轮，总40轮RTA；不重训source，不扩网格，不自动切换seed/backbone/loss。

## Success Metric

- 主描述指标：两组第20轮final HOS百分比及配对差值，方向maximize；同时完整报告OS*/UNK及各自target-oracle best epoch。
- 初步支持标准：adaptive final HOS比固定K高至少1pp，且final OS*/UNK各自不低于固定K超过1pp；单seed噪声与共同前10轮偏差必须同时解释。
- 达到上述标准只说明这一批值得后续验证，不代表语义K正确、统计显著、论文复现或总体研究目标完成。
- 没有目标标签驱动的阈值/seed/epoch搜索或自动keep/revert优化。所有目标性能是预声明训练结束后的描述性评价，不反馈拟合聚类规则。

## Constraints

- max_iterations: 1（一个两arm配对实验；不能耗尽本批后自行扩大矩阵）。
- pause_every: never；只有必需Google授权、工程失败或异常容量导致训练不定义时请求操作，不反复问是否继续。
- Evaluator: _(none — agent judges manually)_；普通collector只机械汇总完整日志，不提供自动搜索奖励。
- keep_policy: 手工综合判断；负面方法代码/结果保留为探索，不晋升默认方法，不回滚用户修改。
- noise_runs: 1；已事后选择seed3，此批不新选seed，不重复训练作为例行审计。
- min_delta: 1pp（探索筛查，不是统计置信阈值）。
- 训练预算：fixed8与refresh各20完整epoch，总计40新RTA epoch，source3共享既有checkpoint。20轮跨过原代码epoch<=10的候选筛选切换；不冒充70轮结论。
- 单arm运行上限：120分钟；本批总训练上限4小时。CLI操作每次短时返回并按实际进程监督，不因一次观察超时重启训练。到运行上限保留已有结果并停止本批，不悄悄追加预算。
- 设备优先T4，batch64、ResNet50、torch1.7.1/cu110旧环境，encoder BN running stats冻结、affine可训练，head BN不变。OOM先保留证据，不偷偷减batch/改架构。
- 当前CPU实例m-s-kkb-use1b1-3hr6nb4xjabhs及Drive挂载保留，不因小诊断结束销毁；需要GPU时说明CLI无法原地无缝转换，不擅自断开它。
- 只监督进程、loss有限、OOM/错误、容量输出/变化、普通逐轮指标和持久保存；不做hash或重复checkpoint评估。

## Current Approach / Baseline

共同source prior来自已完成的C10/source3监督训练。此前T4固定K8原argmax的10轮final OS*=96.3023%、UNK=78.2380%、HOS=86.3354%；另一个固定簇身份arm也是10轮，但改变伪标签，不是本批控制。该10轮原argmax结果是当前状态的已测量参考，不是本批20轮控制，不能用它替代本批fixed8。

本批必须实际运行fixed8完整20轮作为相同预算baseline。两个arm从相同source prior及初始K8开始，前10轮没有研究因素差异；硬件算法/随机初态共同约定，仍记录共同前段数值偏差，不伪称轨迹严格相同。

已有时序诊断：source3身份匹配K8→final10 K4，15.4255% target候选身份变化；已知误入未知26.1%→7.8%，未知被已知吸收7.1%→12.6%。同一final10空间用旧source数值标定K9、当前标定K4。此证据只支持检验更新时机/容量，不预先证明K4更好。

## Search Space

- 唯一训练因素：第10轮完成后，是否把当前source/target共同表示估出的整数K更新到分类头。
- 两组都在相同节点提取当前全量958/564特征并运行同一个无标签估计器，使额外提取/分析流程对称；fixed8仅记录推断，不调整头。
- 估计器冻结为source-calibrated-birth-cost-v1：source seed2026分层split、当前source anchors/先验、source99%半径、source最大初始建簇收益成本、R/N质量、移动已知中心、birth-first、proposal块64；当前C维head条件likelihood一对一匹配已知身份，未匹配非空簇数作K。不得按真实target语义数或HOS改规则。
- 数据仅来自已持久保存Drive Office31压缩包，本地解压；不能公网重复下载。
- 更新使用已有state-preserving resize：已知行不改，未知按无标签预测/新分配对应保留头与SGD状态，新增行随机norm匹配，不用IMP方向初始化。K不变时不重排/重置。K0/已知匹配不成立时明确报告不定义，不强制K1。
- 禁止：改Q20/V更新规则、加簇身份训练标签/新loss、改原未知argmax/筛选、改最终argmax预测、重置scheduler/warmup、挑新seed、动用户根目录修改。
- 不复用不完整的final10训练状态静默续跑：本批20轮配对从共同source checkpoint开始，正常保留关系库/GMM/优化进度，避免漏恢复状态混杂。

## Context & References

- AGENTS.md；experiments/a2w_unknown_ce_v1/IMP_IDENTITY_RECONCILIATION.md；TEMPORAL_CAPACITY_NEXT.md。
- source_precision_capacity.py、robust_capacity.py、prototype_identity_reconciliation.py、alternating_konly.resize_unknown_head。
- 按autoresearch的单因素假设/实验/评价/日志流程组织，按照colab-cli技能执行；本文明确20轮ML预算代替技能通用5分钟例子，不以5分钟观察截止判模型失败。

## History

| # | Change | Metric | Result | Timestamp |
| --- | --- | --- | --- | --- |
| 0 | 本批fixed8，20轮实际控制 | final HOS76.9941%，OS*99.6667%，UNK62.7251% | baseline；exit0并实际保存Drive | 2026-10-06 |
| 1 | 第10轮后K8→估计K4，20轮完整候选 | final HOS85.1042%；ΔHOS+8.1101pp、ΔUNK+11.7097pp、ΔOS*−0.3226pp | 初步支持，保留探索实现，不晋升默认方法 | 2026-10-06 |

状态：协议已声明，新的20轮配对尚未启动。没有新训练成绩，也没有确认收益；必须在代码关键接口检查及输入恢复完成后启动。

## 实现与恢复进度

2026-10-06：capacity_refresh_boundary.py与隔离训练wrapper已实现。真实final10无标签缓存/已保存CLS和SGD的单次CPU接口检查通过：K8→K4，已知权重/动量保持，4个未知对应行保持，classifier alias与optimizer参数替换正确。未重新图像前向、重新拟合缓存K或训练。两arm实际patch build-only编译通过。当前检查只证明接口，不预设训练时K4。

新T4端点gpu-t4-s-kkb-ass1c1-1lpayjb3tm80f，CPU仍在。最初本地daemon TLS连接ECONNRESET，mount exec1/3在执行前crashed；无训练/数据提取，因此不是OOM或模型失败。确认只有该新T4本地daemon1664且无运行任务后仅终止本地连接进程、重新连接，没有重启/销毁Colab实例。新的mount exec4已生成授权URL并等待用户同意，shell12可用；依赖安装进行中。基础代码/19.9KiB支持包/恢复脚本已成功上传，数据仍将从Drive缓存复制，不公网下载。

run_a2w_capacity_refresh_colab.py显式--run才会训练；每arm最多7200秒/总14400秒，失败保存状态后停止，不自动重试或切换配置；每arm完整20轮后普通模型/日志/容量缓存实际复制到Drive，再进入下一arm。collector只读取20轮history、best/final、post-refresh best与K轨迹，保留共同前10轮偏差，不重评checkpoint。当前尚未执行--run，无新成绩。

## 训练已实际启动

2026-10-06 09:32 UTC现场检查：mount exec4实际done，依赖安装与Drive缓存恢复均成功。一次直接exec --file调用启动器被notebook的-f参数挡住（exec5 SystemExit2），未创建训练arm或模型；改用固定venv在shell12显式执行`/content/rta-py38/bin/python -u /content/run_a2w_capacity_refresh_colab.py --run > /content/a2w-capacity-refresh-pair-console.log 2>&1`，不修改研究设置、不重跑已发生的训练。

实际fixed8 worker PID3607确认存活并已输出Epoch0/1（history为1/2），ce0.489/0.305等loss有限，无OOM；GPU确认为Tesla T4。source shared C10/K8/Q20及encoder BN策略均与声明一致。前两轮在原版warmup，UNK0.036/0.000不能据此判定最终方法失败。第10轮边界尚未执行，refresh arm尚未开始，没有新推断K或最终对照结果；不能将先前final10诊断K4当本批固定设定。

监控：shell12跑串行启动器，查看/content/a2w-capacity-refresh-pair-console.log、各arm/console.log、office31-a2w_seed3/history.jsonl与capacity-after-010/estimate.json即可。正常训练不新排队kernel训练，不重复evaluate checkpoint；远程只读检查可用独立shell或短exec，因为训练在独立shell/venv子进程而非Jupyter串行kernel中。普通20轮模型/日志先逐arm实际保存到Drive，结束运行collector再下载summary/ZIP；CPU实例与挂载保留。目标整体仍未完成。

## 固定组完成，自适应组运行中

09:44 UTC现场检查：fixed8恰好20轮，process-status exit_code0/timed_out=false，启动器已实际复制完整arm到Drive，随后启动refresh worker7929。fixed8在第10轮同样推断K4，但没有应用，最终仍K8；其第20轮OS*=99.6667%、UNK=62.7251%、HOS=76.9941%，历史最高HOS85.8254%。以本批20轮实际控制替换History中的旧10轮参考；旧10轮记录仍在Current Approach背景中，不混比。

自适应组当时完成5轮，尚未到容量更新边界，loss有限/进程存活。不得用未实施变化前的best宣称容量更新改善，也不将未知早期低指标当失败。完整比较待两组20轮结束后进行。

附加无标签解释脚本diagnose_a2w_resize_rejection_colab.py已准备，只在训练完成/模型持久保存后读取第10轮缓存logits及head row_mapping。在纯删减未知行、已知权重不变的情况下重建即时拒识变化，无图像前向/重评checkpoint/调参；不能将softmax分母变小直接解释为更多argmax拒识。后续学习可能改变这一即时结果，必须区分。

## 本批结束：40轮预算完成，整体目标未完成

refresh亦正常20轮exit0，无OOM/NaN；K8→K4更新后保持K4，模型/日志实际保存Drive。collector修复一次纯字符串续行错误后成功汇总完整20轮及设置，未重训/重评checkpoint。summary和3.0MiB ZIP均已下载本地pipeline-results；即时pruning报告已保存Drive与本地。

final HOS差值+8.1101pp，best差值仅+0.4004pp。前10轮最大三指标偏差1.7045pp，第10轮预先已有HOS+0.7337pp；不能称严格确定性因果对照。即时删头全部564个known/unknown预测均不变（unknown223→223），支持将收益归于后续学习而非立即改变拒识。完整表、边界与后续安排见final_report.md。

本批max_iterations1已完成，不自动追加训练/扩矩阵，不将总体目标标完成。用户无人值守意图保留；下一批固定K2/阶段性容量70轮预算与是否释放闲置T4已询问，当前CPU与T4仍保留，不擅自销毁。
