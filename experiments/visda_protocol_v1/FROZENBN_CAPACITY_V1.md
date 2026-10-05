# VisDA 全量跨任务验证：冻结当前候选规则

2026-10-04，在本任务训练/估计/评价结果产生前声明。当前source archive正在恢复；没有把历史数据准备结果当当前就绪。

任务Synthetic→Real validation：source原ID1/2/3/6/10/11，共79765图；target全12类55388图。ID依据官方列表与既有论文协议，source映射连续0..5，target语义仅协议/评测使用，不进入特征或K拟合。

第一阶段独立运行source监督：ResNet50/ImageNet、C6头、seed1、全量3轮，batch64/drop_last/workers4；同OfficeHome的SGD encoder/head lr5e-5/5e-4、momentum.9/nesterov/wd5e-4、inverse decay、Resize256x256/RandomCrop224/HorizontalFlip。encoder BN统计保持，权重/affine/head正常训练，无teacher/新loss/DINO。3轮约3738步，因数据规模不同，不声称跨任务同一步数预算。使用target-unlabeled-paths.txt，训练后eval/CenterCrop提取全量256维瓶颈。

第二阶段单独估计：同已冻结的source-calibrated-birth-cost-v1，source70%原型/15%成本校准，seed2026、source99%半径、source负对照最大初始gain*(1+1e-6)、source类计数先验、R类平均计数、moving known/birth-first。使用proposal_block_size256的完整候选分块接口，无采样、无target标签、无阈值网格。内存改善不解决二次计算时间；耗时/收敛仍须实际验证。

第三阶段在source/容量均完成后才能独立启动：固定K2与估计K共享该prior、seed1、ResNet50、Q8（C6+论文固定K2）、保持encoder BN、原发布代码损失/伪标签/argmax/初始化，初期各10轮。这是跨任务短程候选验证，不是已确认论文VisDA总预算或解决VGG表标题矛盾。K仅传整数，不用IMP中心初始化头；估计组保持Q8以隔离头容量，不按估计K自动改Q。

K0原样记录失败，不强制K1；K2与固定组方法输入相同，仅运行一个arm，不伪称独立容量提升；保护上限或未收敛必须保留失败，不当作真实未知数。不同于原论文增加的source3轮及BN模式明确保留在版本描述。

不按target指标挑seed/阈值/K或扩预算。最终保留全部best/final OS*/UNK/HOS及已知/未知取舍；best若使用target标签选epoch则明确oracle探索，非无偏验证或统计显著性。先按阶段验证，失败定位直接原因，不自动更改研究设置重跑。

source launcher：scripts/run_visda_frozenbn_source_colab.py（1fd9885f）；训练入口单独部署为/content/train_visda_frozenbn_source_v1.py，未替换历史OfficeHome入口。日志间隔200步只影响输出，默认log_every0保持旧行为。计划输出/content/imp-runs/visda-frozenbn-capacity-10e-v1。当前只有脚本已部署，source/容量/RTA均未开始。

推断入口已准备scripts/infer_visda_source_cost_colab.py：全量shape与C6/source3检查后调用既定分块接口，保存普通capacity.json及中心/分配capacity.npz，以便必要诊断，不做checkpoint前向。核心将隔离部署到/content/visda-capacity-code-v1，避免误用历史OfficeHome的旧模块；当前推断尚未运行。K结果如为0或不收敛仍保留失败，不据此改规则。

实际进展：source3为exec164，首轮完成、第二轮训练中；全量数据已恢复、无标签target列表准备完成。推断入口及分块核心已隔离部署，容量/RTA未启动。

后续RTA独立入口scripts/run_visda_frozenbn_rta_colab.py已准备并部署：fixed2/estimated单arm、共享source3、seed1、Q8、保持encoder BN、原版初始化/损失/筛选/预测、10轮；不传簇身份。必须先取得完整source summary和收敛正K，K2相同时不重复同一arm；K0/失败不强制改K。类别映射使用仓库class-map.json，ResNet口径和原总预算不伪称已解决。当前source正在第3轮，容量及RTA均未执行。结构版OfficeHome为另一个实验，不能混称K-only。

实际进展：exec164完整完成source3（3738步，最终source CE0.080924、训练accuracy0.97492）及全量79765×256 source/55388×256 target特征，不能当target成绩。普通配置/日志/summary已保存并下载pipeline-results/visda-frozenbn-source3-v1-results.zip。CPU容量推断独立exec166已启动，日志capacity-console.log，尚无K结果；GPU交给OfficeHome结构对照，不同时运行两项GPU训练。VisDA RTA尚未启动。

排程修正（未见VisDA K或目标成绩前）：OfficeHome结构10轮已完整结束，GPU释放；fixed2控制只依赖完整source prior，不依赖估计K，因此允许CPU推断仍在运行时独立启动该控制，避免无谓空置GPU。研究参数不变，不串联两项GPU任务。estimated仍须完成收敛正K，K2则复用等价控制，不强制K1；该排程不表示估计已成功。

fixed2已独立shell4启动，已观察train_visda_frozenbn_rta_entry.py进程及TASK_KONLY_START C6/K2/Q8，encoder BN policy53模块保持；路径/content/imp-runs/visda-frozenbn-capacity-10e-v1/fixed2/visda-synthetic2real_seed1。尚无完整epoch指标。CPU容量仍为exec166，独立shell3可检查文件/进程；不得再启动其他GPU训练与本控制竞争。

## 2026-10-05：runtime 丢失与恢复检查

CLI runtime list 已确认没有活跃 runtime，旧 shell4 不存在。现有本地 exec166.ndjson 只有 VISDA_CAPACITY_START，没有 K 或完成事件；因此容量推断状态记为中断/结果未取得，不能宣称收敛。fixed2 只确认启动过，尚无取得的完整 epoch 指标，不能作为已完成实验。

本地 visda-frozenbn-source3-v1-results.zip 保存 source3 的配置、日志和 summary，证明预热及特征提取曾完成，但收集器没有打包 source-final.pt 或 features.npz。两者此前保存在临时 /content；目前没有可用恢复副本的证据。Drive 授权有效，但列 OSDA 文件夹受共享 OAuth 客户端 Drive 查询限额阻止；尚未完成云盘恢复检查，不能断言云盘绝无副本。没有自动创建 runtime、重跑训练或改研究参数。

工程修正已准备 scripts/persist_visda_source_stage_colab.py：未来 source 阶段完成后、容量/RTA 启动前，将 source checkpoint、features、普通日志及协议列表复制到已挂载 MyDrive 的新目录，最后写 recovery-complete.json。无 hash、无 checkpoint 重评、不覆盖旧文件；中断复制不能当完成备份。该脚本尚未在真实 runtime 执行，不能追回已经丢失的数据；它也不提供 RTA 优化器续训，图片和运行环境仍须另行恢复。是否恢复或重跑须在检查现有云盘材料后决定。

后续读取云盘 OSDA 根目录成功，包含 datasets/pretrained/runs；datasets 成功列出 OfficeHome 目录、Office31 archive及旧 manifest，未在该目录列出 VisDA。runs 列表仍因共享客户端查询限额失败，不能排除其下存在恢复材料。新 runtime/重跑尚待用户确认；仅本地准备先验总权重消融，未对正在恢复的 VisDA 研究设置应用修改。

## 授权恢复（2026-10-05）

用户明确回复“允许”新建 L4、无副本时按原设置恢复必要阶段。runs 目录随后成功读取，仅列出历史 Office31/早期IMP目录，没有列出当前 VisDA/C20 的恢复目录。新 L4 endpoint gpu-l4-s-kkb-ass1a1-28xqgmbpgb0gj 已确认 NVIDIA L4，旧 Py3.8/Torch1.7.1环境恢复。C20 frozen-BN source3 已恢复并保存模型/特征到本地，先验权重消融结果不支持 domain_balanced，VisDA 保持原 source_counts 设置。

VisDA 当前仅重新开始官方 train archive 下载，exec10，source/RTA/容量尚未恢复完成。不串联未确认阶段，不将旧源预热日志当新模型。原source3/seed1/ResNet50/frozenBN、Q8、RTA10预算均不变，恢复模型不宣称与丢失模型逐位一致。Drive 挂载 exec5授权超时；收集器新增显式 --include-recovery，完成新source后须下载含source-final.pt/features.npz的恢复包，再启动容量或RTA。默认普通日志包不变；不hash、不重评模型，图片/环境单独恢复。本阶段没有开启新的参数网格或新方法训练。

用户追问是否每次重新下载后，补充数据长期保存路径：MyDrive/OSDA/datasets/visda-syn2real-v1/{train,validation}.tar。scripts/persist_visda_archive_colab.py 单独保存一个已完整下载的压缩包，先复制 .partial、成功后改名，保留既有或中断副本，不hash、不覆盖旧数据。两个 download 脚本在 /content 尚无文件且 Drive 缓存可用时优先复制，不从官网重复下载；未来 runtime 应先挂载 Drive 再恢复数据。当前下载仍使用同一exec10，不中途重启；上述保存尚未执行，Drive挂载仍需新的授权。不得把“保存脚本已准备”当成“8.7 GB数据已经持久保存”。

恢复进展：train exec10完整结束（7698031104 bytes），源6类解包完成79765图；validation 独立shell6下载完整结束（1023758336 bytes），target准备已启动。用户完成授权，挂载exec11正常结束。train复制exec12正常结束，挂载目录中最终文件存在且大小正确，云端目录ID1M-8hf4AXd3wnU3iSOJR5S2QDCQnPqoYJ；第一次Drive API列表仅看到train-saved.json，尚未看到train.tar，后续查询被共享客户端限流。DriveFS可能异步上传，不能把文件系统复制完成当云端已经可恢复。validation复制为exec13。新source/RTA/容量均未启动；先等大文件在Drive API可见，再开始源预热。

保存器完成标记的语义修正：未来版本写copy_complete而非complete，明确cloud_upload_confirmation尚待独立确认；不改写既有标记以伪造状态。本次旧train/validation标记中的complete只证明挂载目录复制结束，最终云端可用性仍需API文件出现。没有做hash或重复数据检查；这是为了避免runtime回收前尚未上传完导致再次重下。

最终持久化确认：Drive API列出train.tar，ID1J3HiW5HDT_S_ciuvMeffYdVkWS52JCmh、大小7698031104；validation.tar，ID1UTxuO7arJ2tqy2iwfjAmKhHRZcKpANqn、大小1023758336。两包均已云端可见，可供新runtime恢复，不仅是DriveFS待上传文件。target准备完整55388图，原source6类79765图；散图仅存Colab本地。随后source3独立启动；新容量与RTA未启动。

新source为exec14，已观察第1轮batch400平均CE0.396204，进程仍running，无新目标成绩。完整RTA结果收集器scripts/collect_visda_capacity_colab.py已准备，只读普通日志报告best/final及同预算差值；K2复用等价固定组，不制造第二个观察；K0明确估计组不运行/失败，不强制K1。一次指标接口检查通过，尚未执行真实VisDA收集、不前向checkpoint。source完成后必须保存模型/特征，然后容量及固定2控制按既定独立排程进行。

恢复阶段已完整完成：exec14 三轮 source CE 分别为 0.244433、0.111253、0.081264，最终源训练准确率 97.4569%，共3738次更新；source/target特征为79765×256与55388×256，不含target标签。这不是目标域性能。exec15完成含模型与特征的恢复归档，已实际下载到本地 pipeline-results/visda-frozenbn-source3-recovery-20261005.zip（214.9 MiB），不同于旧日志-only包；未做hash或checkpoint重评。

在恢复材料保存之后，全量CPU容量推断exec18已启动；固定K2十轮控制独立shell5已启动，已观察TRAIN START。二者共享上述source prior，容量仍使用原source_counts，未推广效果不佳的domain_balanced。当前尚无K结果或完整RTA epoch成绩。estimated组仍等待收敛正K；K2复用控制、K0不强制K1。新runtime保持gpu-l4-s-kkb-ass1a1-28xqgmbpgb0gj，不启动第二项GPU训练。

固定K2首轮history已写出：epoch1，OS*=60.873605%、UNK=0%、HOS=0%，累计1570.38秒；控制台CE0.059、virtual0.063等损失有限。此时处于原版预热阶段，未知头尚未完成warm-end初始化，不能据首轮UNK0认定最终拒识失败。继续既定10轮，不调损失/seed/预算。CPU容量进程22657运行约26分钟且持续计算，仍未取得K。

第2轮已完整写出history：OS*=58.906786%、UNK=0%、HOS=0%，累计2825.88秒。已知类指标较首轮下降1.96682个百分点，但仍属预热阶段；不以目标指标改参数、提前停训或扩预算。容量进程运行约48分钟、持续CPU计算，尚未写出K结果。

## 全量容量与固定组完成

容量已成功结束（exec18 returncode0），耗时5058.78秒，63步收敛，K=8，无target标签。六个已知中心吸收target数为[183,25,211,871,32,138]，八个候选未知簇支持数为[11591,5970,10206,5134,7775,3512,6897,2770]，noise73。约97.2%的target被分到新簇，这提示源特征下的域偏移/已知先验适配需要诊断，不能把K8直接称为真实未知数。容量普通JSON已下载 pipeline-results/visda-capacity-restored-20261005.json。

固定K2完整10轮正常结束，耗时12920.11秒。目标标签oracle best为第4轮：OS*=53.550094%、UNK=32.632558%、HOS=40.552861%；第10轮final为OS*=0.211454%、UNK=99.907958%、HOS=0.422014%。从第5轮开始明显转向几乎全部拒识，说明固定小K也存在后期塌缩，不能归因于增加K。全部10轮history已下载 pipeline-results/visda-fixed2-restored-20261005-history.jsonl；没有重评checkpoint或事后换seed。

随后在GPU固定组结束后，estimated K8独立shell5已启动，已观察TASK_KONLY_START C6/K8/Q8、53个encoder BN统计保持。共享同一个source3 prior、seed1和10轮预算，仍不传簇身份、不改损失/筛选/预测。当前尚无estimated完整轮次，必须等待后再判断容量改动能否改善或只是同类塌缩；不提前将失败控制替换成更易出好成绩的协议。
