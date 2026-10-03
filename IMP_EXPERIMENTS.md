# IMP / 未知空间建模：分阶段实验记录

## 研究目标与实验边界

利用 source 已知类别结构作为先验，推断 target 原型结构，再配置未知分类空间容量。
原型数量不预设等于真实语义类别数。target 标签只用于离线评价，不用于先验、聚类或容量选择。
现有用户训练代码及已完成 RTA baseline 保留；每一阶段独立执行、检查结果，再进入下一阶段。

### 三个数据集各一个任务（2026-10-03 固定计划）

按用户要求，正式对比覆盖 Office-31、Office-Home、VisDA，各一个任务。
项目 PDF 为 TIP 2025 扩展版，实际还包含 ImageCLEF；第四个数据集不擅自加入本轮范围。
任务在 IMP 成绩产生前选定，不根据结果换任务：

| 数据集 | 任务 | 已知/目标未知语义类 | RTA 未知槽 K | 论文单任务 OS*/UNK/HOS (%) |
| --- | --- | --- | --- | --- |
| Office-31 | A→W | 10/11，沿用官方列表 | 2 | 92.2/93.8/93.0（Table I） |
| Office-Home | Pr→Rw | 25/40，具体类别列表须审核 | 4 | 82.1/77.2/79.5（Table II） |
| VisDA | Synthetic→Real | 6/6，具体类别映射须审核 | 2 | 73.6/83.7/78.3（Table III） |

Office-Home 选择 Table II 的首个任务，不按最优成绩筛选。VisDA Table III 的已知类别列为 Bicycle、Bus、Car、M-cycle、Train、Truck，不能简单假定为数字标签 0–5。
**协议疑点**：正文实现段落写 ResNet-50，但 Table III 标题写 VGGNet。VisDA 复现前必须确定 backbone 和原始代码设置；若不能确认，只能报告同环境受控比较，不能声称精确复现该表。
三个任务均要求同数据列表、同环境、同训练预算的 baseline/IMP 对照；报告 seed 1/2/3 的均值和样本标准差，固定 final 与目标标签 oracle-best 分开。仅有 Office-31 旧 T4 baseline 不满足三任务完成条件。
类别数仅为评价协议，不参与 IMP 容量推断。目标标签及论文真实未知类数不用于选阈值、先验强度、容量或任务。

## Stage 0：当前实现数值审计 — 已完成

- 日期：2026-10-02。
- 通过 MurphyLo/colab-cli 执行，Colab exec ID 23，状态 done。
- 端点：`gpu-t4-s-kkb-ass1c2-g6yzd4pmjjf0`。
- 会话硬件：Tesla T4，15360 MiB；本次实际计算设备 **CPU**，未进行 GPU 训练。
- 环境：原 baseline 的 `/content/rta-py38/bin/python`，torch 1.7.1+cu110。
- 审计对象：未修改的 `IMPClusterer.py`。
- 输入 SHA256：`e25d76ae5cff8ed6927573b534cc5f666d781e10cfd8534b2bea00e704fbf4ba`。
- 合成数据：seed 1，256 维，14 个分离方向，每个方向 5 个扰动样本，总计 70 个；另含退化和下溢压力输入。
- 不涉及 Office-31 特征、准确率或真实未知类别数估计，不是正式 IMP 实验成绩。

| 检查 | 实际结果 | 判断 |
| --- | --- | --- |
| 14 个分离方向 | 返回 14 个簇，中心有限，软分配行和为 1 | 基础建簇在此合成案例可用 |
| `max_clusters=1` | 仍返回 14 个簇 | 上限未执行 |
| 倒序、打乱顺序 | 此案例均为 14 个簇 | 此例稳定，不证明一般顺序不敏感 |
| 20 个相同特征 | sigma=0，阈值、中心及软分配产生 NaN | 零方差处理失败 |
| 单样本 | 返回 1 个有限中心 | 此退化案例通过 |
| 远距离＋窄尺度 | 软分配三行的概率和均为 0 | 指数下溢，归一化失败 |
| α=.01/.05/.1 | 阈值约 1.4686/1.2737/1.1898 | α 越小创建阈值越大；与代码注释相反 |

结论：当前实现不能直接用于可靠的容量推断。先修复数值与容量约束，再审计 source 先验接入。
普通合成案例通过不能抵消退化输入失败，也不能证明已恢复未知语义类别数。

### 证据与复现

- 审计脚本：`scripts/audit_imp_stage0.py`。
- Colab 单阶段执行器：`scripts/run_imp_stage0_colab.py`。
- 本地结果：`pipeline-results/imp-stage0/audit.json`、`console.log`。
- Drive 持久化：`/content/drive/MyDrive/OSDA/runs/imp-stage0-audit-v1/`。
- 执行器拒绝覆盖已有结果目录；重复实验应另设版本，不删除旧记录。

## 分阶段进展与后续顺序

### 方差下界补丁 — 已完成并验证

按用户要求直接在 `IMPClusterer.py` 添加可配置的 `min_variance=1e-8`。
在阈值估计中对 rho 截断，在软分配中对 sigma² 截断；下界同时不低于当前 dtype 的最小正规正数。
这是数值保护，不是对原 IMP 方差理论的重新对齐，也不处理非法 alpha 或非有限输入。

Colab exec 24 已完成，沿用 Stage 0 同一测试协议、同一 CPU 环境。
补丁 SHA256：`3afabbf3b47423342e6076b86aa15ec039543345a683dbe04b0aed5946cd0015`。
全同特征测试：sigma≈1e-4，阈值≈0.0006017，返回 1 个有限中心，软分配行和为 1，无 NaN。
普通 14 簇、倒序、乱序及单样本测试的簇数与原审计相同；普通案例的中心误差指标未变。
本轮仅修复零方差除零：簇数上限仍不生效，远距离窄尺度压力测试仍可能下溢。

结果保留在 Drive `OSDA/runs/imp-variance-floor-v1/`，本地 `pipeline-results/imp-variance-floor/`。
结果 JSON 的 stage 字段沿用旧审计脚本名称，应根据目录和模块 SHA256 区分补丁版与原版。
旧 Stage 0 和 baseline 结果均未覆盖；没有启动训练。

### Stage 1：稳定归一化与容量保护 — 已验证

Colab exec 25：先单独改为距离平移后的 softmax。下溢压力测试行和变为 `[1,1,1]`，有限值成立；普通 14 簇案例及退化案例通过。版本 SHA256 `cc842184ec55a0a07503839ad4e42e50d6a0e8fea5cac946a8bc6b920234dd7f`。

Colab exec 26：再加入容量及输入检查。`max_clusters=1` 时显式报错，不静默返回被截断的簇数；普通案例仍返回 14 个簇。版本 SHA256 `ee30950a06655d99ab9a826effbf7adf807718a9f24f9bb1ee1a5c57f81335b4`。这不是自动选择容量，上限饱和代表无法可靠报告数量。

分别持久化到 Drive `OSDA/runs/imp-stable-normalization-v1/` 与 `imp-capacity-guard-v1/`；本地结果为 `pipeline-results/imp-stable/` 与 `imp-capacity/`。未修复版和各补丁版的 runtime 文件均保留。

### Stage 2：source 锚定原型机制 — 合成测试通过，尚未接入训练

新增独立模块 `source_anchored_imp.py`，保留已知原型身份，按固定软归属的 Gaussian MAP 形式更新已知中心；候选原型由目标数据建立。返回软归属、有效支持量及候选数，明确不返回真实未知语义类别数。

这是阈值式 IMP-inspired 模块，不是完整 DP 后验；观测方差与阈值在机制测试中显式固定。没有修改 RTA 训练入口，也没有将候选数直接映射到未知输出槽。

Colab exec 27 完成 5 项 CPU 合成测试：

1. source 中心为 0，20 个 target 点位于 .4。先验强度 0 的中心为 .4；强度 100 的中心约 .06667，与解析更新值相符。
2. 两个已知中心轻度偏移，加一个新结构：保留两个已知身份，生成一个候选；软归属正确归一化，此例倒序结果相同。
3. 相同输入无 NaN，未生成候选。
4. **反例**：真实仍是一个已知类，但偏移到距离较远的位置时，产生一个候选。因此新增候选不等于未知类。
5. 容量饱和显式报错。

测试使用 2 维可解释合成数据，不能据此声称 Office-31 有效，也不能证明任意多模态场景已解决。
测试脚本 `scripts/audit_source_anchored_imp.py`；模块 SHA256 `3f29c03de9e1979eb4ffbccaadfad1430bdb84f689b156b83e584715e59c2087`。
Drive `OSDA/runs/source-anchored-synthetic-v1/` 保存模块、测试源码、日志和 JSON；本地 `pipeline-results/source-anchored/audit.json`。

**下一行动**：真实 source 预热和冻结特征提取；比较阈值、先验强度、域偏移与候选归属。已观察到的偏移反例要求增加已知结构解释能力/关系诊断，而非直接动态扩头。

三项实验日志及各阶段模块快照已打包下载到 `pipeline-results/imp-mechanism-results-v1.zip`，并解压到 `pipeline-results/imp-mechanism-v1/`。
归档 SHA256：`eb25b4d824b31123c87e634afa34a56504f89c974416c712151abf4b112099b2`，已与远端核对。
Drive 同时保存 `OSDA/runs/imp-mechanism-results-v1.zip`。个别单文件下载出现临时网络失败，通过完整归档补齐证据，未重跑实验。

### 尚待执行

3. Stage 3：L4 上进行源域预热，冻结模型，提取 Office-31 source/target 特征。不得使用适应后的 best checkpoint 代替干净的预热表示。
4. Stage 4：推断目标结构、分析稳定性、支持量及未知归属；明确容量选择规则后，接入一次性未知分类头配置。
5. Stage 5：短程功能测试通过后，做同环境 RTA / 仅虚拟原型自适应 / 未知容量自适应的完整多 seed 对比。软原型监督另做消融。

### Stage 3：L4 预热已完成，但产物未持久化，需显式恢复

通过 colab-cli 创建 L4 High-RAM 会话，端点 `gpu-l4-s-kkb-ass1a0-3repicqqt39xy`。
L4 exec 1 已完成 GPU 矩阵计算预检：NVIDIA L4，23034 MiB，计算能力 8.9，有限结果成立。
环境是 Python 3.13.15、torch 2.11.0+cu130、torchvision 0.26.0+cu130、CUDA 13.0。
与 T4 baseline 旧环境不同，不能用于将跨环境差异归因于 IMP；后续对照须使用同一环境。

源域预热/特征提取脚本 `scripts/source_warmup_features.py` 已在 L4 exec 11 执行成功。
采用 ResNet-50 ImageNet 权重、256 维归一化瓶颈、12 输出，source CE 单独训练 3 轮，batch64，seed1。
特征提取使用确定性中心裁剪及 eval/no_grad；target 标签不进入训练 loader，单独保存评价文件。
记录配置、输入列表与权重哈希、逐轮 source 指标、冻结特征、checkpoint 和产物哈希。
这不是官方 RTA warmup 的精确复现：没有虚拟损失、DomainBus 或 target 训练，workers=0，新框架环境也不同。

数据准备 exec 8 完成；模型 smoke exec 9 因历史权重格式失败，校验固定权重哈希后允许加载可信 legacy 文件，exec 10 前向/反向成功。exec 11 完成 3 个 epoch、42 个 step，冻结 source 准确率 95.6159%，特征 source `[958,256]`、target `[564,256]`，全部有限。该 source 指标不是 OSDA 成绩。

L4 exec 2 的 Drive 挂载授权已经确认超时，错误证据保留。
L4 exec 3 也已授权超时。Drive API 出现配额错误，因此通过 fs 上传已有且校验过的本地数据/权重，在 `/content/imp-runs/source-warmup-l4-seed1-v1` 运行。

2026-10-03 两次 runtime list 确认 **No active runtimes**；本地 CLI 执行历史止于 exec 11，没有归档收集或冻结结构实验的执行记录。本地未发现 features/checkpoint 归档，因此不能继续使用上一运行时产物。已将原始完成日志保存到 `pipeline-results/source-warmup-l4-seed1-v1/exec11.ndjson`，保留日志中的配置及产物哈希；不把它当成完整产物可用的证明。

日志 SHA256：`00c495150c50f8bddcaeb92442acebbcb6eca59260139fa84352bf80f47da920`。
下一轮必须使用新版本目录显式重跑，并在每个阶段完成后立即下载、校验特征及 checkpoint；不是复用旧完成记录。冻结结构 grid 尚未执行，未知槽配置及三数据集对照均未完成。

恢复执行：已新建 L4 `gpu-l4-s-kkb-ass1a1-13n8bs33o8id1`，exec 1 计算预检通过，Python/torch/torchvision/CUDA 与上一轮相同。Drive API 再次返回配额错误，故使用已校验的本地输入经 fs 传输。恢复版 runner 默认目录为 `source-warmup-l4-seed1-v2`；归档器和结构 runner 同步指定该版本，不覆盖 v1。允许通过经 basename 检查的 `IMP_WARMUP_RUN` 指定其他显式版本。

#### v2 恢复及持久化 — 已核验

新 L4 exec 4 数据校验/解包完成，exec 5 forward/backward smoke 通过，exec 6 source-only 预热完成，exec 7 归档审计通过。配置及训练脚本与 v1 相同，特征、评价文件、checkpoint 哈希均逐字节一致；这次是恢复输入，不新增统计种子。
本地特征包 `pipeline-results/source-warmup-l4-seed1-v2-features.zip` SHA256 `078a9deed0eebbf06b5246031c74d67c0a8dc32e7c23b6126dafa94928512495`；完整包 `source-warmup-l4-seed1-v2-complete.zip` SHA256 `8547c163f9cd473bfab4a5a3e3f42348b038633f061f1f4f079cc3604156b83f`，均与远端核对并解压。
解压 checkpoint SHA256 `fbbd30a2284750ab5707766c16d032ccbdefb324694e0ca5943131511de0798c`。因此后续不依赖临时运行时找回特征。下载曾出现 fetch failed，仅重试传输，没有重跑训练。

### Stage 4a：冻结真实特征的阈值/先验敏感性 — 已完成

新 L4 exec 8 完成预先声明的 3×3 grid，exec 11 收集归档。推断仅读取 features.npz 中 source/target features 与 source 标签，不读取 evaluation-only.npz；未使用目标标签评分或选参。
source 类内平方半径 99% 分位数为 `0.535741209983826`，各维平均残差方差 `0.000875418540090322`。这是 in-sample calibration，不是独立验证集校准；不能宣称校准了目标未知类覆盖概率。

| 阈值倍数 | 先验强度 κ=0/5/20 的候选数量 | 观察 |
| --- | --- | --- |
| 0.5 | 三次均显式容量报错 | 达到总原型上限100，数量不可报告为90未知类 |
| 1 | 7/7/7 | 三次均只有5个候选有效质量≥5，候选软概率均值约0.11–0.12 |
| 2 | 0/0/0 | 所有 target 被当前已知原型结构解释，不能据此断言没有未知类 |

六个成功 trial 的软归属行和均为1，未出现 NaN。**数值稳定不等于数量推断可靠**：当前阈值敏感，且先验强度未改变候选数。7是候选几何结构数，不是未知语义类数；这轮不据此设置未知槽，不选表现最好的阈值。
报告 `pipeline-results/source-anchored-frozen-grid-v2-report.json`。完整归档 `source-anchored-frozen-grid-v2.zip` 已下载、核对 SHA256 `7b4106fd70fcbdef61efe1d3a160a59f45485e5d108839ebcd797635f8be66b1` 并解压，含每次软分配、支持量、原型、完整失败记录及代码快照。

下一步：在同一冻结特征上先固定规则检查样本顺序/子采样稳定性，再分析新增候选是否来自已知域偏移。进一步加入 source 持出校准/已知关系诊断时，要单独留版本与对照，不把新增原型自动全部认作未知类。完成这些 gate 后才进入未知容量头的短程功能测试；三个数据集各一任务的完整 baseline/IMP 实验仍未完成。

### Stage 4b：顺序/子采样及离线语义归属 — 已完成

沿用新 L4；exec 12 执行固定阈值1×、κ=5、steps=5、总容量100的12项检查。原序、倒序及5次固定seed打乱使用全部564样本拟合；另5次用451样本拟合，所有trial都在完整564样本上输出软分配。没有改训练、读目标标签或挑选trial。

| 输入顺序/子采样 | 候选数 |
| --- | --- |
| 原序/倒序 | 7/5 |
| shuffle seeds 1–5 | 5/6/5/7/6 |
| 80% subset seeds 1–5 | 5/5/4/7/5 |

候选样本集合相对原序 Jaccard 为0.9385–1.0；全部结果有限且软分配行和为1。这证明当前样本级候选区域比组件个数更稳定，而非证明任意换序均可靠。不能用稳定候选区域替代数量稳定性 gate。
代码 `scripts/audit_frozen_structure_stability.py`，结果 `pipeline-results/source-anchored-stability-v1/`，归档 SHA256 `aa7303c4ccd3b2b4f89159ac5f27214a662adde09956c4f416a08081955ddeb8` 已核对。

随后单独运行离线评价：exec 13 因 notebook argv 带 -f 参数退出；没有结果产生。修复执行方式，exec 14 用独立 Python subprocess 运行 `evaluate_frozen_candidates.py`，推断文件保持不变，真实target标签只进入该评价脚本。没有重新推断或利用标签修改参数。
固定原序结果：295个已知、269个未知样本；64个样本硬分到候选，其中56个未知、8个已知。候选precision=87.5%，unknown recall=20.8178%，已知误入率=2.7119%。这些不是训练后OSDA成绩。
7个候选中两个完全由已知样本组成（2个与6个样本）。其余候选包括未知标签20和25的混合，以及同一未知标签25的碎片化；不能视为7个未知语义类。
评价归档 `pipeline-results/source-anchored-attribution-v1.zip` SHA256 `935eebcd5732cdd33bbad8e7a1c84f184113a5b49d52df64d40ddcc89e185a03` 已核对并解压；包含固定推断输入哈希、标签文件哈希、逐候选构成及评价源码。

#### 观察后的下一项调整

不将当前7个候选接入未知头。将“未知倾向判别”与“候选内部结构发现”拆开：先审计官方 RTA 的 source class soft-label prototype / KL / mixture 判别，再考虑以连续未知倾向约束新组件，避免远离source均值就被自动认作未知。
对齐依据：官方代码用 `F.kl_div(log(p_target), source_soft_prototype[pseudo_class])`，即 source-prototype||target 方向；训练早期 batch GMM=3，epoch末 BGMM最多4/2组件。该一维mixture组件并不是未知语义类数量。冻结特征上若实现全量fixed BGMM审计，须明确它不是官方逐轮RTA训练，也不能替代原warmup。
未知结构数量规则还需source持出校准、顺序稳定性/支持量机制及受控消融。上述target标签归属是探索性分析；任何受其启发的改动不能再把当前特征上的表现称为独立验证成绩。阈值/先验参数须由source或预先声明规则决定，最终三任务baseline/IMP仍按固定协议报告所有seed，不根据目标标签挑最好设置或最好seed。

### Stage 4c：RTA 类间关系信号审计 — 已完成

新增独立 `relation_gate.py`，按官方代码方向计算 source-soft-prototype||target-probability 的 KL。source原型由source标签对应的已知类条件softmax均值构成，target仅用已知类logits的argmax选对应原型。不读target标签；每个source已知类必须有样本，输入有限且类型/device匹配。
数值上使用float64和log_softmax代替log(softmax)，保持KL方向，避免极端logits下溢。相同关系分布KL≈0及极端logits有限性两项测试通过。这不是证明未知语义能被KL准确识别。

新L4 exec 15完成 `audit_relation_gate_colab.py`，复用固定source-only checkpoint logits。预先固定 BGMM最多4组件、max_iter=800、random_state=1/2/3，三个trial全部报告，不以目标标签选seed。结果：

| mixture 初始化seed | 收敛步数 | 最低KL组硬样本数 | 最高KL组硬样本数 | 非最低KL组硬样本数 |
| --- | --- | --- | --- | --- |
| 1 | 285 | 196 | 27 | 368 |
| 2 | 286 | 196 | 27 | 368 |
| 3 | 170 | 195 | 37 | 369 |

三次均收敛且无warning。每次都有一个无硬归属的低权重组件，因此组件上限4不能解释为4类或4个有效语义结构。非最低KL组规模近似稳定，而最高KL组受初始化影响；规模相似本身不证明样本集合完全相同。
这是全量冻结 logits 上的诊断，区别于官方初始化batch GMM3/逐轮BGMM4→2训练：source-only表示没有虚拟空间预留损失，不能据此推断官方RTA判别性能。该诊断未读取目标标签或计算目标准确率。
完整归档 `pipeline-results/relation-gate-frozen-v1.zip` SHA256 `ce610f980eed7e5b0f06bbb2d488cb23e21ec8f1af140a4185804effbedd800c` 已下载校验并解压；包含source/target KL、soft原型、每seed后验、连续最低/最高组概率及报告。三个mixture seed是初始化敏感性测试，不是三个模型训练seed。

下一实施方向：保留原RTA未知判别/空间预留路径，以连续已知兼容度约束候选组件，而非用最高KL组硬标签替代全部未知样本；候选支持量及容量决定必须另有稳定性规则。先建立与RTA空间预留预热对齐的同环境控制，不能只在当前source-only表示上筛选出最有利的gate再声称IMP有效。尚未集成未知头，三个数据集各一任务的完整对照仍待完成。

### Stage 4d：官方空间预留路径的 L4 预热参考 — 训练/特征完成

继续使用同一L4。native环境缺少FAISS，exec17只安装 `faiss-cpu==1.12.0 --no-deps`，K-means预检通过；torch2.11.0+cu130、numpy2.1.3未升级。该FAISS版本与旧T4环境不同。
从已经审计的 `artifacts/rta-baseline` 复制6个代码文件到隔离 `artifacts/rta-l4-control-v1`，没有编辑用户root训练代码或旧baseline。与该基点比较，只有main训练轮数的env开关以及networks固定权重哈希校验/可信legacy加载两个新增变动；其余4文件保持不变。代码包SHA256 `5e14fa31467b1eb8d2a960273994bf48f7a9edeac03b081bc0738cb4a923ba58`；exec18完成原CLS前向/虚拟CE反向预检。

exec20用官方流程执行固定4个完整epoch（索引0–3），seed1，batch64，source/target增广及DomainBus沿用原代码，K_cluster=20、K=2、faiss niter800保持不变。`warmiter=3` 与 `epoch<=warmiter` 实际意味着4轮source CE+virtual CE，而不是3轮。训练中仍计算但不优化adv/entropy/unknown CE；这些代码未删除。保留原epoch3未知头初始化。
四轮source CE均值为2.003/0.730/0.382/0.267；virtual CE为2.739/1.186/0.632/0.430；无训练错误。这是空间预留预热参考，不是70轮完整baseline，不是论文成绩复现。
训练日志仍包含官方目标标签评价和best.pt，但后续推断明确只读取预先固定的 `last.pt`（epoch=4），不使用oracle-best权重。只读评价没有用于参数或checkpoint选择。

exec21完成固定last的确定性中心裁剪特征：source[958,256]、target[564,256]，特征和logits全部有限，source accuracy96.1378%。目标标签单独放evaluation-only.npz，不进入features.npz。与此前3轮source-only预热在轮数、初始化/数据迭代、BN更新与损失等方面有差异，不能据二者对比声称已验证virtual CE的因果作用；消融必须另做匹配控制。
特征包 `pipeline-results/rta-space-warmup-l4-features-v1.zip` 已下载核对SHA256 `9a817caba3a044b96b9ad4b8d601888c56ff8cf3b97d7fb5af7e9c254d937155` 并解压，含6份完整代码快照、训练audit/config/history/console和feature manifest。
feature SHA256 `75cabbddd74d812dd9780b95248e1cc7dc8147d82ac336659162a2f642241823`；固定checkpoint SHA256 `0631572940759e0677632a1eadeb5bb4748912fa071e706a508df23411e03cd3`。
exec22已生成checkpoint归档（不含best.pt）196215458字节，SHA256 `c5a59216baa56552efcbb5ce197f2d1c64083925c89e769f69ee54258d706507`；已下载、校验、解压，固定last.pt哈希与提取manifest一致。完整代码另保存于 `experiments/rta_l4_control_v1/`，便于Git管理，且不覆盖用户root代码。

下一步在此固定空间预留表示上复用预先声明的关系/结构规则，不以目标成绩选参；再建立完全匹配的loss控制与IMP模块消融。未知容量头、完整70轮对照及三个数据集各一个任务均未完成。

### Stage 4e：空间预留表示上的同规则结构诊断 — 已完成

exec25执行 `run_rta_space_structure_colab.py`，复用原先声明的阈值倍数0.5/1/2、κ=0/5/20、steps5、容量100；仅读取固定epoch4 features，未读目标标签，不根据目标成绩改变规则。
新的source类内平方半径99%分位为0.94909530878067，残差各维方差0.0010601026006043。半倍阈值三次均容量饱和；原阈值三次均16个候选，其中14个质量≥5，候选软概率均值约0.42/0.44/0.48；两倍阈值三次均0候选。成功trial全部有限且归属行和为1。
因此，在包含原空间预留训练的表示上，阈值敏感性仍存在；16也不能直接解释为未知语义类数或配置未知槽。不能把7→16归因于virtual CE，因为两个表示的训练流程/轮数等并非单变量对照。
完整归档 `pipeline-results/rta-space-structure-v1.zip` 已下载、核对SHA256 `7f927a3af23b96bc72bce7a7ac58b3bd0d159a34eed3cbca08bf76c8f0009c07` 并解压；含全部失败/成功trial、软归属、有效质量、原型和推断代码快照。

Git：诊断及隔离预热代码已在module_imp本地提交 `d081f47e`，用户root六个已修改训练文件未进入该提交。两次origin/module_imp推送均网络失败（connection reset/443无法连接），不宣称GitHub同步成功；没有force push。最新诊断记录及runner另提交版本。运行产物保存在本地pipeline-results，不进入Git。

### Stage 4f：关系约束原型模块 — 已实现并验证

`SourceAnchoredIMP.fit` 新增可选 `known_compatibility`，默认None保持旧行为。固定关系兼容度p来自source-soft-prototype KL的BGMM最低均值组概率，是plug-in信号，不是校准的真实已知概率。
候选出生同时要求几何距离超阈值和(1-p)≥0.5；软分配对已知每组件加log(p/C)，候选每组件加log((1-p)/M)。这样两个组的总先验权重不随组件个数自行增加。p=0/1由零组权重的负无穷logit表示；有候选时仍至少有一个有限组。无候选时只能在已知组归一化，返回 `compatibility_ignored_without_candidates=True` 明确该局限。
source已知中心仍用κ=5的条件Gaussian MAP更新。该模型没有推断DP stick权重、浓度或未知语义类别后验，不能称为原IMP/DPMM精确实现。

exec26四项合成检查通过：不传关系约束时与旧模块逐元素相同；给定完美关系的已知偏移案例不误生候选、真实新结构生一个候选；全已知无NaN；非法gate拒绝。完美gate是人工测试输入，不证明真实关系模型准确。
exec27在固定RTA空间预留特征上对照未约束/三次BGMM初始化×原序/倒序，共8项；threshold=原source99%分位、κ=5、steps5、容量100固定，没有目标标签输入。三次BGMM均收敛，出生合格样本335/334/335。未约束为16/19候选；三种gate均仍16/19。关系约束改变软归属（约0.51/0.53候选概率均值），没有修复数量顺序敏感性，不据此宣称改善未知发现或OSDA。
完整归档 `pipeline-results/relation-constrained-frozen-v1.zip` SHA256 `90f2a22614589f432c3df954bc3f92d79b99e75c0d8d404581aa5ffb0c8b7a96` 已下载、核验、解压，包含unit报告及固定模块快照（SHA256 `ec808408c570051a4cad50dbb5dc868f009fe87746b3c7c07ffcff29f83c7dd6`）。

### Stage 4g：最远优先出生策略 — 消除换序误差，但数量仍不确定

新增可选 `birth_strategy='farthest'`。先按特征坐标和兼容度规范化内部行序，每轮优先取距离现有中心最远且出生合格的点，直到覆盖阈值或显式容量报错。还原返回soft assignment的输入行对应。默认sequential保留旧算法。
这是确定性的几何覆盖策略，不是原IMP的Bayesian后验或语义类数估计。规范排序也规范了MAP的浮点归约；消除计算顺序差异不能证明阈值/数据采样选择正确。
exec28默认回归、已知身份/行对应及五次合成打乱精确相同均通过。exec29共24项真实检查（无gate/三种gate，各原序/倒序/打乱及三次80%子采样）。完整数据全部20个候选，每组内三种顺序的soft assignment逐元素相同。无gate有17个质量≥5的候选，带gate三组均18个；80%子采样候选18/16/13，带gate支持≥5的候选15/15/12。子采样继承全数据关系gate，不是重拟合整个模型的稳定性评价。
24项归属有限且行和为1。完整归档 `pipeline-results/farthest-constrained-frozen-v1.zip` SHA256 `90e488373967a463bd2ca2cfb707ceb7afbf38c0f72cc2b79385d6fdd2d1feca` 已下载、核验、解压；模块SHA256 `ec38a017ae2b91755499e136d309fc2af10eff802aa7a114b12fa2cd51943a4f`。

### Stage 5a：候选容量头功能验证 — 已通过，尚未适应训练

新增 `candidate_head.py`，在创建优化器前按候选有效质量阈值5选出方向，保留已知fc权重，候选方向归一化并匹配已知权重平均范数。更新原fc对象的Parameter/out_features，保留CLS.main中的共享引用。空支持/零方向等输入显式失败，不偷偷回退到真实类数或固定2。
质量≥5此前仅作为诊断；本次把它声明为**试验用潜在容量提案**，不是最终选择规则。完整gate-seed1为预先固定primary，不按目标标签选最佳seed；其18个支持候选不代表18个未知语义类。子采样不确定性已明确保留，不把该候选容量当作已验证数量推断。
exec30比较max_slots=2与不截断候选提案，在固定epoch4原RTA网络上做source真实4样本前/反向检查，不执行optimizer.step或比较目标准确率。两组分别输出[4,12]与[4,28] logits，已知权重初始化保持逐元素不变，原fc/main共享引用保持一致，新优化器确实持有新Parameter。两组loss及所有梯度有限，未知权重梯度范数约0.815/0.867。单batch loss约2.763/3.433，仅作sanity日志，不能用作性能对比。
完整归档 `pipeline-results/candidate-head-smoke-v1.zip` SHA256 `1c721b3790321e5805e73e8df855578fa4a666f41d99150a9c4ec8b22a65323c` 已下载、核验、解压；helper SHA256 `dd64dea98138b29f88ffcccf520d4a45c92983d0b22385595ab8b5dcd9cce601`。容量提案文件哈希 `5bfd8dd4e51d0f7a818fa240b526ed942c3f61ebb63fca48f3bc4a4145aff9c2`。
任何完整控制实验都必须同样重建/迁移优化器状态；当前helper替换Parameter会使旧优化器引用失效，不能直接在已有optimizer上调用然后继续训练。下一阶段是同checkpoint、同预算的2槽/数据提案容量短程适应pilot，另保留原RTA初始化对照，避免把容量与初始化方法的联合变化误归因于容量。此pilot只检验潜在容量假设，不宣称恢复真实未知类数。正式三数据集各一任务对照仍未完成。

### Stage 5b：同预热 checkpoint 的短程容量对照 — 已完成

2026-10-03，通过 colab-cli 在 `gpu-l4-s-kkb-ass1a1-13n8bs33o8id1` 的 L4 上完成。
隔离代码 `experiments/rta_capacity_pilot_v1`，未编辑用户root训练文件。
exec31解包，代码包SHA256 `e27ed37fb53a8b23f15ed81019e8d1b19894fb54760aa04fe2e9064579eb5dd9`。
exec32三组状态交接通过，exec33/34/35分别执行original2/proto2/proto18；均exit0。

预先固定seed1、warm epoch4 checkpoint、2个完整adaptation epoch（循环索引4/5，保存epoch5/6）、batch64、RTA原损失及学习率。
原两槽head保留warm权重；候选两槽按有效质量选top2，十八槽保留全部质量≥5候选。
不按目标标签选容量、seed、checkpoint或延长表现较好的组。

**不是精确无缝续训**：warm checkpoint没有source soft bank、GMM、virtual templates、RNG、scheduler/GRL计数。
三组统一用固定center-crop特征重建soft bank、BGMM4（random_state1、max_iter800）及FAISS20虚拟模板。
恢复feature/discriminator SGD；分类头已知行及其他参数动量保留，所有组未知行动量统一清零；scheduler step56、GRL step112，初始化后统一reset seed1。
已知权重/动量、bank、GMM、virtual模板的交接哈希或数值逐组完全一致。
因此original2是**受控交接后的原头控制**，不是已验证的完整官方baseline续训。
增加有限loss/gradient检查，隔离OptimizerManager遇异常不step；两轮内没有触发异常。

以下均为固定final epoch6百分比，不选best seed：

| 实验组 | OS* | UNK | HOS | 相对original2 HOS（百分点） |
| --- | --- | --- | --- | --- |
| 原RTA两槽初始化 | 85.99 | 76.73 | 81.10 | — |
| 候选原型两槽 | 86.47 | 69.37 | 76.98 | -4.12 |
| 候选原型十八槽 | 89.78 | 78.90 | 83.99 | +2.89 |

本pilot三个组的目标标签oracle-best恰为最后一轮，与final数值相同；这个巧合不使oracle选择成为无标签方法。
单seed、仅两个适应epoch，无均值/标准差或显著性结论，不与论文70轮结果直接比较。
十八槽相对候选两槽高7.01 HOS百分点，但不能据此证明真实未知类数为18、IMP理论正确或最终训练更好。
候选两槽低于原两槽也说明初始化效应不能省略。

exec36完成固定final无目标标签的槽占用诊断：

| 组 | 硬预测未知样本数（总564） | 有硬归属未知槽数 | 平均未知组soft概率 |
| --- | --- | --- | --- |
| original2 | 244 | 2 | 0.3108 |
| proto2 | 225 | 2 | 0.2915 |
| proto18 | 229 | 18 | 0.4497 |

十八槽每槽硬归属5–23样本，没有完全闲置槽；这不证明语义纯度、跨采样稳定性或应有18类。
未知组soft质量更大但硬未知预测少于原头，不能简单归因于“预测更多未知”。
新增槽改变softmax分母/组总先验，是潜在混杂；下一合理机制对照是保持候选方向及训练预算，检查组容量先验归一化/关系先验，不能根据HOS调一个最优偏置。
候选支持阈值、几何半径、子采样13–18的不确定性仍待解决；现在仍不是完整DPMM后验推断。

exec37结果归档 `pipeline-results/rta-capacity-pilot-v1-results.zip` 已下载、核验、解压，SHA256
`3cab5cf49a10765557c49828b3b21fce65b960d2b7029697e1be0c945c5e291c`。
包含三组逐轮指标、配置、状态审计、console、固定final target logits（无标签）、占用报告和代码快照。
三组checkpoint另由exec38分开归档（每个约211MB），因Drive挂载/API不可用使用直接FS传输；三组均已下载到pipeline-results。归档SHA256及内部last.pt SHA256均本地核验匹配，未只依赖传输成功提示：

| 组 | checkpoint归档 SHA256 |
| --- | --- |
| original2 | `593b758a8a0e35e0d7398f680e082834f19f6f6418579c30ee1d419d77fa15a5` |
| proto2 | `b55825410614ccb452cec78531ddb9786a6508c443c1eb8cac985cdbc8858c6e` |
| proto18 | `27c89949e1b87d5fc37e67cd26c1a25583874021c912deae5b7cb1cfaf5c6b08` |

实现及pilot记录已本地提交 `d5c3cce3`。本轮origin/module_imp push仍失败（Connection was reset），不宣称GitHub已更新；用户root修改未纳入提交。

正式目标仍包括Office31 A→W、OfficeHome Pr→Rw、VisDA Synthetic→Real，各一任务，同协议baseline/IMP完整训练，seed1/2/3全部报告。尚未完成三任务，不能用此pilot替代。
没有重启或销毁runtime，没有恢复已暂停的baseline自动检查。研究目标仍未完成。

### Stage 5c：容量组先验机制对照 — 已完成，负结果保留

固定上一stage的候选十八槽、warm epoch4 checkpoint及两epoch预算、seed1。
新增一个uniform-reference2组：已知logit不变，每个未知logit加 `log(2/K)`，K=18时为`-log(9)`。
reference2来自原控制头槽数，不是目标标签调出的阈值；没有搜索偏置强度或重选候选。
假设检验：复制同一未知logit不应该凭槽数量改变未知组总概率；不假设此控制必然提升OSDA。
先做K=2零修正、重复槽组概率/已知概率及梯度不变、极端logit有限等检查，再启动训练。
**局限预先声明**：此修正仅控制group softmax先验总量；flat entropy和伪标签CE仍随槽拆分变化，最大单槽argmax也不具有组重复不变性。
这不是完整DPMM，也不是已证明正确的未知判别规则。不能把这个单变量机制控制当最终模型；保留无修正结果，无论其性能高低。

同L4执行exec45解包（代码包SHA256 `bba3a5861e0541b4102db61f442162852bb39e955cd336fa33c7f43e5e8f261b`），exec46代数测试通过。
K=2零修正逐元素不变；两未知槽各复制9次后，修正保持已知概率、未知组概率及组概率梯度不变；极端logit梯度有限，4个非法计数输入被拒绝。
同时实证验证flat entropy仍增加 `P(U)*log(9)`；因此这不是整个训练目标的槽重复不变修复。

exec47首次notebook预检在checkpoint prior持久化断言处失败，**没有执行训练或optimizer.step**。
exec48核实Python缓存实际来自旧`rta-l4-control-v1/networks.py`和旧pilot_state：旧CLS没有注册buffer，虽然临时赋予prior属性，state_dict不保存它。
没有放松断言、重启runtime或调参。改用独立Python worker并断言模块路径；exec49实际新版模型的同状态交接、真实source前反向、prior buffer持久化及重载logit精确相同均通过，未step。
exec50只释放失败预检临时对象。此后训练与收集均在独立子进程执行。
此前Stage5b训练本来就是独立子进程，不因本轮notebook验证缓存问题而自动重跑。

exec51执行唯一预先声明的uniform18两epoch适应，exit0；无非有限loss/gradient，不自动延长或偏置搜索。exec52固定final收集成功。

| 容量18设置 | 固定final OS* (%) | UNK (%) | HOS (%) | 硬预测未知数 | 有硬归属未知槽 | 平均未知组soft概率 |
| --- | --- | --- | --- | --- | --- | --- |
| 原始候选头，无先验修正 | 89.78 | 78.90 | 83.99 | 229 | 18 | 0.4497 |
| uniform-reference2修正 | 95.20 | 37.30 | 53.60 | 95 | 13 | 0.2098 |

修正组epoch5/6 HOS为39.76/53.60%，oracle-best为最后一轮，仍只把固定final作为主比较。
最终HOS下降30.38百分点；source CE约0.225、unknown伪标签CE约2.307（未修正组约0.397/1.418）。
解释限于本次短程机制对照：直接把组质量归一化放进原RTA平坦分类/最大单槽决策，会更难预测未知，不能作为当前最终方案。
不能由此推断所有Bayesian先验无效、原始18槽正确或13个占用槽就是13个语义类。

**下一阶段原则**：将“是否未知”的组级判别与“未知内部哪个原型”的条件分配分开；在先验加权原型混合中边缘化未知组件，不能把潜在槽直接等同监督语义类别。
先验证拆分/重复组件后的组级输出与组级loss不变，再做同checkpoint、同预算对照。
保留RTA关系判别/空间预留控制，逐项区分判别、原型结构loss和先验权重；不通过扫目标HOS补偿失败偏置。
正式三个数据集各一任务及多seed完整对照仍是最终要求，此单seed pilot不代替。

结果和checkpoint已生成独立归档：results SHA256 `a63b92d07cf0b5454c90167535d359c5619fce261e6f7b1dd0d4dcf01820e3fb`；checkpoint archive SHA256 `e8326b0f697659a7d5daa8be16cbb8ad57c6190b14150875fe5a23e0dc6f4c9c`。
checkpoint内部last.pt SHA256 `e96827ab4d55eaa8d997379a78c929a52709f61d31af09847c920e033ec205c2`。
两个归档均已下载到pipeline-results，结果归档SHA256已校验并解压；checkpoint归档SHA256及内部last.pt SHA256也已本地核验匹配。
结果包包含失败缓存诊断、独立预检日志、worker源文件、模型快照、逐轮指标、配置和无标签final logits。

### Stage 5d：未知组件边缘化的分层头 — 已完成

保留raw已知/未知组件logits，把未知组件作为潜变量，而不是K个监督语义类别。
组未知证据 `logsumexp(z_U + log(pi)) + log(2)`，本stage固定pi=1/K，已知logit不变；条件原型分配为softmax(z_U+log(pi))。
已知10类+一个未知的C+1输出用于source CE、未知伪标签CE、target entropy、virtual CE和最终预测。
RTA关系KL/GMM、样本选择、adv、virtual模板及原损失系数保留；未知CE不再强迫单个原型赢得全部概率。
reference2固定来自原头总槽先验，仍非校准未知比例。没有扫目标标签阈值。

固定seed1/warm epoch4/两个完整adaptation epoch，三组original2/proto2/proto18，初始化、优化器状态及初始关系bank等按Stage5b同规则交接。
这同时改变了平坦损失/决策语义，不能把相对Stage5b的差异只归因于容量。组内结构本阶段由候选初始化与边缘似然梯度学习，尚不增加独立结构loss或更新DP浓度/组件数。
先验证分裂一个组件并分配其权重后的组输出、组级loss和共享logit梯度不变；不宣称独立组件SGD轨迹不变。
没有latent语义标签监督，原型可能塌缩，必须记录条件占用及中心方向相似性。真正语义类数和数量不确定性仍未解决。

exec55解包，代码包SHA256 `e8fd2b1a8c335cb4adf02a7f278fb7e2c220a7765dd00f0a2fb35a1e44f4657f`；exec56代数测试通过。
均匀/非均匀prior分裂组件后的semantic logits、预测、source CE、unknown组CE、semantic entropy、virtual CE及共享logit梯度相同，极端logit有限。
exec57在独立worker中检查三组真实图像的同状态交接、全损失前反向、mixture buffer持久化及重载semantic logits精确相同，未step。

实际运行顺序为exec58 proto18、exec59 original2、exec60 proto2，各完成固定epoch5/6，exit0。
首次入口继承notebook遗留PILOT_ARM=proto18，所以并非最初口头描述的先跑original2。实际proto18配置/预算均符合预声明，审计与console一致，没有重跑或覆盖。
后续入口取消隐式default，并由三个wrapper显式指定实验组。此运行顺序变动不改每组checkpoint、seed、训练预算，也不按成绩选组。

| 分层初始化 | 固定final OS* (%) | UNK (%) | HOS (%) | 硬预测未知样本数（564中） | 有硬归属latent槽 |
| --- | --- | --- | --- | --- | --- |
| original2 | 86.93 | 75.75 | 80.95 | 236 | 2 |
| proto2 | 87.32 | 74.96 | 80.67 | 237 | 2 |
| proto18 | 91.62 | 65.52 | 76.40 | 188 | 18 |

三组oracle-best恰为final，仅作诊断，不因此改变固定final选择规则。
分层original2相对Stage5b原平坦original2 HOS低0.14百分点；分层proto18低于分层proto2约4.27百分点，低于原平坦proto18约7.59百分点。
没有多seed方差或完整训练证据，不能认定分层方法普遍无效，也不能把组级一致性通过当作性能提升。

exec61固定final诊断/归档：proto18每槽1–25个硬归属未知预测样本，没有空槽；条件soft质量和输出组概率均有限。
未知组soft概率均值0.2063，方向两两cosine均值0.49794（初始化0.51007），最大0.86714（初始化0.86677）。
初始/末轮同槽方向cosine均值0.99930，尚未显示明显方向坍缩，但两轮内原型方向几乎未变。
方向相似度/占用不是语义纯度检验，既不证明18类，也不证明结构学习充分。原采集字段occupancy.semantic_unknown_count是未知**样本数**，不是语义类数。
本阶段只用边缘似然更新latent heads，**没有独立条件结构loss、数据驱动mixture权重或完整DP后验更新**，这些仍待分项验证。

### Stage 5e：同checkpoint的未知决策规则离线审计 — 已完成

exec66复用Stage5c uniform18固定final logits，原始logits已经带log(2/K)，没有再次加先验。
固定比较原最大单槽后collapse与已知10+unknown logsumexp的边缘化决策；无训练、无偏置扫描、无checkpoint选择。
使用单独evaluation-only标签仅报告每类宏平均；首先复算原指标与Stage5c逐项精确匹配。

| 同一个uniform18 checkpoint的决策 | OS* (%) | UNK (%) | HOS (%) | 硬预测未知样本数 |
| --- | --- | --- | --- | --- |
| 最大单槽后collapse | 95.20 | 37.30 | 53.60 | 95 |
| 未知组证据边缘化 | 92.23 | 65.36 | 76.50 | 185 |

仅改变决策，就有90个样本改判为未知，HOS提高22.90百分点；这是明确隔离决策影响的证据，不是新训练成果。
而Stage5d分层proto18训练后的HOS76.40，并未比这次固定checkpoint决策审计表现更高。
不能因目标评价更高就偷偷把Stage5c的旧结果改写为分层训练结果；两个规则全部保留。

下一步先区分已修复的输出语义与尚未学习充分的原型结构/先验；不再通过尝试不同推断阈值挑目标HOS。
在加入新loss前核实其源先验、无标签责任分配和条件结构约束，并冻结方案后开展完整预算对照，避免长期停留在两epoch pilot。
最终三数据集各一任务及seed1/2/3范围不变，当前仍未完成。

### 三任务协议补充审计（2026-10-03）

详见 `RTA_PROTOCOL_AUDIT.md`。已核验 Pr→Rw 官方列表在本地可用：源1785张/25类，目标4357张/65类；相对 `data/` 所有图像路径存在，已知类别名称一致。尚未完成 Colab 传输与通用训练入口。
同时确认发布代码的损失系数、熵输出空间、加权归一化、KL方向、虚拟模板数与论文公式有差异。既有结果标注为发布代码口径；论文公式忠实实现必须另立基线，不能静默替换。VisDA backbone 仍需澄清。此次没有启动新训练。

新增独立 `task_protocol.py`，提供 Office31/OfficeHome 明确类别集合、VisDA 经外部原始ID映射构造的协议、严格路径/类别/重复样本检查，以及保留原始未知类别的宏平均评价函数。`scripts/test_task_protocol.py` 的 VisDA 映射仅为单元测试 fixture，不是已下载/核实的真实 VisDA 文件列表。
本地五项测试及真实 Pr→Rw 输入审计通过；各类样本数和列表SHA已输出核验。未知类别预测虽合并成语义unknown，UNK仍按原始未知类别准确率平均，不能用合并样本准确率替代。该模块尚未接入训练入口，不宣称已经实现三任务训练。

### 多任务 baseline 入口 v1（2026-10-03，尚未训练）

新增独立快照 `experiments/rta_multitask_baseline_v1`，复制已冻结 L4 published-code control 后接入任务协议，不修改根目录用户训练代码或旧实验快照。源标签按已知集合映射；target_train 标签替换为常量unknown sentinel（原记录lt不被loss/聚类使用）；target_test 保留原始标签用于逐类宏平均。损失系数、预处理、优化器、预热与mixture schedule不变。
新增任务专属日志目录且拒绝覆盖；virtual cluster数必须显式声明，不能把Office31原20静默套到OfficeHome。VisDA要求完整class-map和显式接受尚未解决的ResNet比较口径；未实现VGG且拒绝假VGG开关。
七个Python文件语法检查通过；`scripts/test_multitask_entry.py` 三项测试通过，直接抽取实际入口transform执行（mock图像预处理），验证两Office任务source映射、目标训练label恒定、目标评价raw label保留，以及原Office31宏平均公式一致。它不是实际图像batch/GPU前反向检查；尚未运行训练、传输OfficeHome或验证VisDA真实列表。下一步真实batch preflight通过后才能启动新训练。

### 多任务入口实际 A→W batch 预检（2026-10-03）

colab-cli exec69在现有L4端点完成，状态done。独立worker执行实际main.py中的参数/文件审计/三个DataLoader定义，停在模型和聚类初始化之前；再用该快照模型与真实source/target各64张224×224图像前反向。入口SHA256 `89ae1377f4d852b96eb2fe6f2d2786928dbe2a8fde0e4d015cf86701c8a755f4` 与本地一致。
损失7.816242218，所有梯度有限，峰值已分配GPU显存11,307,886,592字节（约10.53GiB），optimizer step=0。target_train label恒为unknown sentinel，target_test保留原始31类one-hot标签。
这是损失连接预检：virtual使用固定随机方向、gate使用全1权重、unknown selection使用前16样本；没有验证完整GMM、虚拟聚类刷新、真实未知选样、整轮训练与长期显存。不将它写成完整训练通过；Pr→Rw和VisDA尚未做真实batch检查。
报告及console已下载至 `pipeline-results/multitask-preflight-v1.json` 和 `multitask-preflight-console-v1.log`；报告SHA256 `64f8e338b96a3985566d745fb207f37f0fd8010f5047dd8152c83b19dc23231d`。未重启runtime、未修改既有模型checkpoint。

### Pr→Rw 数据打包与传输（2026-10-03，进行中）

新增 `scripts/pack-officehome-task.ps1`，仅按两个任务列表打包，并生成6142图片+2列表的逐文件SHA/大小manifest。源包43,309,310字节，SHA `fef471edd5535d0ccdcf4d3d3f6809bd44388eb4445cdea9c70c460e88c8c76b`；目标包757,071,899字节，SHA `04088dfeeee9da261439f609f0257cf24c890a1e32c87c2b5c15ae97fdfb5e35`。本地路径 `artifacts/officehome-pr2rw-v1`，不纳入Git。
Drive API列表查询间歇成功/请求配额失败。成功创建 `OSDA/datasets/officehome-pr2rw-v1` 文件夹ID `10Q6tfMPLwtZB1l1xJU536sqJErIVWgbo`，但目标包 resumable upload 初始化403：`storageQuotaExceeded`，用户Drive存储已满；未上传目标包，未删除任何Drive文件。
源包已通过fs直传现有L4 `/content/product_0-24_train_all.zip`；运行时字节数/整包SHA核验与本地一致。目标包超过500MiB，下一步需改为受校验的分片直传再重组（Drive不可用的回退），或者用户自行释放Drive空间。新增 `scripts/verify_officehome_colab.py` 为逐条归档/解压/图片验证准备，尚未执行。尚未完成目标传输、数据解压或Pr→Rw GPU预检。

### Pr→Rw 数据已就绪、实际 batch 预检通过（2026-10-03）

Drive storageQuotaExceeded回退：目标包分成250MiB/250MiB/232,783,899字节三片，通过fs传到原L4；没有删除用户Drive文件。exec75逐片校验并重组，完整目标包SHA与本地 `04088dfeeee9da261439f609f0257cf24c890a1e32c87c2b5c15ae97fdfb5e35` 一致。重组脚本两项本地测试通过（损坏片阻止生成、拒绝覆盖）。
exec76独立worker验证两个归档整包SHA、全部6144条目（6142图片+2列表）的归档内及解压后SHA/大小，并对全部图片执行verify和像素load；状态done。数据解压于 `/content/osda-officehome-pr2rw-v1`，这是临时运行时存储，不是已持久化到Drive。
exec77实际入口DataLoader+模型前反向预检，source/target各64张224×224真实图片，分类输出29（25known+4unknown槽），target_train标签恒定、评价保留65原始类别。损失10.302968979，所有梯度有限，峰值已分配GPU 11,307,958,272字节（约10.53GiB），优化器step=0，状态done。
仍只是损失连接测试：virtual为随机10方向、gate=1、unknown选前16；传入virtual-clusters=35仅为了预检入口参数，未运行Kmeans，未把35定为正式OfficeHome配置。完整聚类刷新/选样/训练尚未验证，VisDA仍待真实列表与backbone选择。
重组、输入校验、预检报告及两个console已下载到 `pipeline-results/officehome-*`。入口SHA与本地 `89ae1377f4d852b96eb2fe6f2d2786928dbe2a8fde0e4d015cf86701c8a755f4` 一致。本地完整数据包与逐文件manifest保留于 `artifacts/officehome-pr2rw-v1`；没有删除分片、旧实验或checkpoint。

### L4 完整预算 published-code baseline：A→W seed1（进行中）

colab-cli exec78，端点 `gpu-l4-s-kkb-ass1a1-13n8bs33o8id1`；独立worker仅运行本组，不串联实验矩阵。目录 `/content/imp-runs/rta-multitask-baseline-v1/office31-a2w_seed1`，seed1、70完整epoch、batch64、未知槽2、virtual Kmeans20，发布代码损失口径不变。没有IMP、没有调参、不是论文公式忠实重实现。
启动前独立快照增加有限性检查（KL在原inf→10处理后检查NaN，loss/grad检查）及异常时OptimizerManager不step；两项fail-stop单测通过。正常数值时公式与更新不变。main SHA `683c35e76a89ad1588f35dea83e3ecbc6f9a8af5bb3250e9e3de1cbfc3c7a148`，utilities SHA `6ba276bb2d0076a4d64cde811496d2dfb71b330c8e1660add02d8fbd42e636e7`。
checkpoint扩充实际source/target关系bank、virtual模板、sklearn mixture对象、优化器scheduler/GRL计数和Python/NumPy/Torch/CUDA RNG，固定第4个预热epoch另存 `warmup-complete.pt`。这为后续准确恢复状态准备，不把此前重建bank的pilot追溯改成精确resume。
已观测完整预热、适应期真实GMM选择与每轮虚拟聚类刷新正常。最近检查日志epoch43（0-based）且exec仍running；中途指标不作最终成绩、不据此选checkpoint。以完整70条history及固定final报告为完成依据。Drive已满，输出目前在临时/content，完成后必须立即下载checkpoint/配置/history/audit。启动manifest已下载至 `pipeline-results/rta-multitask-a2w-seed1-launch-v1.json`。
GitHub上次push返回RPC/curl55失败且同时打印Everything up-to-date，不能据后一句宣称同步成功；尚需核验远端ref。没有重启runtime。

### Stage 6a：有限Dirichlet组件先验与条件结构KL（模块测试完成，未接入训练）

新增独立 `prototype_structure.py`，无target语义标签输入：

- 冻结软责任质量n，有限Dirichlet总浓度α=1、默认均匀base b，posterior mean π=(n+αb)/(sum n+α)。不是每组件α=1，不是DP后验或语义类数推断。
- 用冻结候选中心与source校准方差构造q(k|x,U)=softmax(log π-distance²/(2σ²))；teacher、先验及gate均detach，σ²正数下界1e-8。
- 新结构项为unknown组内部KL(q_teacher || q_student)，与组级unknown证据分开，按unknown权重和归一化。零权重批次返回零loss/gradient；尚未选择训练系数或接入正在运行的baseline。

通过独立shell1的CPU测试（不占用训练kernel队列）：posterior mean正确、只拆分部分组件时同步拆分count/base质量后loss及共享logit梯度不变、teacher/gate无梯度、zero-weight零梯度、zero-variance正数下界。第一次测试脚本因重复使用已释放计算图失败；只修测试图重建后通过，未改模块公式、未调参或重跑训练。报告 `pipeline-results/prototype-structure-unit-v1.json` 已下载。
冻结真实feature/proposal哈希与Stage5固定输入一致；source方差由source类别残差重算；采用既定质量≥5的18个候选、α=1，没有标签或性能挑选。全部564目标的条件teacher有限且归一化，18组件均有硬归属；平均熵0.0131313，最大概率中位数=1，显示teacher几乎硬分配。报告 `pipeline-results/prototype-structure-frozen-v1.json` 已下载。
这是全target几何诊断，不是训练未知选样或语义正确性证明。它提示不能将teacher高置信度当成可靠语义监督，也不能把18组件当18未知类。下一步需核验未知gate下的结构监督及梯度规模，再冻结消融设计；尚无结构模块训练收益结论。三任务、每任务seed1/2/3及完整baseline/IMP对照仍未完成。

Stage5d结果包SHA256 `8ac6452bd6b96c7ddaa16629b3d2721bcc1adf77a2954065dfc3fa1149478d6c` 已下载、校验、解压。
三组checkpoint均已下载，本地归档哈希及内部last.pt哈希全部核验匹配：

| 组 | checkpoint归档SHA256 |
| --- | --- |
| original2 | `10abf749b10e1cd24a87e0adef3f917a83e80f46a5cf7d7500f4c2dda507cf5d` |
| proto2 | `44c680aad7d1972c316a33b319cee4845809e70473595a03a75c26437ee52d64` |
| proto18 | `b90383b060fc5ddde2f3807cee1a1eab4d80fd26dbb285b486110dd3a725d9ae` |

Stage5e审计归档SHA256 `f965d585efe21a52b50f80a56fcc9141a3f71c6e348fc31948a176a094812f87` 已下载、核验、解压；包含两规则结果和完整审计脚本。
