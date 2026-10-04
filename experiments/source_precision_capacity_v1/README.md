# 固定协议：Source precision capacity v1

接口source_precision_capacity.estimate_capacity(source, source_labels, target)，不接受target标签。

冻结规则：各source类按RandomState2026分层70/30，70%建立中心；source类内平方距离99%分位λ；每类κ_c取校准样本数，R取校准每类平均样本数；target残差权重R/N，birth-first，移动已知中心，noise截断λ，整体残差收益须超过λ建簇成本。固定100个候选保护上限，触顶明确失败，不能作为估计K；最多100步，未收敛明确失败。K0原样返回，调用RTA时不能偷偷强制K1。

只把整数K传入K-only训练，保持Q/V、原版伪标签、损失、预测及共享source训练初始化。不能把新候选方向初始化、teacher、marginal预测一起加入仍称K-only。A→W三个seed估计K2，分类训练与相同初始化固定K2在方法输入层面等价，不重复运行来声称性能提升。

该协议是在A→W探索后提出，不作为无偏预声明验证。后续OfficeHome Pr→Rw与VisDA Syn→Real固定规则做外部验证。三数据集整体实验尚未完成；VisDA backbone尚待确认，不伪称论文对齐。当前核心实现缓存N×N距离矩阵，VisDA全量运行前需改为分块或明确声明推断采样，不得默默OOM后减样本。

2026-10-04资源检查：现有L4只保留Office31图像及三个fusion source prior；/content/osda-officehome-pr2rw-v1不存在，未找到OfficeHome source features。需先恢复OfficeHome数据、source C25监督3轮和冻结特征，不能直接复用Office31网络作为OfficeHome已监督prior。

恢复进度：本地artifacts/officehome-pr2rw-v1有43MB source ZIP及757MB target ZIP（三个已有分片）。Drive共享客户端202264815644返回请求配额超限，改用已有分片直接上传，不更改Google客户端设置。source包已上传到/content/product_0-24_train_all.zip；target part000/part001上传中。尚未解压、尚未训练，不能提前宣称数据就绪。

恢复脚本restore_officehome_inputs_colab.py仅合并已上传分片、拒绝路径越界/覆盖、确认列表图片存在及C25/65协议，不做hash/逐图重复审计。source launcher run_precision_officehome_source_colab.py单独训练seed1、C25、3轮，复用已验证source-only代码；只有数据恢复成功后启动，输出/content/imp-runs/source-precision-officehome-v1/seed1/source。后续固定K与估计K共享该prior，RTA尚未启动。

最新状态：三个target分片均上传成功，恢复脚本已正常结束，restoration.json确认source 1785张/C25、target 4357张/65类。已在既有L4端点gpu-l4-s-kkb-ass1a1-2xg3a949wz74o启动source监督3轮（seed1，后台exec63），输出/content/imp-runs/source-precision-officehome-v1/seed1/source。尚未完成source特征提取、尚未估计OfficeHome K，RTA对照尚未启动。target标签不进入source训练或K估计；65类仅用于确认任务列表口径。

随后exec63已正常完成（returncode0），三个source epoch loss为2.895804/1.905485/1.342651。冻结特征已保存；独立启动固定规则推断exec65，脚本infer_precision_officehome_colab.py仅使用source、source_labels、target特征，结果保存seed1/capacity.json。未根据此任务调规则，尚未启动RTA对照。

外部验证结果：exec65正常结束并收敛（7步），OfficeHome seed1估计K=0；λ=0.5512014144，R=49.44，target4357张中noise116（2.66%），其余4241张被25个已知中心吸收。结果pipeline-results/source-precision-officehome-v1-capacity.json。该结果否定目前规则可直接稳定跨任务使用的假设，不强制K1，也不在K0下启动需要未知输出槽的RTA。下一步在无target标签的冻结几何中区分阈值覆盖过宽与整体收益/建簇成本问题；不把真实40个未知语义数作为调参目标。

无标签诊断exec67完成：initial已知中心未移动时，最佳建簇gain=-0.425504（λ=0.551201），支持45个样本；final gain=-0.428356，支持44个。当前w=R/N=0.011347，单点收益最多wλ，因此任何建簇必须支持n>N/R=88.127个样本（必要条件而非充分条件）。最佳initial收益仅0.125697，是成本的22.80%。初始化target到最近已知中心距离/λ分位数50/90/95/99%=0.6976/0.8967/0.9604/1.0915；全局阈值外3.05%，改为描述性的“所有source类半径之外”仅1.61%。source holdout自身类距离全局阈值外1.64%，类自身半径外4.37%。这些不证明target语义已知，只说明当前source特征几何与惩罚不支持新簇；主要失败在初始状态已有，不能归罪于中心移动。

下一版设计约束：将“已知类包络/异常截断尺度”和“未知簇复杂度成本”分离，目前二者同用λ且R/N权重使支持下限被N/R决定。重新推导容量成本后，首先用source-only已知负对照与留类正对照检验，再冻结规则做外部任务；不能以OfficeHome真实未知类数或target HOS选成本，不能在本失败后宣称此前预声明外部验证成功。诊断输出pipeline-results/source-precision-officehome-v1-geometry.json，脚本不读取target标签、不修改v1设置。
