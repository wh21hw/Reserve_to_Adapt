# 复杂度合并：目标下降，但仍未保护已知

T4/ResNet50/source3/frozenencoderBN/seed3，同完整warm4各续训6(epoch5–10)。唯一因素为candidate-candidate objective merge；两组都承接当前成员，正常BN/原损失/原未知选择/预测相同。

| 模式 | final OS* | final UNK | final HOS | post4best epoch / HOS |
| --- | --- | --- | --- | --- |
| 不合并 | 96.5456 | 49.6126 | 65.5438 | 5 / 76.1255 |
| 代价下降才合并 | 95.2511 | 52.1034 | 67.3601 | 5 / 74.6532 |

HOS+1.8163pp但known−1.2946pp，guard失败，不采用。相比已完成source_only参考97.6774/54.9944/70.3694，三项均低；不能仅击败失败承接组就宣称有效。两组fullbest均来自共享warm4（86.4022/73.9103/79.6696），不是本改造收益。实际训练454.06秒、collector退出0、ZIP/summary本地保存。

只有3次合并获接受：epoch5两次、epoch7一次，K34→32→31，条件目标总降约0.34503。已知中心/先验项在合并阶段固定；这仅保证固定当前特征的SSE+lambda候选数下降，不保证softIMP/神经训练整个过程单调，不证明语义类别恢复。[DP-means的原始复杂度目标](https://icml.cc/2012/papers/291.pdf)是启发，不将此改造等同完整原论文算法或贝叶斯后验。

独立unknown结构监督覆盖128→140，误known结构标签曝光24→0，但原RTA未知selector仍选中known123→129次；未知语义类21曝光仍0，29只有1。ARI .3519→.3588、NMI .6777→.6752，不能声称结构明显改善。相关真值仅结束注释，不进入K/阈值/训练；结构快照是epoch10开始，不是最后模型。

下一步回到source_only/无merge/正常BN骨架，单独检验“几何可靠IMP成员进入未知CE”能否补足训练入口，不继续扫簇数惩罚，不叠加失败mask/BN/初始化。新实验另批预声明。

seed3事后选择、target-oracle best、额外source预热/encoderBN需披露；不是三seed均值或论文完整复现。OfficeHome/VisDA和整体未知结构目标仍未完成，实例/cache/mount/models保留。
