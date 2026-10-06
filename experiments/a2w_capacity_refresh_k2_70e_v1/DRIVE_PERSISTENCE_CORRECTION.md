# Drive模型持久保存更正与本地备份（2026-10-06）

## 已确认问题

CPU独立Drive挂载看得到本批refresh目录、日志/JSON/边界缓存，但该目录没有best.pt/last.pt。T4挂载中的存在检查不是云端上传完成证明。

T4 DriveFS日志明确返回：`The user has exceeded their Drive storage quota`、`QUOTA_EXCEEDED`，并反复记录quota超出后跳过resumable upload。2026-10-06T14:10:54Z仍有该错误。此为用户Drive存储配额，不是先前CLI共享OAuth client的API每分钟请求限额。

因此更正此前“普通模型已实际保存Drive”的表述：模型复制到挂载目录返回成功，但云端大模型持久保存尚未完成。普通训练及70轮日志结果没有因此失败；原始模型仍在T4 `/content/imp-runs/a2w-capacity-refresh-k2-70e-v1`。不把mounted文件存在包装成云端保存完成。

## 备份与保留

本次四文件源大小（仅传输字节长度，不hash）：fixed2 best104726336、last211493614；refresh best104729408、last211499758字节，合计约632.45MB。需一次本地备份，不属于例行重复传输/审计。

colab fs大文件直传四次均fetch failed。改用colab-cli的仅本机127.0.0.1端口转发：local18765→T4 remote8765，只读HTTP服务目录限定为本批模型运行目录，不公开上传到第三方。服务PID86827，forward ID1；备份结束后只清理自己创建的服务/forward，不重启或销毁runtime。

本地输出目录：`pipeline-results/a2w-capacity-refresh-k2-70e-v1-models`。当前refresh-last.pt已完整下载，curl HTTP200、211499758字节；其余三个仍在传输，`.part`不当已完成备份。活跃本地download sessions：fixed2-last76919、fixed2-best60628、refresh-best60392。最终应根据实际传输exit0及文件长度标记，不做SHA/CRC、checkpoint重复评分。

保持T4和CPU，不删除云盘旧数据，不改模型、重训、重启或静默腾空间。已异步请用户自行腾出至少约1GB或扩容；实际积压上传是否还需更多空间必须看错误状态，不能提前保证。

## 对当前研究的影响

新增槽初始化的CPU压力测试读取大模型失败，发生在计算前，不是算法负面结果。已从T4原始完整last.pt提取只需约20KB的fc/BN状态，避免向CPU搬整模型；但初始化检查暂缓，先保护训练产物。

模型本地备份与云端持久保存恢复之前，不宣称本批产物完全保护。日志/统计/图已在本地与可见的Drive目录，性能报告数值不改。总体研究仍继续，外部Drive容量选择不能代用户删除或购买。

## 后续恢复进展

CPU独立挂载已看到fixed2/best.pt（104726336字节）与last.pt（211493614字节），refresh两模型仍未看到，不能称全部云端保存成功。

旧HTTP三个下载过慢，已只终止本地curl客户端PID5520/5562/5565，保留所有.part。新增只读allowlist Range服务PID89372，local18766→remote8766，forward ID2；1MiB试传HTTP206、1048576字节成功。续传handle为fixed2-last22794、fixed2-best87073、refresh-best78224（600秒超时，观察超时不等于进程终止）。refresh-last完整本地副本保持不动。完成后只清理两个自建服务与forward，不重启实例。

用户要求清理Drive，已只读盘点并提出清理旧20e capacity与旧10e identity实验的模型（合计约1.26GB），保留日志、数据包、baseline及本次70轮模型。具体永久删除范围等待用户确认，尚未删除任何文件；一般继续研究授权不代替该确认。

600秒下载已实际超时；curl内部retry可能回退part进度，故停掉唯一仍retry的本地curl PID5873，改为不带--retry的新独立续传。当前handle：fixed2-last58484、fixed2-best27836、refresh-best9948，每次上限1800秒；必须看exit与实际长度，不能按计划判完成。若实际失败，只根据现存part长度重新续传，不覆盖重下，不重复停止runtime。旧handle22794/78224已失败，87073客户端已终止。

初始化缓存诊断已成功完成，与备份并行，结果记录于experiments/a2w_new_slot_init_proxy_v1；source保护guard失败，未修改权重、不启动长训练。

## 最终本地备份状态（已完成）

三个无内部retry续传均exit0/HTTP206：fixed2-last补166050160字节19.33秒、fixed2-best补78753600字节8.84秒、refresh-best补79342400字节8.68秒。最终四个文件长度均与上述源文件长度一致，已以Move-Item去掉.part后缀，四模型本地备份完整，不做hash或重复评分。

只读服务器PID86827/89372已通过命令身份检查后SIGTERM；forward ID1/2均已关闭。T4/CPU原件与实例/挂载未关闭或删除。refresh两模型云端仍未确认，不能将本地备份成功混同云端保存成功。Drive旧实验模型删除仍待具体范围确认。
