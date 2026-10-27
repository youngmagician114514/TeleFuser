# TeleWorldServer：按服务指标分池，池内 latent/step 调度

更新：2026-10-27。新入口为 `tools.validation.replay_abot_async_phase_trace`。
原 chunk、FIFO/TurboServe/SlackServe 和旧全局 wave 入口保持原调用方式。

## 执行边界

一个 Job 仍对应 3-latent / 12-frame chunk，S/W 在 Job admission 时固定。
逻辑依赖为 `D1 → … → DS → refresh`。refresh 是 timestep=0 的一次 DiT forward，写入 clean KV，
不属于 VAE。下一 Job 必须等 clean KV 提交并保持 session FIFO。

集群拆成 denoise 与 refresh 两个互斥 GPU 子集。每张 GPU 有独立执行循环，
在本地 ready heads 内按 latent-step 重新选择兼容 batch。不同 Job 的不同 denoise step 可以合批，
不同 W 使用已有的分组 attention，每行保留自己的 timestep、窗口和状态。
一次 invocation 包含一个或多个 Job 的完整 sibling group；默认三个 latent 共置，
避免每层 attention 交换 K/V。没有声称可以独立执行缺少 sibling 的任意单个 latent。

新入口移除了旧 PD 的 world-wave 完成边界。GPU 不再共同驱动 step，
没有逐 step 全局 broadcast、all-reduce、输出 gather 或 barrier。
一个 session 等待 refresh/KV 时，其他 session 继续执行。
只有启动初始化使用 collective，运行时 CPU 元数据走 TCP，GPU 数据走 NCCL P2P。

## 全局层：指标、实测成本和双端显存

启动前使用真实模型测量各 W、各 batch 行数的 denoise / refresh 时间曲线，测量排除在 trace 指标外。
原 profile 仅提供既有 S/W 的质量 Q，不用 chunk latency 替代 step 时间。
在线同 phase、同 W 的完成样本以 EMA 更新曲线；混合 W 的总时间不同时归因给多个 W。

对每个候选分区及配置 c=(S,W)，预测两个池的 Job/s：D 每 Job 包含 S 次 forward，R 包含一次。
供应能力取两池容量较小者。初始分池最大化 `Q(c) × min(1, capacity / declared_request_rate)`，
并要求两池均能保存声明的 resident cohort，不固定为 3+1。
这是容量层面的质量与连续性近似，不能当作最终测得 CPR。

Job admission 联合评估首次 D/R placement 与 S/W：

`T = max(D_queue + D_service + latent_transfer, R_queue) + R_service + KV_return + VAE`

代价沿用 playback-debt 面积与质量损失：

`J = late × (max(0,-slack) + late/2) / output_seconds + λq × (1-Q) × output_seconds`。

对于 action，剩余预算取播放 slack 与 `output_seconds - action_queue_age` 两者较小值，
对应已有的 action 完成期限。已有视频冗余不能掩盖正在老化的 action FIFO；
idle 继续使用播放预算。不改变质量表、质量权重或输入窗口。

双端显存约束按物理驻留页计数，另计逐层 attention 工作区、交接 buffer 和剩余 cohort 的最低窗口容量。
一个匹配 Job 的 clean-KV 完成事件会归还未实际分配或因窗口缩小而释放的预约额度，
避免永久按历史最大 W 记账。扩大窗口仍须在派发前预约完整候选窗口的空间。
完成反馈更新队列成本、分阶段曲线、handoff 和 VAE 成本。

首次 placement 还将每个已有 resident key 的下一项 transition 投射到各 GPU 的实测服务时间线上。
即使 session 在到达时没有 action，也不会把所有空队列都视为没有未来工作。
端到端代价相同时，优先选择本地预计完成更早的 GPU，避免共同 refresh 瓶颈掩盖 denoise 侧的负载差异。
这项 resident-cohort 预测只用于首次 placement；真实 Job 的 batch 容量预测使用已提交的有效工作，
不会把长期 idle session 算成可合批的 ready Job。

**当前分区在启动时确定；session 的 D/R 执行位置可以在阶段边界改变。**
`phase_pool.py` 的 `SharedPhaseQueue` 分别管理 D/R 共享就绪队列；队列中的任务尚未绑定执行 worker。
派发比较池内所有成员的执行 credit、预计完成时间、deadline 和历史状态复制成本，并在派发前预约显存。
候选 batch 使用按 deadline 排序的可行前缀，优先降低平均迟到，再比较每 Job 的摊销完成时间。
这是有界在线派发策略，不是全 trace 的联合最优求解。

质量 admission 仍先基于当前热状态 owner 选择 S/W；随后池级派发再选择执行位置。
尚未把每一种冷状态 placement 与所有 S/W 一起穷举。池内局部 step 规划沿用下述 ready DAG 目标。
不能把该版本描述成在线弹性扩缩池、全局最优质量选择或任意跨 GPU 的逐 latent 迁移。

逻辑池成员使用通信 rank，不包含 CUDA ordinal；rank 到节点和本地 GPU 的映射由进程部署决定。
多个 R 成员可从同一个 R 队列取任务；只有一个 R 成员时，仍然只有一张物理 GPU 的 R 算力。
worker 进程保持设备、权重和 CUDA stream 的绑定，未引入运行中跨角色借用物理 GPU。

## 池内：独立、不等待的局部规划

worker 暴露 `(session, chunk, latent, step, phase, owner)` 就绪头，
在每个 DiT step 边界重新选择，允许不同 Job 的不同 step 混合。
ABot chunk 内 attention 双向依赖 sibling 的当前层 KV，因此物理执行必须包含完整依赖组；
这不是为整个 Job 预留后续 step，也不能在不改变模型的情况下独立推进缺少 sibling 的一个 latent。
明确区分 Job batch 与 latent rows：Job batch=4 对应 latent rows=12。

`phase_batching.py` 枚举兼容依赖组的子集，使用记忆化动态规划构造这张 GPU
清空当前 ready step DAG 的真实串行时间线。首先最小化清空时间，即最大化已确定质量/产出的
ready cohort 供应率；再最小化总完成时间，同代价按 deadline 顺序。
未选任务只有在后续实际 invocation 中才推进，不再假设所有未选任务都能下一轮同时完成。
依赖组每次只推进一个 step，下一次重新纳入新到任务和更新后的成本。
每卡 12 latent slots 时最多四组，S4 的状态上限为 5^4；使用可采纳的服务成本下界剪枝，
不通过启发式删掉最优候选。这个有限 ready cohort 目标不等价于全 trace 的最优 Q×CPR。
没有额外凑批等待，也没有负 slack 禁用规则；实测 batch 没有收益时仍允许选择单组。
CUDA events 在后续循环用 query 收割，在线计时不逐 step synchronize。
CPU 命令和完成事件采用有界批量处理，避免持续 ingress 饿死执行或 admission。

## 异步增量 SST

每个 session 在 D/R 各维护一个独立逻辑页表和游标。一个 self-KV 物理页存放一帧 latent 在全部
DiT 层的 K/V；text-KV 单独存放在一个不可变页中。逻辑保留帧映射到 worker-local physical page IDs。
三个 sibling latent 共用此页表。12 个 latent 查询槽位不再隐含 12 份或 4 份完整 dense KV arena。
新增帧按需申请页，窗口滚动仅改变页表，淘汰页回到有预算上限的 worker 页池。
phase 完成后的 park 只发布游标并释放执行槽位，不分配或复制整份 canonical 历史。
每个 worker 同时预约的 Job 不超过 latent_slots/3，其余 heads 留在池级共享队列，
避免过早塞入某个 worker 的私有长队列。实际执行只向 compute stream 提交一个未完成 invocation，
期间继续处理控制与传输事件，完成后重新观察新到任务，不使用全局同步凑批。
attention 读取当前层时，按页表将可见 K/V 汇集给现有 SDPA/Sage 后端；临时张量只含当前层，
没有全部层历史的执行副本。该版本实现分页 KV 管理，尚不是直接读取页表的 fused paged-attention kernel。
投影、FFN 和 attention 的数学定义不变；动态页地址的本地执行使用 eager fallback，
原有显式跨 rank sibling 执行和其 graph 路径保持原行为。

1. D 完成发送最终 latent；首次额外携带 first-frame conditioning 与 prompt，R 不重复编码。
2. R 用自己的 clean 历史执行 refresh。
3. 只打包本 chunk 的 clean KV 尾部交回 D，再更新页表尾部对应的物理页。
4. D 提交后才可执行本 session 下一 Job，历史 KV 不逐 chunk 重传。

热状态 owner 只是一个有成本优势的位置，不是永久身份：池内其他 worker 的排队收益足以覆盖复制成本时，
SST 在阶段边界导出有效历史前缀、conditioning、游标和随机数状态，目标安装后才执行该阶段。
随机数状态编码为 CPU 十六进制字符串，随控制元数据发送，兼容 JSON 日志；
不经过 H2D/NCCL/D2H，GPU tensor lane 仅搬运模型状态。
源状态异步回收；独立 session 的搬运仍可并发。同一 Job 的 D-state 与 R-state 交接使用不同 ID。
冷迁移传输实际有效的 KV 页。接收直接使用目标 worker 的统一物理 KV 页池，
导入时接管已接收页并安装逻辑页表，不另外重建完整 canonical 副本；
迁入页在传输开始时即计入物理池预算。源 owner 在目标 state-ready 后回收原页。
池预算包括常驻页、正在接收的页和 GPU 尚未完成最后读取的退役页；可回收空闲缓存目标为 1 GiB。
退役批次记录一个 CUDA event，跨 stream 复用只等待该页的 event，不等待整张卡；
event 完成前不会将该页当作已释放的物理显存。校准另扣除非状态交接工作区。
传输成本先用保守带宽估计，再根据真实 pair 的字节/时间观测更新。
这条冷迁移路径会传历史 KV，不能把它计作仅传新尾部的增量快路径。

常态每 chunk 的逻辑 payload：latent 为 `F × C_latent × H_latent × W_latent × element_bytes`；
clean KV 为 `2 × layers × F × tokens_per_latent × hidden_dim × element_bytes`。

NCCL 使用持久通信 stream；冷迁移接收复用统一 KV 页池，其余短期 payload 使用原有有界接收 pool。
每个无序 worker pair 使用独立 NCCL communicator，启动时统一创建并预热；
不同 pair 不再共享 world communicator 的提交时间线，各自持有短提交锁和持久 stream。
发送只等待 payload 的 producer event，不等待源 GPU 后面排入的无关计算。
当前 NCCL 无可靠 tag，同一无序 GPU pair 保持交接顺序，不同 pair 可并发。
接收页在模型 stream 消费完成的 event 可用后才归还 pool。
新接收页直接在对应 peer 的传输 stream 上申请，避免 CUDA allocator 将默认 stream 上
仍有排队读取的已释放块交给并发 NCCL 写入；不通过等待整个计算 stream 来规避这一竞态。
启动预热实际 P2P 通道，将 NCCL 常驻缓冲计入后续显存校准，并为持久接收页缓存单独预留预算。
状态导入沿用已有 CUDA event，在 conditioning 等必要读取完成后报告 `state_ready`。
被接管的 KV 页继续属于 session，不归还短期接收池；其余 buffer 在读取完成后回收。

## VAE 独立异步组 batch

D 完成最终 latent 后即可提交 TAeW，不等待 refresh。
decoder owner 固定在 session 首次创建的 D worker，与可变的 D/R compute owner 分开维护。
每个 decoder owner 有独立 VAE 队列，跨 session 合批，按 session 串行更新 decoder state，使用独立 CUDA stream。
D 换 worker 后，仅将最终 latent 发回 decoder owner；不随 R 换卡迁移可变 decoder state。
跨 worker 的 latent 按 chunk 游标重排，重复/过期输出会报错，避免乱序推进 decoder。
KV 回传、其他 session 的 DiT 和 VAE 可以重叠。VAE 尚不是第三个专用 GPU 子集，
与 D 共用 GPU，不保证 kernel 并发一定产生加速。

Future 在实际 RGB decode 完成后发布，不能把 enqueue 当作生成完成。
回放在实际 decode 后计帧；CPU 帧转换、LiveKit 和客户端消费不限制 producer 指标。
成功生成的每个 12 帧 chunk 均进入原逻辑 playout 统计。
下一 Job 在 clean KV 提交后即可释放，不等待上一 Job 的 RGB decode。
生成计帧、质量和播放 slack 只在实际 RGB 完成后更新，RGB 先于 clean KV 到达时也立即计入播放余量；
每个 session 的 VAE FIFO 单独保持。
session 离开时等待已有输出完成后再清理，旧输出不能释放后续 Job 的执行预约。

## 验证和实验入口

CPU 测试覆盖真实小 DiT 的 native parity、跨 Job clean KV、随机数状态、窗口缩小/滚动/扩大、
独立 actor 调用链、双池显存、分区对阶段成本的响应、同 pair 有序及不同 pair 并发，
以及共享 R 汇聚多个 D、多 R 派发、D/R owner 切换后的连续生成、独立 decoder owner 和乱序 latent 到达。
回放保留 Producer FPS/session、CPR、Q、Q×CPR、action deadline attainment；另记录逐阶段 batch、
handoff 字节/时间、在线成本和事件 lag。性能收益须由完整回放判断。

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --standalone \
  --nproc-per-node=4 --module tools.validation.replay_abot_async_phase_trace \
  --trace /path/to/unchanged-balanced60-peak16.json --model-root /path/to/model \
  --profile /path/to/profile.csv --output /path/to/results \
  --collection-window 0.5 --pending-slots 4 --latent-slots 12 \
  --backend eager --attention sdpa --duration 1800
```

入口沿用 seeded controls、相位同步 release、FIFO4 和 held-idle 语义。
本轮不调整 session 数、频率、active 窗口或质量表。实验保存输入副本和代码快照。
pending 容量和满队列丢弃规则保持原配置；修复目标是提高处理能力，不能用扩队列掩盖过载。


### 动态 cohort 的显存峰值

旧 dense 路径将查询行、执行 arena 和历史状态的生命周期绑定在一起：cohort 切换时可能同时保留
canonical 历史、离批历史、执行副本和迁入缓冲。即使三个 sibling 共享离批历史，固定执行 arena
和目的端重建仍会占用额外显存。启动时的稳态校准不能约束这些动态峰值。

本地 D/R 路径现由 `paged_kv.py` 分离这三种身份：session 的逻辑保留帧、worker 的物理页，
以及临时 invocation 的查询行。cohort 改变只重绑查询与页表；三个 sibling 读写同一份 Job 历史。
页池同时为常驻状态和冷迁移接收提供物理页，park 不再生成历史副本。
旧显式跨 rank sibling 路径仍保留 dense arena，不能将其内存行为与新的本地 D/R 路径混为一谈。

显存预算还需覆盖模型权重、逐层 attention 聚合、VAE、增量交接与 NCCL 的真实开销。
分页不消除历史 KV 本身，也不消除 D/R 在不同 GPU 上保留两份状态的成本。
启动校准包含仍有其他 Job 活跃时的窗口增长与 cohort 切换；完整 trace 还需检查逐阶段
PyTorch allocated bytes 与物理页池峰值，不能仅凭校准通过断言不会 OOM。

未来用户的最小驻留空间只约束新 session 或需要增长 KV 分配的配置；已有状态不增长的
Job 仍可执行，不能因未来准入空间不足而停止所有计算。计算池内常驻用户数按唯一 session
key 统计，迁移中的源和目标副本分别占显存，但只代表一个用户。
