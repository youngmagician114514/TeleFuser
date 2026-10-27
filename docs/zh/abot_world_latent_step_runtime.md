# ABot-World：完整 Latent–Step 任务执行器

日期：2026-10-24。本实现是可选的直接 pipeline 路径，尚未替换已有 LiveKit 回放的执行协议。

## 调度单位

每个调度任务由 `(session_id, chunk_start, latent_index, step_index, refreshing)` 标识。
物理 batch 的一行是一个**完整 latent 的一次 DiT step**，形状为 `[C,1,H,W]`。
保留整个空间网格，不将一个 latent 按 GPU 数继续切成空间分片。

默认 chunk 仍产生三个 latent、十二帧，使用四次 denoising forward 和一次 KV-refresh forward。
chunk 是输出和因果依赖的边界；物理执行行与原来的一整个 chunk 不同。
例如同一张卡的物理 batch 可以同时包含 A 的 latent 0 / step 2 和 B 的 latent 1 / step 1。

## 多卡分配

所有 GPU 都加载完整 DiT 权重。新 session 的三个 latent 按轮转顺序分配到 GPU 池，
该 session 的 latent affinity 在后续 step 和 chunk 间保持不变，避免每轮重新搬运历史 KV。
GPU 数不要求等于三，也不要求整除三个 latent。

两个 session 在四卡上的一种实际分配为：

| GPU | 完整 latent 任务 |
| --- | --- |
| 0 | A-latent0、B-latent1 |
| 1 | A-latent1、B-latent2 |
| 2 | A-latent2 |
| 3 | B-latent0 |

同一张卡可以处理多个 session 的 latent；同一 session 的两个 latent 也可以放在同一张卡。
单卡同样支持这套任务语义，不需要初始化 distributed process group。

十八个 session 共暴露五十四个 latent：四卡均匀分配需要每卡最多十四个任务槽，八卡最多七个。
这是槽位数量，不是性能或显存容量保证。新 batch size 的单位是 latent，不能直接套用原来的 chunk-batch profile。

## Attention 依赖与通信

当前 ABot attention 的每层使用同一 chunk 全部 latent 的当前 K/V。
因此三个 latent 可以放在不同 GPU，却不能在缺少兄弟 latent 当前 K/V 时独自跨层、跨 step 推进。
调度器按兼容性组织 dependency wave，并为所有兄弟 latent 安排执行资源；一个 wave 可以包含多个 session 的不同 step。
每个 GPU 内，投影、attention、cross-attention 和 FFN 都以本地完整 latent 为 batch row。

`LatentTaskExchange` 负责将每张卡产生的当前 K/V 按 task routing table 恢复到所属 session 的 latent 顺序。
通信内容是每层的当前 K/V；常规三个 latent 的 chunk 不在每步广播历史 KV。
只有完成 refresh 后才汇集生成 latent，由 rank 0 批量 VAE 解码。

当前使用整个执行组的 all-gather，收发 batch row 数不同的 GPU 采用 padding；padding 不进入实际 attention。
空闲 rank 只参与通信，不执行无效 latent 的 DiT。全组 collective 仍会造成 wave 内的同步，
当前版本不是独立 session 间完全解耦的异步 P2P 执行器。

可选的一/二 latent chunk 会在完成时同步有效历史状态，确保下一次三个 latent 恢复执行时没有缺失历史。
该额外同步不出现在正常的三 latent 路径。

## 状态存储和执行后端

- `LatentStepKey`：任务身份、step 和 owner。
- `ABotWorldLatentStepRuntime`：admission、任务槽位、依赖 wave、采样器和完成提交。
- `LatentStepArena`：复用本地稠密 KV 与输入缓冲区。稳定成员的 batch 不逐步复制历史 KV。
- `LatentStepExecutor`：可替换的 eager / per-step CUDA Graph 执行后端。
- `LatentTaskExchange`：独立于调度策略的当前 K/V 和最终 latent 传输。

输入、action、timestep、KV arena 地址复用；Graph 按执行形状、KV occupancy、追加/覆盖阶段、
refresh 状态和全局 owner 布局缓存。不同 session 或 timestep 值可以更新相同静态输入缓冲区。
Graph 捕获前后恢复 KV，避免 warmup/capture 使 session 多推进一次。
各 rank 根据同一全局 profile 失效 Graph，避免空闲 rank 的本地分配时机造成 capture/replay 不一致。

**CUDA Graph 后端不是整模型单一常驻 kernel。** 它复用现有模型和 attention 算子，减少逐算子 CPU 提交开销。
执行器与任务状态分离，为后续专用 GPU 常驻执行后端保留替换边界；本版本没有实现 GPU 端任务队列或整 DiT megakernel。

现有 attention 使用连续 KV batch，因此成员变化仍有历史 KV 搬运；同卡的多个兄弟 latent 仍有独立的稠密 KV row。
arena 按层、按 K/V 字段暂存并立即交回任务引用，避免同时保留整套第三份 KV。
重组按组内最大有效前缀复制，离队保存按各自有效前缀复制；
同为 W6 的任务不再因为物理 allocation 保留 W18 容量就复制整份 W18。
物理行现在按 Job 需要的 W 保留容量，W6 不再为每个离队行分配 W18。
同批稠密 arena 保留已达到的最大容量，之后只扩容、不随 W 缩小而重建；
attention 仍按当前 W 读取有效历史，离队状态仍只保留各自 W 所需的容量。
这样避免 W6/W12/W18 交替时反复分配大块显存、搬运未换位成员和失效 Graph；扩大窗口不会恢复已经丢弃的帧。
显式 materialize 时仍还原 canonical 容量，只发布有效 KV，未使用尾部清零。
这些成本仍须在 GPU 实验中衡量，不能仅根据任务粒度宣称一定提速。

## 调用方式

```python
runtime = pipeline.create_continuous_runtime(
    granularity="latent_step",
    max_batch_size=14,       # Local whole-latent slots, not session count.
    backend="cuda_graph",  # Use "eager" for the reference path.
    max_graphs=4,
)
session_a = runtime.create_session(image_a, prompt_a, seed=42)
session_b = runtime.create_session(image_b, prompt_b, seed=43)
runtime.submit(session_a, {"W": True})
runtime.step()
runtime.submit(session_b, {"D": True})
while runtime.has_pending:
    for chunk in runtime.step():
        if runtime.is_output_rank:
            publish(chunk.session_id, chunk.decoded)

# Keep runtime ownership across subsequent chunks; history stays on its owner.
runtime.submit(session_a, {"W": True})
while runtime.has_pending:
    runtime.step()

# Explicitly materialize canonical pipeline state before snapshot/native calls.
runtime.release_session(session_a.session_id)
runtime.close()
```

所有 rank 必须以相同顺序提交 session、执行 step 和释放状态，使用相同控制输入、seed 和 profile。
runtime 保持 session 的独占租约；普通生成、snapshot、迁移必须在显式 release 后进行。
release 只要求该 session 的当前 chunk 已完成，其他 session 可以继续保持 pending；
完成的 session 可以交还状态，再重新进入任务池，无须先排空整个执行组。
交还时保留其他任务的 KV 缓冲和 Graph 缓存，不复制整批历史 KV。
所有 rank 仍须在相同的执行边界调用 release，其状态广播在交还所有权时发生，不加入正常 step 的前台路径。
`release_session(session_id, materialize=False)` 用于已经离开的 session：直接关闭并丢弃状态，
不回传完整历史 KV；默认仍为 materialize=True，供后续原生 pipeline 调用或 snapshot 使用。
`close(materialize=False)` 丢弃并关闭 session，供一次性 benchmark 结束时使用。
丢弃关闭不为即将释放的任务克隆历史 KV，避免清理阶段额外占用一整批 KV 显存。

## 后续 GPU 实验入口

本轮没有启动 GPU 实验。以下命令供 GPU 可用后顺序运行：

```bash
# Four GPUs; --batch-size counts sessions, --latent-slots counts tasks per GPU.
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone \
  --nproc-per-node=4 --module tools.validation.benchmark_abot_latent_parallel \
  --mode latent --execution latent-step --backend eager \
  --batch-size 18 --latent-slots 14 --warmup 6 --chunks 12 \
  --model-root /path/to/ABot-World-0-5B-LF --image /path/to/initial.png \
  --save-decoded --output /tmp/abot-latent-step-4gpu-eager.json

# Change only the execution backend to compare graph costs and decoded output.
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone \
  --nproc-per-node=4 --module tools.validation.benchmark_abot_latent_parallel \
  --mode latent --execution latent-step --backend cuda_graph \
  --batch-size 18 --latent-slots 14 --warmup 6 --chunks 12 \
  --model-root /path/to/ABot-World-0-5B-LF --image /path/to/initial.png \
  --save-decoded --output /tmp/abot-latent-step-4gpu-graph.json
```

八卡使用 `--nproc-per-node=8`、`--latent-slots 7`；`--stagger-steps 2` 可验证晚到请求，
该参数使用 dispatch 次数，不代表真实世界的 action 时间。
脚本记录最慢 rank wall time、实际 latent batch、每卡工作量/峰值显存、Graph capture/replay 次数和逻辑通信字节。
通信字节是包含 padding 的 ring all-gather 模型，不是测量的 NCCL wire traffic。

`tools.validation.replay_abot_latent_step_trace` 可直接回放已有 lifecycle trace。
它读取原 session active/idle、到达和离开时间，按原 seed、heartbeat interval 和 jitter
重建客户端 controls，先应用原服务的单 session 最小释放间隔，再按 0.5 秒公共边界合并并释放 action；
pending 保持有界 FIFO。这里不能只按公共边界释放，否则 heartbeat 抖动会让生成的 action 数量增加。
输入的逻辑时间不包含 HTTP/LiveKit 的网络抖动。控制时间通过 CPU Gloo 同步，模型数据仍走 NCCL。
DiT 和 TAeW 视频解码器常驻 GPU；仅在 session 创建时使用的 T5 和图像编码 VAE
使用已有 stage CPU offload 支持，避免这些权重与 latent 历史 KV、dense batch arena 争用显存。
session 创建时的上载、编码和卸载时间仍计入 trace 实际运行时间。
该工具默认固定 S4/W18；传入 `--profile` 后启用逐 Job 的 S/W 质量选择，
使用当前 playout slack、在线完成时间预测和完整生成路径 Q，复用 session/idle 语义和 producer 指标聚合。
它衡量新的共享 latent 执行池，没有接入原 TeleWorldServer 的 worker placement 搜索和 SST 迁移。
输出 `dispatch-trace.jsonl` 记录 chunk 完成，`latent-step-dispatch.jsonl` 记录实际 latent-step 批次，
两者计数单位不同。配置 Q=1 表示参考 profile，不代表已经通过视频数值一致性验证。

## 混合窗口与在线耗时更新

`runtime.submit(session, controls, fidelity=...)` 在 Job 边界选择 S2/S3/S4 和 W6/W12/W18；
同一 Job 内的 profile 保持不变。每个 step 可以同时执行不同 session 的不同 timestep 和不同窗口。
投影与 FFN 使用整个物理 batch；self-attention 根据窗口、sink 和实际 KV 游标分组，
只读取各组的有效历史，不把 W6 的 attention 计算填充到 W18。refresh 与 denoising 仍分批执行。
Relative-RoPE 的 KV 更新只使用 `global_end - current_start` 和本地游标，
不同绝对 chunk 编号不再强制拆成独立更新。写入完成后，连续行若具有相同的有效 KV 长度和
本地 query 位置，会合并成一次 attention；append 和 refinement 的不同写入方式因此不会继续拆分相同的读取布局。
原生 whole-chunk 和绝对 RoPE 的位置语义保留，混合 S/W 的选择集合没有改变。
窗口缩小保留有效 prefix 和最近 tail；之后扩大窗口不能恢复已经丢弃的历史。
session 的 canonical 容量上限仍为 W18，latent 执行器中的保留行按当前 Job 的 W 分配；
本次没有实现 paged KV，成员换入/换出仍需要 dense KV 复制。

新策略不使用离线 chunk-batch 延迟预测 latent-step。CSV 只提供归一化质量 Q；
执行时间由启动校准和回放期间的 CUDA event 测量提供：

- 启动前用足以覆盖每卡槽位上限的临时 session 填满历史并测量各窗口；四卡、每卡八行时使用十个 session。
  每卡十二行对应十六个 session。先按 W6→W12→W18 测量完整占用，再关闭额外临时状态，
  用剩余半池按 W18→W12→W6 测量部分占用，避免完整 W18 保留池与 partial-batch arena 同时占用显存。
  这些 session 关闭后才开始真实 trace 的时钟和指标。
- denoising 与 refresh 分别维护各窗口的 EWMA 成本；refresh 测量包含输出汇集和 TAeW 解码。
- 每次 wave 记录 CUDA event，在已有 chunk 完成边界批量读取先前已完成的 event，
  用 CPU Gloo 聚合各 rank 的最大耗时。dispatch 路径不新增 `cuda.synchronize()`。
- 按观测 batch 大小保留各窗口的耗时曲线，并估计各 rank 的剩余工作，使用最慢 rank 的完成时间，
  不把多张 GPU 当成一个串行资源。最近 128 种实际 wave 布局另存整体耗时 EWMA，
  按 rank、窗口、append/refinement/refresh 行数匹配；其余布局用在线行成本估计。
  整体时间已经包含 NCCL 和同步等待，不再叠加同一笔通信费用。

这仍是在线吞吐近似：插值不代表 kernel 时间严格线性；首次出现的 batch 组成需要实际观测后修正。
`online-calibration.json` 保存起始测量，`online-timing.jsonl` 保存实际 wave 时间；
`result.json` 保存最终在线成本、观测次数和各 profile 的完成数量，便于审计预测依据。
启动校准不计入真实 trace 指标，但真实 session 的创建、编码和关闭仍计入。

## 配置、placement 与 step batch 的统一决策

一个 action Job 的三个 latent 构成 attention 依赖组：`{A_s, B_s, C_s}`。
每个 latent 完整放在一个 GPU 上；兄弟 latent 可以同卡、两卡或三卡分布。
在每层 attention 需要的兄弟 K/V 到齐后才能继续该层，不能单独把 A 推进到下一个完整 DiT step。

新策略采用一个目标，时间单位为秒：

\[
\min \sum_j \left[\frac{D_j\ell_j+\frac12\ell_j^2}{L_j}
  +\lambda_Q(1-Q(p_j))L_j\right],\qquad
D_j=\max(0,-s_j),\quad
\ell_j=\max(0,\widehat C_j-\max(0,s_j)).
\]

`s_j` 是当前可播放 slack；`C_j` 是相对现在的预计 Job 完成时间，包含剩余 denoising、KV refresh 和解码；
`p_j=(S,W)` 是完整去噪路径；`L_j` 是生成视频秒数。第一项是预计等待期间累计播放欠账的面积除以视频长度，
第二项是质量损失。线性迟到在所有 session 已经超时时，过去欠账只成为不影响选择的常数，容易让短 Job 持续挤出长等候 Job；
这里欠账越大，相同未来等待的代价越高，无需另外增加 aging 阈值或等待超时策略。
这不是直接计算真实 CPR 的精确优化器，也不保证全局最优。负 slack 的 Job 保留候选资格。
数值相同的 `lambda_quality` 在这个直接质量损失目标中，不等价于旧的 quality-EMA utility 权重。

每次 `plan()` 执行一次增量近似求解：

1. 收集已运行 Job 的下一 step，以及尚未 dispatch 的 action/idle Job；每个 session 仍只暴露一个当前 Job。
2. 新 Job 先用相同 S/W 候选和完成时间下界初始化排序预测，避免所有新 Job 都以最慢的参考配置参与排序。
   用各 rank 的剩余 row work 和 weighted processor-sharing 尾部近似估计完成时间，
   权重为 `1 + D_j/L_j`，与欠账目标一致；按“先推进该组”对目标的改善排序。
3. 依次尝试加入完整兄弟组。新 Job 同时比较合法 S/W 和首次 placement，已有 Job 沿用配置及 owner。
   四 GPU 的等计算量 placement 共有 20 种 GPU 多重集合，包括同卡、两卡和三卡布局；不枚举等价的 latent 排列。
4. 对每个候选前缀，模拟当前因果头和已接受 FIFO action 的完整物理 wave，再用预计完成时间评估同一目标。
   后续 wave 同样受每卡 latent slots、固定 owner 和 denoise/refresh 精度组约束，
   不允许被推迟的 Job 免费获得将来的理想满 batch 吞吐。该有限前瞻只使用真实队列中的 action，
   不预测尚未收到的输入，也不生成虚构 idle。前驱完成 refresh 后才暴露后继，后继重新支付全部 denoise 和 refresh 成本。
   持续扩展可行前缀并保存目标最好的前缀，允许跨过暂时的 rank 不均衡；
   同目标值时选择更多工作的前缀。有可执行组时至少 dispatch 一组。
5. 首次 dispatch 才固定新 Job 的 S/W 和 owner；完成一个 step 后重新规划下一 wave。

未接纳的新 Job 只持有临时 S/W 预测，不会提前修改真实 Job、RNG 或 KV；最终 admission 仍重新比较所有 S/W。
内层 S/W 与 placement 排序仍使用低成本 weighted processor-sharing 近似；
外层 batch 前缀比较和最终报告的完成时间使用可实际装入槽位的 wave 前瞻。
FIFO 前瞻中，第 k 个后续 Job 使用 `s_j + k L_j` 计算预计播放欠账；
S/W 暂沿用当前候选作 receding-horizon 估计，真正 dispatch 后继时仍重新选择，未提前提交质量配置。
已运行 Job 的后继数量取真实 pending 数；新队首的后继数取 pending 数减一。
因此每个 session 依然只有一个可执行 Job，队列顺序和四个 pending 槽位没有改变。
忽略这些已知后继会低估延后小 batch 的代价，尤其在多个欠账不一的 session 持续排队时。
前瞻只更新剩余 step、物理布局计数和完成时间，不为每个假想 step 构造新任务；
同次规划中相同物理布局复用时间预测，在线观测更新后清除缓存。
前瞻采用与欠账目标一致的优先级，不代表全局最优，首次布局和全局 NCCL barrier 的成本仍需要 GPU 回放校验。
同样的目标值与完成时间下，placement 依次比较最大显存占用、占用平方和，避免等价布局长期偏向较小 rank 编号。
首次 placement 的 KV 预算根据校准后各 rank 的可用内存、allocator 缓存和观测瞬时峰值计算；
模型、固定 arena 和 workspace 另行保留。保留状态按 owner 数和当前物理 W 容量计费，
新配置按将分配的 W 容量计费；不把未分配的 canonical 上限当作实际显存。
已有状态超过估计预算时允许不增加内存的执行布局继续推进。

`plan()` 返回 `session_ids`、新 Job 的配置与 owners、预计 wave 时间及目标值。
`runtime.submit(..., owners=(...))` 应用首次布局；`runtime.step(plan.session_ids)` 执行明确选定的完整依赖组。
原有不传 owners、不传 wave 的调用方式继续使用原来的默认布局和装批逻辑。
历史 KV 已驻留后，本实现不会在普通 step 边界重新搬运它们；该新执行池仍未对接原 SST 的在线迁移入口。
当前通信后端即使兄弟同卡也仍参与全组 all-gather，规划不能假定“同卡等于零通信”。

新 session 的 cross-attention 缓存初始化状态不再是装批兼容条件。
同一物理 batch 内只对尚未初始化的行计算文本 K/V，已保留行的 prompt KV 不重新计算；
原生 chunk 路径没有行初始化元数据时仍按原逻辑初始化整个 batch。
Graph key 保留所有 session 的初始化状态，确保捕获路径与真实状态一致。

在线时间模型保留按 `(W, denoise/refresh, batch rows)` 测量的耗时曲线，
同窗口样本更新对应行数的 EWMA，其他行数用插值或外推。不同 rank 的时间取并行关键路径。
启动校准覆盖至每卡槽位上限，并补测较小 batch 行数，让模型能够观测 launch 摊薄，
不再从三行的小 batch 斜率预测八行。校准 session 创建后立即提交并取得分布式 owner，
避免先在每卡同时积累整个校准池的 canonical KV。
超出实测范围时保留最大已测 batch 的吞吐率，不额外假设尚未测到的摊薄收益。
混合 W 的一次整体耗时不能辨识每个 W 的独立成本，因此只更新对应联合布局的 EWMA，
不再用同一缩放因子同时调整所有 W。已有精确布局观测优先于曲线近似。

KV refresh 与普通 denoise 仍属于不同精度执行组：原生采样器的 refresh 刻意在 autocast 外执行，
其保留 KV 会影响后续 chunk。当前没有为了消除这个组界而改变原生精度语义。

## 稳定执行槽位与初始化复用

2026-10-24 后续修复保持每个仍在 batch 中的 latent 行尽量占用原 dense slot，
只把新成员和需要移动的行复制到 arena。所有 rank 从相同全局 wave 历史计算行排列，
K/V 通信、frame index、采样结果和 Graph key 使用同一排列；没有增加协商 collective。
空出的前缀槽位才补入其他成员。window、sink 和 cursor 的 attention 分组仍按实际本地行顺序描述。

session 离开时保留其他成员的 slot 身份，直到下一次 bind：
清掉整个 cohort 会使暂未执行的剩余 session 持有无法追踪的 arena view，存在被新成员覆盖的风险。
离开成员只标记为丢弃，不为其复制即将释放的 KV；其他成员按正常换出路径保留历史。

每个 pipeline 分别缓存最近一份图像编码和文本编码，仅在图像内容/尺寸、prompt、编码 stage、
输出配置和 dtype/device 完全匹配时复用。图像 VAE 使用 posterior mean，文本编码为确定性推理；
每个 session 获得独立克隆，KV、RNG、scheduler 和解码状态继续独立。
这是有界的精确输入复用，其他图像或 prompt 会重新编码，不能推断不同用户输入也有相同收益。
关闭 pipeline 时释放缓存。回放在校准结束后主动清空该缓存，首个真实 session 仍支付冷编码成本。

arena 高水位容量增长可能更换指针，使用从全局容量布局派生的 epoch 统一失效 Graph，
防止复用仍指向旧 KV allocation 的捕获。行排列也进入 Graph key。

latent 执行器用 host cursor 描述每个 session 的一致进度，attention 直接接收 wave 的已知位置。
不再为每层、每个 sibling 重复发射 GPU scalar fill，也不在每个 Job 读取 GPU cursor；
显式交还 canonical 状态时才发布各层的游标 tensor。普通 whole-chunk 路径继续使用原有设备游标。

采样在一个 physical batch 中更新 x0，并按下一 sigma 分组加噪。
官方四个训练时间位置对应的 sigma 在 Job 建立时确定，避免旧 `add_noise` 每行把 CUDA timestep 拷回 CPU。
每个 session 仍按原来顺序生成完整 chunk 的 noise，所有 rank 保持相同 RNG 消耗；
加噪沿用 CPU scalar 的原生类型提升和分次舍入，未用更低精度公式代替。
H100 上的 FP32/BF16 对照验证该采样更新与原逐行路径逐元素一致，RNG 状态相同。

启动校准关闭临时池后只执行一次 `empty_cache()`，归还校准阶段的大块 allocator reservation；
真实 dispatch 路径不清缓存、不增加 GPU 同步或重试等待。

`runtime.create_session()` 先创建独占租约下的 session 元数据和空 KV shell；
首次 `submit()` 确定 owner 和 W 后，只在对应 rank 分配实际 latent 行。
这样无需先在所有 rank 建立完整 W18 状态，再在 placement 后释放副本。
显式交还时才还原完整 pipeline 状态；尚未提交 Job 的 session 也可以关闭或交还。
原生 `pipeline.create_interactive_session()` 继续立即建立 canonical KV，原 whole-chunk 接口行为不变。
四进程 CPU 测试覆盖延迟分配、W6→W18→W6 切换、RNG 一致性和状态交还。

2026-10-24 修改完成后的 CPU 验证不代表 H100 性能结论；FPS、CPR、质量和首帧延迟的改善必须由同 trace GPU 回放确认。

CPU 测试对照实际小型 DiT，覆盖单进程和 2/4/8 进程、空闲 rank、晚到请求、混合 step、
多窗口混合与动态窗口切换、多 chunk 与 KV 淘汰、每 session RNG、其他任务仍 pending 时的独立状态交还、
重新入池及后续原生生成，也覆盖同卡/分卡自定义 owner、指定 wave、负 slack 与 KV 预算。
在线策略测试确认旧 CSV 延迟变化不影响决策，而实测成本变化会影响决策。
GPU CUDA Graph / NCCL 捕获、H100 BF16 视频一致性、显存容量和性能仍须实测；CPU 结果不替代这些结论。

## 兼容性边界

默认 `granularity="chunk_step"` 与已有原生 whole-chunk 路径保留。旧三卡 frame-sharded 路径仍可用于对照。
新增接口位于 ABot 模型/pipeline 和独立 distributed 通信模块，没有新增环境变量或共享服务 schema。
LiveKit worker 的原有 whole-chunk dispatch/completion 和 SST 没有自动切换为本执行器。
