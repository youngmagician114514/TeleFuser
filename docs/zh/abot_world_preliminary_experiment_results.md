# ABot-World 五类 Full Trace 初步实验结果

日期：2026-09-10

## 结论摘要

当前 TeleFuser Motivation 实现在 4 张 H100、24 session 的五类真实 full trace 上稳定运行，
没有 admission violation、trace action drop、model compute error 或 `already in flight`。
五类共完成 21,750 个 action/idle job，生成 261,000 帧。

严格 action-request SLO 的五类微平均为 **75.43%**，producer CPR 的五类宏平均为
**86.67%**，归一化质量 Q 为 **0.903**，质量调整后的 `Q × CPR` 为 **0.783**。
已经完成的 job 中有 99.80% 能在 1 秒 deadline 内完成；因此当前主要损失不是 GPU job
执行过慢，而是约 24.4% 已发布 action 在进入 GPU 前被更新的 action 覆盖。

这组结果说明系统已解决“不 batch”和 LiveKit 反向限流两个早期实现问题，但仍没有达到
CPU 仿真的理想调度效果。当前最重要的真实实现缺口是：缺少能够保留未来 GPU slot 的
有界 dispatch queue，加上 KV local-cache 长度带来的 batch compatibility 碎片化。

## 实验配置

- Serving revision：`0f73894`（迁移关键路径与 producer 指标边界）。
- Post-run metric/teardown fix：`0217f1f`。
- GPU：H100 80GB，物理卡 4、5、6、7。
- Worker：4 个 process-NCCL worker，每 worker 最多保留 8 个 session。
- Workload：`action_world_v2` 五类 full trace，每类 24 session。
- Policy：Motivation；最大物理 batch 为 4；迁移开启。
- Profile：当前 H100 LF3 eager full-pipeline profile；不考虑稀疏注意力以及 FP16/FP8 切换。
- 质量：以 `b1_s4_w18_rho0_bf16` 为 `Q=1`；batch size 本身不改变质量。
- `output_gate=0`：LiveKit 仍实际发送视频并记录诊断，但其消费速度不参与模型 admission。
- 已知硬件基准：单张 H100 的无质量损失 ABot-World 请求接近 36 FPS。

完整实验目录：
`results/experiments/abot_action_suite_full_4gpu_20260910T112614Z_producer`

## 论文主指标定义

固定使用以下五项主指标：

1. **Action-request SLO**：按时完成的 action job / 输入 trace 发布的 action job。一个 job
   生成 12 帧、目标 12 FPS，因此 deadline 为 1 秒。被新 action 覆盖或未执行的请求算 miss。
2. **Producer CPR**：从 model-completed event 重建每个 session 的逻辑播放缓冲；action 和
   idle job 的 12 帧都完整计入，最后一个缓冲完整排空，LiveKit/WebRTC 丢帧不影响该值。
3. **Q**：按生成帧加权的归一化语义质量，`S4_W18=1`。
4. **Q × CPR**：质量调整后的连续播放体验。
5. **First-action job p95**：action job 在调度器内的 queue + model 延迟，不包含建房、
   WebRTC 建连和首轨发布冷启动。

`completed-job deadline attainment` 只作为诊断：它不包含未执行的 action，不能代替论文主 SLO。

## 五类主结果

| Workload | Action-request SLO | Producer CPR | Q | Q × CPR | First-action p95 |
|---|---:|---:|---:|---:|---:|
| steady_control | 75.73% | 89.74% | 0.904 | 0.811 | 259 ms |
| burst_join | 73.80% | 87.12% | 0.904 | 0.787 | 455 ms |
| rapid_action_change | 69.30% | 83.02% | 0.905 | 0.751 | 271 ms |
| interaction_pause | 71.91% | 77.90% | 0.900 | 0.701 | 279 ms |
| mixed_nonstationary | 86.37% | 95.55% | 0.903 | 0.863 | 564 ms |
| 五类平均 | 75.42% | 86.67% | 0.903 | 0.783 | 366 ms |

Action-request SLO 的微平均为 75.43%；表中“五类平均”是 workload 等权宏平均。

## 生产能力与 batch 诊断

| Workload | Released action | Completed action | Completed-job deadline | Producer FPS/demand | Mean physical B | Mean GPU util. |
|---|---:|---:|---:|---:|---:|---:|
| steady_control | 3,984 | 3,028 | 99.73% | 10.77 | 1.639 | 29.65% |
| burst_join | 3,984 | 2,965 | 99.28% | 10.45 | 1.571 | 25.48% |
| rapid_action_change | 3,984 | 2,761 | 100.00% | 9.96 | 1.437 | 26.05% |
| interaction_pause | 4,033 | 2,900 | 100.00% | 9.35 | 1.349 | 25.79% |
| mixed_nonstationary | 4,007 | 3,461 | 100.00% | 11.47 | 1.676 | 23.36% |

GPU utilization 是完整 capture 的一秒采样均值，包含 session 低需求和 pause 区间；它不能单独
解释 kernel 效率。不过，released/completed action 的差距以及低于仿真的 mean-B 共同证明，
当前运行时确实仍有可优化的供给与组批问题。

## 与 CPU 仿真的共性和差异

两者共用相同的核心思想：每 session 保存 slack、质量、最新动作和 KV 状态；搜索在 profile
给出的 latency/quality 上选择 GPU、batch、质量，并允许通过迁移改善后续收益。两者也都使用
每 session 至多一个 pending action 的 latest-action 语义。

| Workload | CPU 仿真 mean-B（参考） | 实际 mean-B |
|---|---:|---:|
| steady_control | 2.308 | 1.639 |
| burst_join | 2.276 | 1.571 |
| rapid_action_change | 2.224 | 1.437 |
| interaction_pause | 1.370 | 1.349 |
| mixed_nonstationary | 1.780 | 1.676 |

差异不是 LiveKit 消费侧造成的。本轮已关闭 output admission gate，所有成功 model job 的 12 帧
均在 producer 指标中完整计数。主要原因如下：

- CPU 仿真可以在预测的未来 GPU slot 上立即“安排”job；实际 child worker 同一时刻只接受一个
  物理 lease。GPU 忙时，action 仍留在 pending slot，下一次 heartbeat 会用最新 action 覆盖它。
- 实际 batch 要求 session 位于同一 GPU，并满足真实结构兼容性；当前 key 包含首 chunk 状态、
  control geometry、fidelity 和 KV `local_end_index`。不同生成进度会把 ready session 分散成多个 cohort。
- 实际实现受每 worker 8-session cap、迁移并发/cooldown、目标显存分配和 IPC/NCCL 命令调度约束；
  CPU 仿真对此更理想化。
- Pause 和 nonstationary trace 中，ready-set 的时间分布本身不同，因此它们不应被要求与 steady
  workload 有相同 mean-B。

因此，仿真高估了可实现 batch，但实际系统也仍有明确的工程缺口；不能把差异全部归因于仿真失真。

## 迁移结果

| Workload | Success/attempt | Route-ready p50 | Route-ready p95 | Raw first-layer avg. | Raw full-copy avg. |
|---|---:|---:|---:|---:|---:|
| steady_control | 48/48 | 77 ms | 1,182 ms | 4.57 ms | 27.25 ms |
| burst_join | 40/40 | 68 ms | 628 ms | 3.33 ms | 22.11 ms |
| rapid_action_change | 44/44 | 91 ms | 1,326 ms | 2.75 ms | 21.51 ms |
| interaction_pause | 58/58 | 48 ms | 1,496 ms | 3.87 ms | 24.78 ms |
| mixed_nonstationary | 72/73 | 54 ms | 1,614 ms | 3.65 ms | 27.56 ms |

典型 route-ready 已达到 200 ms 以内，五类 p50 范围为 48–91 ms。P95 仍为 0.63–1.61 秒，
所以“所有迁移约 200 ms”尚未达成。Raw P2P 首层和完整复制只有约 3.6 ms 与 24.6 ms；尾部主要
来自 source export、target allocation/prepare 和 child 命令排队，而不是 NCCL wire transfer。

`mixed_nonstationary` 的一次失败发生在 session departure 与 first-layer-ready 同时到达时：room 已从
admission scheduler 删除，旧回调仍尝试发布 target owner。它没有造成 job/trace 丢失；`0217f1f`
已用 runtime 生命周期锁消除该 teardown 假故障。该修复尚未另跑五类 GPU 实验，但已有聚焦回归覆盖。

## LiveKit 传输仅作诊断

| Workload | Client demand-SLO | Client CPR | Client A2F p95 |
|---|---:|---:|---:|
| steady_control | 71.19% | 80.16% | 4.28 s |
| burst_join | 67.03% | 79.65% | 24.75 s |
| rapid_action_change | 63.33% | 73.69% | 3.98 s |
| interaction_pause | 65.16% | 68.81% | 4.06 s |
| mixed_nonstationary | 83.81% | 84.75% | 4.99 s |

这些值包含 LiveKit/WebRTC queue、解码、建房、worker-ready 和首轨发布，不作为调度论文主指标。
尤其 `burst_join` 的 24.75 秒 A2F 是并发冷启动/建连尾部，而 producer first-action p95 只有 455 ms。

## 当前剩余缺陷与下一步

1. **严格 SLO 的主要缺口：** 增加每 GPU 的有界 future-dispatch queue，或等价的多 slot reservation，
   让已发布 action 在 GPU 忙时获得明确 slot，而不是停留在可被覆盖的单一 pending cell。
2. **Batch compatibility 碎片：** 研究按 KV local length bucketing，或在保持结果正确的前提下为不同
   `local_end_index` 做 padding/mask；该优化应放在 batch executor，不应污染 MotivationPolicy。
3. **迁移尾延迟：** 预分配/复用 target KV pages，并降低 export/prepare 命令排队；当前传输引擎本身
   已经接近 SlackServe 所描述的低开销范围。
4. **对比实验：** MotivationPolicy 与 FIFOPolicy 已是并列 policy，实现可复用同一 runner、producer
   SLO/CPR/Q 分析器和迁移/吞吐诊断。后续 baseline 不应自行定义另一套指标。

## 复现实验

```bash
source /public/fanyk1/lwb/dflash_psw_ess/.venv/bin/activate
ABOT_ACTION_SUITE_OUTPUT_GATE_ENABLED=0 \
ABOT_ACTION_SUITE_FRAME_CREDIT_ENABLED=1 \
ABOT_ACTION_SUITE_MOTIVATION_MIGRATION_ENABLED=1 \
bash tools/validation/run_abot_4gpu_action_suite.sh \
  --mode full \
  --gpu-ids 4,5,6,7 \
  --max-sessions-per-worker 8 \
  --motivation-policy motivation \
  --run-root results/experiments/<new-run-name>
```

每个 workload 的权威主结果位于 `result.json -> paper_metrics`；物理 batch 分布位于
`dispatch_analysis/summary.json`；迁移分位数位于 `metadata-after.json ->
turboserve_routing.migration_calibration`。
