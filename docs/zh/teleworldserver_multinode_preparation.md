# TeleWorldServer 跨机运行（2 × 8 H100）

当前版本已经提供可运行的跨机 serving data plane。每台机器启动一个
`node-agent`，它只管理本机的 process-NCCL workers 和 LiveKit transport；中心
`stream-serve` 进程管理全局 admission、Motivation policy 和 session placement。
拥有 session 的节点直接加入同一个 LiveKit 房间并发布视频，视频帧不会经过中心
节点二次转发。

## 节点配置

两台机器使用同一份 pipeline、LiveKit 凭据、拓扑字符串和 NCCL master。每台机器
声明本机的 8 张卡；节点地址必须是对另一台机器可达的地址。

```bash
export PIPE=/path/to/abot_world_pipeline.py
export TOPOLOGY='node-0@host-a=0,1,2,3,4,5,6,7;node-1@host-b=0,1,2,3,4,5,6,7'
export LIVEKIT_URL='https://livekit.example'
export LIVEKIT_KEY='...'
export LIVEKIT_SECRET='...'
export AGENT_SECRET='change-me'

# 只在非中心节点执行；下面以 node-1 为例，使用 node-1 的地址和 rank。
telefuser node-agent "$PIPE" \
  --livekit-url "$LIVEKIT_URL" --livekit-api-key "$LIVEKIT_KEY" \
  --livekit-api-secret "$LIVEKIT_SECRET" --num-workers 8 \
  --max-sessions-per-worker 12 \
  --worker-gpu-map '0;1;2;3;4;5;6;7' \
  --node-id node-1 --node-address host-b --node-rank 1 \
  --node-agent-port 8090 --node-agent-secret "$AGENT_SECRET" \
  --cluster-world-size 2 --cluster-topology "$TOPOLOGY" \
  --nccl-master-address host-a --nccl-master-port 29500 --nccl-world-size 16 \
  --skip-validation

# 仅在 node-0 启动中心 API；node-0 的本地 worker 由 stream-serve 管理，不能再启动一个 node-agent。
telefuser stream-serve "$PIPE" --port 8088 \
  --livekit-url "$LIVEKIT_URL" --livekit-api-key "$LIVEKIT_KEY" \
  --livekit-api-secret "$LIVEKIT_SECRET" --num-workers 8 \
  --max-sessions-per-worker 12 \
  --worker-gpu-map '0;1;2;3;4;5;6;7' --worker-mode process-nccl \
  --node-id node-0 --node-address host-a --node-rank 0 \
  --node-agent-port 8090 --node-agent-secret "$AGENT_SECRET" \
  --cluster-world-size 2 --cluster-topology "$TOPOLOGY" \
  --nccl-master-address host-a --nccl-master-port 29500 --nccl-world-size 16 \
  --motivation-profile /path/to/profile.csv --skip-validation
```

`cluster_topology` 中的 GPU 编号是节点本地编号，因此两个节点都出现 `0`–`7` 是正确的。

## 控制面和迁移边界

- 单机默认值仍为 `node-0`、`127.0.0.1`、world size 1；单机 NCCL 继续使用原来的本地临时端口。
- 跨机 NCCL 使用固定 master 地址/端口和全局 rank：node-0 使用 rank 0–7，node-1 使用 rank 8–15。
- `node-agent` 的端口只承载内部 JSON 控制 RPC，提供 health、session start、batch dispatch 和 stop。请求按节点串行发送以保持 session 生命周期顺序，但不同节点可以并行。
- `node-agent` 会先绑定控制端口，再在后台加入全局 NCCL rendezvous；因此应先启动 node-1 agent，再启动 node-0 coordinator，但两边不会因为等待对方进程而互相死锁。协调器会在启动阶段重试远端 health。
- 全局 scheduler 使用 `node-id/worker-id` 作为 owner，因此两个节点重复的本地 GPU 编号不会冲突。新 session 可以直接分配到任一节点，远端 agent 返回 pipeline-session ready 后中心才向 Motivation 注册该 session。
- 节点内 re-home 使用 NCCL P2P state-transfer；跨节点 re-home 使用同一套 export/prepare/send/receive/commit 协议，由源、目标 NodeAgent 协调，NCCL 使用全局 rank 进行跨机 P2P。跨节点候选会额外计入 `motivation_migration_cross_node_extra_seconds` 的 RDMA/跨机估计成本。
- 跨节点迁移先恢复目标模型 session，再把 LiveKit transport 接到目标节点；源节点在 ownership commit 后释放旧 model/transport。迁移失败时目标 discard、源 resume，session 仍保持原 owner。
- 当前跨节点迁移需要节点间 NCCL 网络可达；如果部署环境没有稳定的 RDMA/NCCL 跨机链路，应通过 `--no-motivation-migration` 或关闭 Motivation migration，而不是让调度器把跨机链路当作本地 NVLink。
- 单机 `cluster_world_size=1` 时仍使用原来的 worker id、单机 pool、单机 NCCL rendezvous 和原有调度路径。
