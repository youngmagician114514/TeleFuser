# TeleWorldServer 跨机准备（2 × 8 H100）

这份文档记录当前代码已经接入的跨机边界。它完成拓扑描述、全局 NCCL rendezvous、
全局 worker endpoint 标识和结构化跨节点迁移请求；真正的远端 session 路由仍需要
一个实现 `WorkerPool.migrate_session_remote` 的多机 WorkerPool，当前本机 router
不会误把远端 GPU 当成本机 GPU。

## 节点配置

两台机器各自启动一个服务进程，每个服务进程只声明本机的 8 张卡。两个进程使用同一个拓扑字符串和同一个 NCCL master 地址：

```bash
# node-0 / host-a
export TELEFUSER_LIVEKIT_NODE_ID=node-0
export TELEFUSER_LIVEKIT_NODE_ADDRESS=host-a
export TELEFUSER_LIVEKIT_NODE_RANK=0
export TELEFUSER_LIVEKIT_CLUSTER_WORLD_SIZE=2
export TELEFUSER_LIVEKIT_CLUSTER_TOPOLOGY='node-0@host-a=0,1,2,3,4,5,6,7;node-1@host-b=0,1,2,3,4,5,6,7'
export TELEFUSER_LIVEKIT_NCCL_MASTER_ADDRESS=host-a
export TELEFUSER_LIVEKIT_NCCL_MASTER_PORT=29500
export TELEFUSER_LIVEKIT_NUM_WORKERS=8
export TELEFUSER_LIVEKIT_WORKER_GPU_MAP='0;1;2;3;4;5;6;7'
export TELEFUSER_LIVEKIT_WORKER_MODE=process-nccl

# node-1 / host-b：只修改以下三项
export TELEFUSER_LIVEKIT_NODE_ID=node-1
export TELEFUSER_LIVEKIT_NODE_ADDRESS=host-b
export TELEFUSER_LIVEKIT_NODE_RANK=1
```

也可以使用命令行参数 `--node-id`、`--node-address`、`--node-rank`、`--cluster-world-size` 和 `--cluster-topology`。`cluster_topology` 中的 GPU 编号是节点本地编号，因此两个节点都出现 `0`–`7` 是正确的。

## 当前边界

- 单机默认值仍为 `node-0`、`127.0.0.1`、world size 1；单机 NCCL 继续使用原来的本地临时端口。
- 跨机 NCCL 使用固定 master 地址/端口和全局 rank：node-0 使用 rank 0–7，node-1 使用 rank 8–15。
- `GpuSchedulingState`、`WorkerState` 和 `SessionStateTransferRequest` 现在携带 node/address 元数据，调度诊断可以区分 `intra_node` 与 `inter_node`。
- 每个本地 worker 都有稳定的 `node-id/worker-id` endpoint；同名的 `worker-0` 在两个节点上不会冲突。`ProcessWorkerSpec.rank` 同时记录它在全局 NCCL group 中的 rank。
- `NodeAgentMigrationRequest` 是远端 agent 的控制面契约，完整携带 session、源/目标节点、GPU、地址、时间估计和 state bytes；它不把 KV payload 放进 HTTP/控制面消息。
- 没有远端 WorkerPool 时，跨节点迁移会返回明确错误并保留源 ownership；它不会把 `node-1/gpu-0` 当成 node-0 的本地 GPU 执行。
- 当前版本还没有宣称可以直接完成跨节点 session state transfer。下一步需要实现远端 WorkerPool/NodeAgent：目标节点准备 paged state、跨节点传输、然后由全局 ownership commit 原子切换路由。

因此，这一版适合先进行 2×8 拓扑启动和 NCCL rendezvous smoke test；在接入远端 WorkerPool 之前，不应把它当作完整的跨机性能结果。
