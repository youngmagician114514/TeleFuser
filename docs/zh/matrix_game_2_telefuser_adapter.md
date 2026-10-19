# Matrix-Game 2.0 的 TeleFuser 适配

这次适配把 Matrix-Game 2.0 作为独立的 TeleFuser pipeline。入口文件是
`examples/matrix_game/matrix_game_v2_livekit_service.py`，核心实现位于
`telefuser/pipelines/matrix_game_v2/`。它复用 TeleFuser 的
`create_session`、`push_chunk`、`pull_chunks`、`close_session` 契约，不导入
ABot-World 的 pipeline、profile、KV cache 或调度实现。

## 为什么需要 backend 边界

官方 Matrix-Game 2.0 仓库提供的是单个长流的交互式推理入口。该入口在一次
`inference` 调用中准备完整条件、维护本地 cache，并在调用内部读取动作；它没有
TeleFuser 所需的持久化 session API，也没有跨 session batching 或状态迁移边界。
因此适配层不会把这段单会话脚本直接伪装成多会话服务。

TeleFuser 侧定义了窄的 `MatrixGameSessionBackend` 接口：

* `start/stop` 管理模型运行时；
* `create_session` 建立一个模型拥有的 session state；
* `step(session, action)` 消费一个规范化的键盘/鼠标动作并返回 `MatrixGameStep`；
* `close_session` 释放模型状态。

实际权重和 Matrix 专用依赖由外部模块提供。设置
`TELEFUSER_MATRIX_GAME_BACKEND_MODULE=your_module` 后，该模块只需要暴露
`create_backend(config)`。这样 Matrix 的 Python、CUDA、FlashAttention 和模型
路径不会进入 TeleFuser 的基础导入路径，后续可以单独替换 backend 而无需修改
LiveKit 控制面。

## 当前能力边界

默认能力声明为 Matrix-Game 2.0、25 FPS、每个逻辑 job 12 帧；当前 backend
默认关闭跨 session batching 和 session state transfer。服务每个 session 只允
许一个 in-flight `step`；执行期间只保留最新的 pending action，从而避免旧控制
在 GPU 队列中无限堆积。输出通过有界 session queue 返回，错误会作为 `error`
chunk 发送并结束该 session。

这意味着第一版适合先用 FIFO/B1 或单 session 验证 Matrix backend 的正确性。要
启用 TeleWorldServer 的迁移和跨 session batch，backend 必须另外实现状态 codec、
batch compatibility key 以及相应的 runtime capability；这些能力不能从官方单会
话脚本安全推断出来。

## 配置示例

```bash
export TELEFUSER_MATRIX_GAME_BACKEND_MODULE=my_matrix_backend
export MATRIX_GAME_ROOT=/path/to/Matrix-Game/Matrix-Game-2
export MATRIX_GAME_CONFIG=/path/to/inference_universal.yaml
export MATRIX_GAME_CHECKPOINT=/path/to/checkpoint
python -m telefuser stream-serve \
  --pipeline-file examples/matrix_game/matrix_game_v2_livekit_service.py \
  --gpu-ids 0
```

`my_matrix_backend.create_backend(config)` 应返回实现
`MatrixGameSessionBackend` 的对象，并暴露
`ModelServingCapabilities` 类型的 `capabilities` 属性。backend 可以在内部
复用官方 Matrix pipeline，但必须自己建立每个 session 的 state 边界；不应让
多个 TeleFuser session 共享官方 pipeline 的可变 cache。
