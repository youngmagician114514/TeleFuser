from __future__ import annotations

import asyncio
import json

import pytest

from telefuser.service.livekit.config import LiveKitServeConfig
from telefuser.service.livekit.distributed_worker_pool import NodeAgentRpcClient
from telefuser.service.livekit.topology import WorkerEndpoint


def test_cluster_config_exposes_global_worker_endpoints() -> None:
    config = LiveKitServeConfig(
        livekit_url="https://livekit",
        livekit_api_key="key",
        livekit_api_secret="secret",
        num_workers=2,
        worker_gpu_map="0;1",
        worker_mode="process-nccl",
        node_id="node-0",
        node_address="node-a",
        node_rank=0,
        cluster_world_size=2,
        cluster_topology="node-0@node-a=0,1;node-1@node-b=0,1",
        nccl_master_address="node-a",
        nccl_world_size=4,
    )
    assert [endpoint.endpoint_id for endpoint in config.cluster_worker_endpoints()] == [
        "node-0/worker-0",
        "node-0/worker-1",
        "node-1/worker-0",
        "node-1/worker-1",
    ]
    assert [endpoint.endpoint_id for endpoint in config.local_cluster_worker_endpoints()] == [
        "node-0/worker-0",
        "node-0/worker-1",
    ]


@pytest.mark.asyncio
async def test_node_agent_rpc_client_health_and_ordering() -> None:
    seen: list[str] = []

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        raw = await reader.readline()
        request = json.loads(raw)
        seen.append(request["method"])
        writer.write(b'{"ok":true,"result":{"status":"ready"}}\n')
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        endpoint = WorkerEndpoint("agent", node_id="node-1", address="127.0.0.1", port=port)
        client = NodeAgentRpcClient(endpoint, secret="shared")
        assert (await client.health())["status"] == "ready"
        assert seen == ["health"]
    finally:
        server.close()
        await server.wait_closed()
