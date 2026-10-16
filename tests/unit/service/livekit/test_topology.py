import pytest

from telefuser.service.livekit.config import LiveKitServeConfig
from telefuser.service.livekit.motivation_scheduler import GpuSchedulingState
from telefuser.service.livekit.session_state_transfer import SessionStateTransferRequest
from telefuser.service.livekit.topology import ClusterTopology, NodeAgent, WorkerEndpoint, transport_scope


def test_two_node_topology_keeps_gpu_ordinals_local() -> None:
    topology = ClusterTopology.from_spec(
        "node-0@h100-a=0,1,2,3,4,5,6,7;node-1@h100-b=0,1,2,3,4,5,6,7"
    )

    assert len(topology.nodes) == 2
    assert topology.node("node-1").gpu_ids == tuple(str(index) for index in range(8))
    assert topology.node_for_gpu("0", node_id="node-1").address == "h100-b"

    local = WorkerEndpoint("worker-0", node_id="node-0", gpu_ids=("0",), address="h100-a")
    remote = WorkerEndpoint("worker-0", node_id="node-1", gpu_ids=("0",), address="h100-b")
    assert local.endpoint_id != remote.endpoint_id
    assert transport_scope(local, remote) == "inter_node"


def test_endpoint_groups_use_existing_worker_ids() -> None:
    topology = ClusterTopology((NodeAgent("node-0", "h100-a", ("0", "1")),))

    endpoints = topology.endpoints_for_worker_groups(
        node_id="node-0",
        worker_gpu_groups=(("0",), ("1",)),
        worker_ids=("worker-a", "worker-b"),
    )

    assert [endpoint.endpoint_id for endpoint in endpoints] == ["node-0/worker-a", "node-0/worker-b"]
    assert all(endpoint.address == "h100-a" for endpoint in endpoints)


def test_config_exposes_two_node_topology_without_changing_worker_map() -> None:
    config = LiveKitServeConfig(
        num_workers=8,
        worker_gpu_map="0;1;2;3;4;5;6;7",
        node_id="node-1",
        node_address="h100-b",
        node_rank=1,
        cluster_world_size=2,
        worker_mode="process-nccl",
        cluster_topology="node-0@h100-a=0,1,2,3,4,5,6,7;node-1@h100-b=0,1,2,3,4,5,6,7",
    )

    assert config.worker_gpu_groups() == [[str(index)] for index in range(8)]
    assert config.topology().node("node-1").address == "h100-b"
    assert config.effective_node_address() == "h100-b"
    assert config.nccl_rendezvous(local_worker_count=8) == ("h100-a", 29500, 16, 8)


def test_single_node_nccl_rendezvous_keeps_ephemeral_port_contract() -> None:
    config = LiveKitServeConfig(num_workers=8, worker_gpu_map="0;1;2;3;4;5;6;7")

    assert config.nccl_rendezvous(local_worker_count=8) == ("127.0.0.1", 0, 8, 0)


def test_transfer_request_defaults_to_single_node_and_marks_remote_transfers() -> None:
    local = SessionStateTransferRequest("session-a", "gpu-0", "gpu-1", 0.0, 0.1)
    remote = SessionStateTransferRequest(
        "session-a",
        "gpu-0",
        "gpu-1",
        0.0,
        0.1,
        source_node_id="node-0",
        target_node_id="node-1",
    )

    assert local.transport_scope == "intra_node"
    assert local.crosses_node is False
    assert remote.transport_scope == "inter_node"
    assert remote.crosses_node is True


def test_gpu_state_retains_node_metadata_when_registered_and_updated() -> None:
    from telefuser.service.livekit.motivation_scheduler import (
        MotivationProfile,
        MotivationScheduler,
        StaticMotivationProfileTable,
    )

    scheduler = MotivationScheduler(
        StaticMotivationProfileTable([MotivationProfile(1, "b1", 0.1, 1.0, 1.0)])
    )
    scheduler.add_gpu(GpuSchedulingState("gpu-0", node_id="node-1", endpoint_address="h100-b"))
    scheduler.update_gpu("gpu-0", available=False)

    state = scheduler.gpus()[0]
    assert state.node_id == "node-1"
    assert state.endpoint_address == "h100-b"
    assert state.available is False
