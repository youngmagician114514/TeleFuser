"""Independent TCP workers exchange events without advancing together."""

import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Client

from telefuser.service.livekit.phase_channels import PhaseControlServer


def test_tcp_workers_progress_independently_and_exit_cleanly():
    key = b"phase-test-only"
    server = PhaseControlServer("127.0.0.1", key)
    accept = threading.Thread(target=server.accept, args=(2,))
    accept.start()
    connections = []
    for rank in range(2):
        connection = Client(server.listener.address, authkey=key)
        connection.send({"rank": rank})
        connections.append(connection)
    accept.join(timeout=5)
    assert not accept.is_alive()
    # Worker 1 stays silent while worker 0 emits multiple step completions.
    for step in range(4):
        connections[0].send({"event": "phase_batch", "step": step})
    assert [server.events.get(timeout=5)["step"] for _ in range(4)] == list(range(4))
    server.send(1, {"command": "receive", "transfer_id": "chunk-0"})
    assert connections[1].recv()["transfer_id"] == "chunk-0"
    for connection in connections:
        connection.send({"event": "stopped"})
    assert {server.events.get(timeout=5)["rank"] for _ in range(2)} == {0, 1}
    for connection in connections:
        connection.close()
    server.close()


def test_concurrent_worker_bootstrap_accepts_a_full_cluster_cohort():
    key = b"concurrent-phase-test"
    server = PhaseControlServer("127.0.0.1", key)
    count = 16
    barrier = threading.Barrier(count)
    accept = threading.Thread(target=server.accept, args=(count,))
    accept.start()

    def connect(rank):
        barrier.wait(timeout=5)
        connection = Client(server.listener.address, authkey=key)
        connection.send({"rank": rank})
        return connection

    with ThreadPoolExecutor(max_workers=count) as workers:
        connections = list(workers.map(connect, range(count)))
    accept.join(timeout=5)
    assert not accept.is_alive() and set(server.connections) == set(range(count))
    for rank, connection in enumerate(connections):
        server.send(rank, {"command": "configure", "rank": rank})
        assert connection.recv()["rank"] == rank
        connection.send({"event": "stopped"})
    assert {server.events.get(timeout=5)["rank"] for _ in range(count)} == set(range(count))
    for connection in connections:
        connection.close()
    server.close()
