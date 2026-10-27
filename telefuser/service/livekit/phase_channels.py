"""CPU command channels and ordered, asynchronous GPU handoffs.

Only startup exchanges use a collective. During serving each worker owns its
compute loop; the coordinator exchanges small Python records over TCP. Tensor
payloads use persistent NCCL streams and reusable target pages.
"""

from __future__ import annotations

import queue
import threading
from collections import defaultdict, deque
from multiprocessing.connection import Connection, Listener
from typing import Any


class PhaseControlServer:
    """One reader per worker, with no barrier between worker events."""

    def __init__(self, host: str, authkey: bytes) -> None:
        # Workers connect together after rendezvous. The default backlog of
        # one can strand a client while authentication handles another peer.
        self.listener = Listener((host, 0), authkey=authkey, backlog=128)
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
        self.connections: dict[int, Connection] = {}
        self.locks: dict[int, threading.Lock] = {}

    def accept(self, world_size: int) -> None:
        for _ in range(world_size):
            connection = self.listener.accept()
            rank = int(connection.recv()["rank"])
            if rank in self.connections:
                raise ValueError(f"Duplicate phase worker rank {rank}")
            self.connections[rank] = connection
            self.locks[rank] = threading.Lock()
            threading.Thread(target=self._read, args=(rank, connection), daemon=True).start()

    def _read(self, rank: int, connection: Connection) -> None:
        try:
            while True:
                event = connection.recv()
                self.events.put(dict(event, rank=rank))
                if event["event"] == "stopped":
                    return
        except (EOFError, OSError) as exc:
            self.events.put({"event": "error", "rank": rank, "message": f"Worker channel closed: {exc}"})

    def send(self, rank: int, command: dict[str, Any]) -> None:
        with self.locks[rank]:
            self.connections[rank].send(command)

    def close(self) -> None:
        for connection in self.connections.values():
            connection.close()
        self.listener.close()


class PairHandoffQueue:
    """Order an untagged NCCL channel per pair, without a global transfer lock.

    The pair is released only after both peers report transfer completion.
    A later transfer cannot reuse an outstanding receive's source or buffers.
    Different pairs may progress concurrently with all compute actors.
    """

    def __init__(self, send: Any) -> None:
        self.send = send
        self.pending: dict[tuple[int, int], deque[dict[str, Any]]] = defaultdict(deque)
        self.active: dict[str, tuple[dict[str, Any], set[int]]] = {}

    def submit(self, record: dict[str, Any]) -> None:
        pair = tuple(sorted((record["source"], record["destination"])))
        self.pending[pair].append(record)
        if len(self.pending[pair]) == 1:
            self._start(record)

    def _start(self, record: dict[str, Any]) -> None:
        self.active[record["transfer_id"]] = (record, set())
        self.send(record["destination"], dict(record, command="receive"))
        self.send(record["source"], dict(record, command="send"))

    def complete(self, transfer_id: str, rank: int) -> None:
        record, completed = self.active[transfer_id]
        if rank not in (record["source"], record["destination"]) or rank in completed:
            raise RuntimeError(f"Invalid transfer completion {transfer_id} from {rank}")
        completed.add(rank)
        if len(completed) < 2:
            return
        pair = tuple(sorted(completed))
        self.pending[pair].popleft()
        del self.active[transfer_id]
        if self.pending[pair]:
            self._start(self.pending[pair][0])
        else:
            del self.pending[pair]
