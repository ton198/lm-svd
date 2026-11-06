"""Shared slot claims and restart progress; no model or array dependencies."""

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
from threading import Lock


# POSIX record locks exclude other processes, not threads in this process.
_local_lock = Lock()


@contextmanager
def state_lock(store_dir: Path):
    # Keep this inode stable: only state.json is atomically replaced.
    with _local_lock, (store_dir / '.state.lock').open('a+') as handle:
        fcntl.lockf(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.lockf(handle, fcntl.LOCK_UN)


def read_store_state(store_dir: Path) -> dict:
    return json.loads((store_dir / 'state.json').read_text(encoding='utf-8'))


def write_store_state(store_dir: Path, state: dict) -> None:
    """Publish a snapshot. Caller must hold state_lock when writers coexist."""
    path = store_dir / 'state.json'
    temporary = store_dir / 'state.json.tmp'
    try:
        temporary.write_text(json.dumps(state, separators=(',', ':')), encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def register_worker(store_dir: Path, worker_id: int) -> None:
    with state_lock(store_dir):
        state = read_store_state(store_dir)
        state['workers'].setdefault(str(worker_id), {'slot': None, 'sequence': 0})
        write_store_state(store_dir, state)


def claim_slot(store_dir: Path, num_slots: int, worker_id: int) -> tuple[int, int] | None:
    with state_lock(store_dir):
        state = read_store_state(store_dir)
        worker = state['workers'][str(worker_id)]
        if worker['slot'] is not None:
            return worker['slot'], worker['sequence']
        occupied = {entry['slot'] for entry in state['workers'].values()}
        published = state['published']
        slot = next((i for i in range(num_slots)
                     if i not in occupied and str(i) not in published), None)
        if slot is None:
            available = [int(i) for i in published if int(i) not in occupied]
            if not available:
                return None
            slot = min(available, key=lambda i: (published[str(i)], i))
        worker['slot'] = slot
        write_store_state(store_dir, state)
        return slot, worker['sequence']


def publish_slot(store_dir: Path, worker_id: int) -> int:
    """Commit an already replaced block and advance this worker's sequence."""
    with state_lock(store_dir):
        state = read_store_state(store_dir)
        worker = state['workers'][str(worker_id)]
        slot = worker['slot']
        if slot is None:
            raise ValueError(f'worker {worker_id} has no claimed slot')
        state['published'][str(slot)] = state['next_commit']
        state['next_commit'] += 1
        worker['sequence'] += 1
        worker['slot'] = None
        write_store_state(store_dir, state)
        return worker['sequence']
