import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from data_pipline import sample_store_state as state


class SampleStoreStateTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = Path(tmp.name)
        state.write_store_state(self.store, {'workers': {}, 'published': {}, 'next_commit': 0})

    def test_claims_fill_empty_slots_then_refresh_oldest_available(self):
        state.register_worker(self.store, 10)
        state.register_worker(self.store, 20)
        self.assertEqual(state.claim_slot(self.store, 3, 10), (0, 0))
        self.assertEqual(state.claim_slot(self.store, 3, 20), (1, 0))
        state.publish_slot(self.store, 20)
        self.assertEqual(state.claim_slot(self.store, 3, 20), (2, 1))
        state.publish_slot(self.store, 10)
        state.publish_slot(self.store, 20)
        # Slot 1 was committed first, despite being claimed second.
        self.assertEqual(state.claim_slot(self.store, 3, 10), (1, 1))
        self.assertEqual(state.claim_slot(self.store, 3, 20), (0, 2))

    def test_registration_resumes_claim_without_advancing_sequence(self):
        state.register_worker(self.store, 17)
        self.assertEqual(state.claim_slot(self.store, 1, 17), (0, 0))
        state.register_worker(self.store, 17)
        self.assertEqual(state.claim_slot(self.store, 1, 17), (0, 0))
        state.publish_slot(self.store, 17)
        self.assertEqual(state.read_store_state(self.store)['workers']['17'],
                         {'slot': None, 'sequence': 1})

    def test_all_claimed_returns_none_and_does_not_steal(self):
        for worker in (0, 1, 2):
            state.register_worker(self.store, worker)
        state.claim_slot(self.store, 2, 0)
        state.claim_slot(self.store, 2, 1)
        self.assertIsNone(state.claim_slot(self.store, 2, 2))
        state.publish_slot(self.store, 0)
        self.assertEqual(state.claim_slot(self.store, 2, 2), (0, 0))

    def test_processes_do_not_duplicate_claims_or_lose_commits(self):
        code = '''
import sys, time
from pathlib import Path
from data_pipline.sample_store_state import register_worker, claim_slot, publish_slot
p, worker = Path(sys.argv[1]), int(sys.argv[2])
register_worker(p, worker)
for _ in range(30):
    while claim_slot(p, 8, worker) is None:
        time.sleep(0.001)
    time.sleep(0.002)
    publish_slot(p, worker)
'''
        processes = [subprocess.Popen([sys.executable, '-c', code, str(self.store), str(i)])
                     for i in range(4)]
        try:
            while any(p.poll() is None for p in processes):
                snapshot = state.read_store_state(self.store)
                slots = [w['slot'] for w in snapshot['workers'].values() if w['slot'] is not None]
                self.assertEqual(len(slots), len(set(slots)))
            for process in processes:
                self.assertEqual(process.wait(timeout=10), 0)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.wait()
        snapshot = state.read_store_state(self.store)
        self.assertEqual(snapshot['next_commit'], 120)
        self.assertEqual(len(snapshot['published']), 8)
        self.assertEqual([snapshot['workers'][str(i)] for i in range(4)],
                         [{'slot': None, 'sequence': 30}] * 4)

    def test_failed_state_write_preserves_previous_snapshot(self):
        state.register_worker(self.store, 1)
        before = json.loads((self.store / 'state.json').read_text())
        # An unserializable transaction must leave the published JSON intact.
        with self.assertRaises(TypeError):
            state.write_store_state(self.store, {'broken': object()})
        self.assertEqual(state.read_store_state(self.store), before)


if __name__ == '__main__':
    unittest.main()
