import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class SampleStoreSlurmTests(unittest.TestCase):
    def test_script_uses_only_slurm_job_id(self):
        script = Path(__file__).resolve().parents[1] / 'bluehive_run/sample-store-produce.sbatch'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            commands = root / 'bin'
            commands.mkdir()
            venv = root / '.venv/bin'
            venv.mkdir(parents=True)
            (venv / 'activate').write_text('true\n')
            for name in ('nvidia-smi', 'scontrol'):
                path = commands / name
                path.write_text('#!/bin/sh\nexit 0\n')
                path.chmod(0o755)
            producer = commands / 'data-pipline'
            producer.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE"\n')
            producer.chmod(0o755)
            env = dict(os.environ, PATH=str(commands) + os.pathsep + os.environ['PATH'],
                       SLURM_SUBMIT_DIR=str(root), SLURM_JOB_ID='456',
                       SLURM_JOB_NAME='sample-store', CAPTURE=str(root / 'args'),
                       MAX_BLOCKS='2', MAX_GIB='1')
            env.pop('WORKER_ID', None)
            for override, expected in ((None, '456'), ('123', '456')):
                with self.subTest(override=override):
                    if override is not None:
                        env['WORKER_ID'] = override
                    result = subprocess.run(['bash', str(script)], env=env, capture_output=True,
                                            text=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    args = (root / 'args').read_text().splitlines()
                    self.assertEqual(args[args.index('--worker-id') + 1], expected)
                    self.assertEqual(args[args.index('--max-blocks') + 1], '2')
                    self.assertEqual(args[args.index('--max-gib') + 1], '1')
                    self.assertNotIn('--overwrite', args)
                    self.assertNotIn('--shard', args)


if __name__ == '__main__':
    unittest.main()
