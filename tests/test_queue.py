import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_queue(directory, python_exit=0, busy_once=False):
    directory = Path(directory)
    binaries = directory / 'bin'
    binaries.mkdir()
    calls = directory / 'calls.txt'
    counter = directory / 'busy.txt'
    gpu_script = (
        '#!/usr/bin/env bash\n'
        'if [[ "$*" == *"query-compute-apps"* ]]; then\n'
        '  if [[ "${FAKE_BUSY:-0}" == 1 && ! -f "$FAKE_COUNTER" ]]; then\n'
        '    touch "$FAKE_COUNTER"; echo 1234\n'
        '  fi\n'
        'else\n'
        '  echo 999\n'
        'fi\n')
    python_script = (
        '#!/usr/bin/env bash\n'
        'printf "%s\\n" "$*" >> "$FAKE_CALLS"\n'
        'exit "${FAKE_EXIT:-0}"\n')
    for name, text in [('nvidia-smi', gpu_script), ('python', python_script)]:
        file = binaries / name
        file.write_text(text)
        file.chmod(0o755)
    env = os.environ.copy()
    env.update(
        PATH=f'{binaries}{os.pathsep}{env["PATH"]}',
        PYTHON_BIN=str(binaries / 'python'),
        FAKE_CALLS=str(calls), FAKE_COUNTER=str(counter),
        FAKE_EXIT=str(python_exit), FAKE_BUSY=str(int(busy_once)),
        WAIT_SECONDS='0.01')
    command = [
        'bash', 'scripts/queue.sh', '999', str(directory / 'runs'),
        'configs/visdrone/a1_fcos_p2_p5.py',
        'configs/visdrone/a5_tafe_csta_appearance.py']
    result = subprocess.run(command, cwd=ROOT, env=env, text=True,
                            capture_output=True, timeout=20)
    return result, calls, command, env


def test_queue_waits_and_runs_configs_in_order():
    with tempfile.TemporaryDirectory() as directory:
        result, calls, command, env = _run_queue(directory, busy_once=True)
        assert result.returncode == 0, result.stderr
        rows = calls.read_text().splitlines()
        assert len(rows) == 2
        assert 'a1_fcos_p2_p5.py' in rows[0]
        assert 'a5_tafe_csta_appearance.py' in rows[1]
        assert '--gpu-id 0' in rows[0]
        assert 'GPU 999 is busy' in result.stdout
        again = subprocess.run(command, cwd=ROOT, env=env, text=True,
                               capture_output=True, timeout=20)
        assert again.returncode == 0
        assert calls.read_text().splitlines() == rows


def test_queue_stops_when_training_fails():
    with tempfile.TemporaryDirectory() as directory:
        result, calls, _, _ = _run_queue(directory, python_exit=17)
        assert result.returncode == 17
        assert len(calls.read_text().splitlines()) == 1
        status = (Path(directory) / 'runs/queue_status.tsv').read_text()
        assert '\tfailed\t17' in status
        assert not (Path(directory) / 'runs/a1_fcos_p2_p5/COMPLETED').exists()


def test_queue_refuses_existing_unfinished_directory():
    with tempfile.TemporaryDirectory() as directory:
        work_dir = Path(directory) / 'runs/a1_fcos_p2_p5'
        work_dir.mkdir(parents=True)
        (work_dir / 'existing.txt').write_text('unfinished run')
        result, calls, _, _ = _run_queue(directory)
        assert result.returncode == 2
        assert 'Refusing a non-empty run directory' in result.stderr
        assert not calls.exists()
