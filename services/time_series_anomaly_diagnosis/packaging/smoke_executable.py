"""Exercise the packaged process, not the source server, using synthetic waveforms."""
from argparse import ArgumentParser
from contextlib import closing
import json
import math
from pathlib import Path
import socket
import sqlite3
import subprocess
from tempfile import TemporaryDirectory
import time
from urllib.request import Request, urlopen

from services.time_series_anomaly_diagnosis.tests.test_dataset_importer import make_source_database, values_blob


def smoke(executable, report):
    with TemporaryDirectory() as directory:
        root = Path(directory)
        raw, data = root / 'synthetic.db', root / 'data'
        make_source_database(raw, name='Trace_normal_2', train_labels=('7',) * 9 + ('0',) * 3,
                             test_labels=('7', '0', '7', '0'),
                             blobs=tuple(values_blob(*(math.sin(p * (.12 + i * .013))
                                        + .15 * math.cos(p * .37 + i * .23) for p in range(64)))
                                         for i in range(16)))
        with closing(sqlite3.connect(raw)) as connection:
            connection.execute('update datasets set series_length=64')
            connection.commit()
        subprocess.run([str(executable), '--import-database', str(raw), '--data-dir', str(data)],
                       check=True, timeout=120)
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
        base = f'http://127.0.0.1:{port}'

        def request(path, payload=None):
            with urlopen(Request(base + path, data=json.dumps(payload).encode() if payload else None,
                                 headers={'Content-Type': 'application/json'}), timeout=20) as response:
                return json.load(response)

        completed = None
        for cycle in range(2):
            with (root / f'process-{cycle}.log').open('w') as log:
                process = subprocess.Popen([str(executable), '--data-dir', str(data), '--host', '127.0.0.1',
                                            '--port', str(port), '--public-host', '127.0.0.1'], stdout=log, stderr=log)
                try:
                    deadline = time.monotonic() + 120
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError((root / f'process-{cycle}.log').read_text(errors='replace'))
                        try:
                            assert request('/api/status')['status'] == 'ready'
                            break
                        except OSError:
                            if time.monotonic() > deadline:
                                raise TimeoutError('packaged service did not become ready')
                            time.sleep(1)
                    assert len(request('/api/datasets')) == 1
                    with urlopen(base) as page:
                        assert b'available-only' in page.read()
                    if cycle == 0:
                        run = request('/api/analysis-runs', {'dataset_name': 'Trace_normal_2'})
                        deadline = time.monotonic() + 240
                        while True:
                            completed = request('/api/analysis-runs/' + run['run_id'])
                            if completed['status'] not in ('queued', 'running'):
                                break
                            if time.monotonic() > deadline:
                                raise TimeoutError('packaged analysis did not finish')
                            time.sleep(1)
                        assert completed['status'] == 'completed', completed
                        assert len(completed['result']['rows']) == 4
                    else:
                        restored = request('/api/analysis-runs/' + completed['run_id'])
                        assert restored['result'] == completed['result']
                finally:
                    process.terminate()
                    process.wait(timeout=30)
        report.write_text(json.dumps({'passed': True, 'input': 'synthetic', 'rows': 4,
                                     'restart_preserved_result': True}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    options = parser.parse_args()
    smoke(options.executable.resolve(), options.report.resolve())
