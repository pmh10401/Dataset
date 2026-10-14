import hashlib
from contextlib import closing
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
import zipfile


class ReleaseBundleTest(unittest.TestCase):
    def test_parts_restore_complete_database_without_personal_history(self):
        module = Path(__file__).resolve().parents[1] / 'packaging/release_bundle.py'
        self.assertTrue(module.is_file(), 'release bundle builder is missing')
        spec = importlib.util.spec_from_file_location('release_bundle', module)
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / 'app'
            (package / 'data/cache').mkdir(parents=True)
            (package / 'TimeSeriesAnomalyDiagnosis.exe').write_bytes(b'executable')
            (package / 'data/analysis_history.sqlite3').write_bytes(b'PRIVATE HISTORY')
            (package / 'data/cache/private').write_bytes(b'PRIVATE CACHE')
            database = root / 'source.db'
            with closing(sqlite3.connect(database)) as connection:
                connection.execute('create table datasets (name text)')
                connection.execute("insert into datasets values ('all-data')")
                connection.commit()
            output = root / 'release'
            builder.create_bundle(package, database, output, part_size=200)
            manifest = json.loads((output / 'manifest.json').read_text())
            restored = root / 'restored.zip'
            with restored.open('wb') as stream:
                for part in manifest['parts']:
                    content = (output / part['name']).read_bytes()
                    self.assertLessEqual(len(content), 200)
                    self.assertEqual(hashlib.sha256(content).hexdigest(), part['sha256'])
                    stream.write(content)
            self.assertGreater(len(manifest['parts']), 1)
            with zipfile.ZipFile(restored) as archive:
                self.assertIsNone(archive.testzip())
                names = archive.namelist()
                self.assertNotIn('TimeSeriesAnomalyDiagnosis/data/analysis_history.sqlite3', names)
                self.assertFalse(any('/cache/' in name for name in names))
                archive.extractall(root / 'restored')
            with closing(sqlite3.connect(root / 'restored/TimeSeriesAnomalyDiagnosis/data/univariate_ts.db')) as connection:
                self.assertEqual(connection.execute('select name from datasets').fetchall(), [('all-data',)])
            self.assertTrue((output / 'restore.cmd').is_file())
            if sys.platform == 'win32':
                subprocess.run(['powershell.exe', '-NoProfile', '-File', str(output / 'restore.ps1')], check=True)
                self.assertTrue((output / 'TimeSeriesAnomalyDiagnosis/TimeSeriesAnomalyDiagnosis.exe').is_file())
            with self.assertRaises(FileExistsError):
                builder.create_bundle(package, database, output, part_size=200)


if __name__ == '__main__':
    unittest.main()
