"""Package a verified Windows distribution and a consistent full database snapshot."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sqlite3
from tempfile import TemporaryDirectory
import zipfile

ROOT = Path(__file__).resolve().parent
APP = 'TimeSeriesAnomalyDiagnosis'


def file_hash(path):
    digest = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def create_bundle(package, database, output, part_size=500 * 1024 * 1024):
    package, database, output = map(lambda path: Path(path).resolve(), (package, database, output))
    if part_size <= 0:
        raise ValueError('part size must be positive')
    if not (package / f'{APP}.exe').is_file() or not database.is_file():
        raise FileNotFoundError('Windows executable and full database are required')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('release output must be empty')
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as directory:
        staging = Path(directory)
        snapshot = staging / 'univariate_ts.db'
        source = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)
        target = sqlite3.connect(snapshot)
        try:
            source.backup(target)
            if target.execute('pragma quick_check').fetchall() != [('ok',)]:
                raise ValueError('database integrity check failed')
            dataset_count = target.execute('select count(*) from datasets').fetchone()[0]
        finally:
            source.close()
            target.close()
        archive_path = staging / 'diagnosis-windows.zip'
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
            for path in sorted(package.rglob('*')):
                relative = path.relative_to(package)
                if relative.parts[0] == 'data' or not path.is_file():
                    continue
                archive.write(path, f'{APP}/{relative.as_posix()}')
            archive.write(snapshot, f'{APP}/data/univariate_ts.db')
        manifest = {'schema': 1, 'dataset_count': dataset_count,
                    'database_sha256': file_hash(snapshot),
                    'archive_sha256': file_hash(archive_path), 'parts': []}
        with archive_path.open('rb') as stream:
            index = 1
            while stream.tell() < archive_path.stat().st_size:
                name = f'diagnosis-windows.zip.{index:03d}'
                destination = output / name
                with destination.open('wb') as part:
                    remaining = part_size
                    while remaining:
                        block = stream.read(min(remaining, 4 * 1024 * 1024))
                        if not block:
                            break
                        part.write(block)
                        remaining -= len(block)
                manifest['parts'].append({'name': name, 'bytes': destination.stat().st_size,
                                          'sha256': file_hash(destination)})
                index += 1
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    for name in ('restore.ps1', 'restore.cmd'):
        shutil.copy2(ROOT / name, output / name)
    shutil.copy2(ROOT.parent / 'docs/windows_release_guide.html', output / 'START_HERE.html')
    return manifest


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    print(json.dumps(create_bundle(options.package, options.database, options.output), indent=2))
