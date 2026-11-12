"""Bounded ordinary file helpers; no CUDA libraries or GPU entry points."""
import hashlib
import json
from pathlib import Path

LIMIT = 16 * 1024 * 1024


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > LIMIT:
        raise ValueError('missing/oversized JSON: ' + str(path))
    return json.loads(path.read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def ref(path, root=None):
    path = Path(path).resolve()
    return {'path': path.relative_to(root).as_posix() if root else str(path),
            'bytes': path.stat().st_size, 'sha256': sha(path)}


def checked(root, row):
    path = (Path(root) / row['path']).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError('artifact outside bundle or missing')
    if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
        raise ValueError('artifact bytes differ: ' + row['path'])
    return path
