"""Download immutable release packages, verify SHA256, and safely extract."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def safe_path(root, name):
    path = Path(name)
    dest = (root/path).resolve()
    if path.is_absolute() or '..' in path.parts or not dest.is_relative_to(root.resolve()):
        raise ValueError('Unsafe archive path')
    return dest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('packages', nargs='+', choices=('data', 'models', 'raw', 'all'))
    p.add_argument('--archive-dir', type=Path, help='Use downloaded files without network')
    args = p.parse_args()
    manifest = json.loads((ROOT/'reproduction-manifest.json').read_text())
    wanted = ('data','models','raw') if 'all' in args.packages else args.packages
    for pack in manifest['packages']:
        if pack['id'] not in wanted:continue
        cache = args.archive_dir or ROOT/'.cache/reproduction'
        cache.mkdir(parents=True, exist_ok=True)
        archive = cache/pack['filename']
        if not archive.exists():
            if args.archive_dir:raise FileNotFoundError(archive)
            partial = archive.with_suffix(archive.suffix+'.part')
            with urlopen(pack['url'], timeout=120) as stream, partial.open('wb') as out:
                shutil.copyfileobj(stream, out)
            if partial.stat().st_size != pack['bytes'] or sha(partial) != pack['sha256']:
                raise ValueError('Downloaded package integrity check failed')
            partial.replace(archive)
        if archive.stat().st_size != pack['bytes'] or sha(archive) != pack['sha256']:
            raise ValueError('Package integrity check failed')
        expected = {i['path']:i for i in manifest['files'] if i['package']==pack['id']}
        with tarfile.open(archive, 'r:gz') as tar:
            members = tar.getmembers()
            if {m.name for m in members} != set(expected) or len(members)!=len(expected):
                raise ValueError('Unexpected or duplicate archive members')
            for member in members:
                dest = safe_path(ROOT, member.name)
                info = expected[member.name]
                if not member.isfile() or member.size!=info['bytes']:
                    raise ValueError('Unexpected archive member type or size')
                if dest.exists():
                    if sha(dest)!=info['sha256']:raise FileExistsError('Changed local file: '+member.name)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                part = dest.with_suffix(dest.suffix+'.part')
                with tar.extractfile(member) as stream, part.open('wb') as out:
                    shutil.copyfileobj(stream, out)
                if sha(part)!=info['sha256']:raise ValueError('Member integrity check failed')
                part.replace(dest)
        print('Verified and extracted', pack['id'], len(expected), 'files', flush=True)


if __name__ == '__main__':main()
