"""Add frozen SO101 real evidence to the existing local v1 candidate; no network/device."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'release' / 'v1'

REPORTS = [
    'so101-shared-boundary-attempt09-success-2026-09-27.md',
    'so101-real-success-vs-sim-v1-completion-2026-09-27.md',
    'so101-shared-backend-delivery-closeout-2026-09-27.md',
    'act-so101-shared-boundary-offline-2026-09-27.md',
    'so101-act-attempt01-result-2026-09-27.md',
    'so101-act-attempt02-result-2026-09-27.md',
    'so101-act-attempt03-result-2026-09-27.md',
]
SCRIPTS = [
    'verify_so101_shared_success_offline.py',
    'audit_so101_shared_live_boundary.py',
    'compare_completion_records.py',
]
SOURCE = [
    'completion_record.py', 'execution_contract.py', 'so101_lerobot_backend_adapter.py',
    'legacy_so101_sync_guard.py', 'legacy_bus_audit.py', 'run_record.py',
    'offline_backend_adapters.py', 'move_pot_policy.py', 'smolvla_move_pot.py',
    'chunk_policy.py',
]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def copy(relative):
    source, target = ROOT / relative, RELEASE / relative
    if not source.is_file() or source.is_symlink() or source.suffix in ('.safetensors', '.pt', '.pth', '.ckpt'):
        raise ValueError(f'invalid release source: {relative}')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_tree(relative):
    source = ROOT / relative
    for item in sorted(source.rglob('*')):
        if item.is_symlink():
            raise ValueError(f'symlink in evidence: {item}')
        if item.is_file() and item.suffix != '.pyc':
            copy(item.relative_to(ROOT))


def main():
    if not (RELEASE / 'release-manifest.json').is_file():
        raise FileNotFoundError('existing v1 release candidate required')
    frozen = ROOT / 'artifacts/so101-shared-success-freeze-20260927/manifest.json'
    manifest = json.loads(frozen.read_text())
    for item in manifest['files']:
        relative = Path(item['path'])
        source = ROOT / relative
        if source.stat().st_size != item['size_bytes'] or digest(source) != item['sha256']:
            raise ValueError(f'frozen evidence changed: {relative}')
        copy(relative)
    copy(frozen.relative_to(ROOT))
    for name in REPORTS:
        copy(Path('reports') / name)
    for name in SCRIPTS:
        copy(Path('scripts') / name)
    for name in SOURCE:
        copy(Path('src/cross_backend') / name)
    for number in (1, 2, 3):
        stem = f'attempt{number:02d}'
        copy_tree(Path('artifacts') / f'so101-act-shared-real-pilot-20260927-{stem}')
        for relative in (
            f'artifacts/so101-act-{stem}-ordered-audit-20260927.json',
            f'artifacts/so101-act-{stem}-evaluation-20260927/evaluation.json',
            f'artifacts/so101-act-{stem}-final-scene-20260927/summary.json',
            f'artifacts/so101-act-{stem}-postcheck-registers-20260927/summary.json',
            f'artifacts/so101-act-{stem}-postcheck-scene-20260927/summary.json',
            f'artifacts/so101-act-{stem}-postcheck-scene-20260927/02-follower.png',
            f'artifacts/so101-act-{stem}-postcheck-scene-20260927/02-camera2.png',
        ):
            # The first preflight used a different base name.
            if stem == 'attempt01' and 'final-scene' in relative:
                relative = 'artifacts/so101-act-final-scene-20260927/summary.json'
            copy(Path(relative))
    copy(Path('artifacts/act-so101-shared-boundary-offline-20260927-attempt03/result.json'))
    for name in ('README.md', 'EVIDENCE.md'):
        shutil.copy2(ROOT / 'docs/release-v1' / name, RELEASE / name)
    files = []
    for path in sorted(RELEASE.rglob('*')):
        if path.is_file() and path.name != 'release-manifest.json':
            if path.is_symlink() or path.suffix in ('.safetensors', '.pt', '.pth', '.ckpt', '.pyc'):
                raise ValueError(f'forbidden release file: {path}')
            files.append({'path': path.relative_to(RELEASE).as_posix(),
                          'bytes': path.stat().st_size, 'sha256': digest(path)})
    result = {'release': 'cross-backend-v1-local-candidate',
              'scope': 'simulation v1 plus one SO101/SmolVLA real task success and ACT real execution-boundary development evidence; no formal real-world performance comparison',
              'files': files, 'file_count': len(files),
              'total_bytes': sum(x['bytes'] for x in files),
              'weights_included': False, 'hardware_access': False,
              'archived_hardware_scripts_included': True,
              'archive_created': False, 'remote_published': False}
    (RELEASE / 'release-manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('release','file_count','total_bytes','weights_included','remote_published')}))


if __name__ == '__main__':
    main()
