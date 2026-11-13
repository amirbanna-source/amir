"""Read-only checks of returned campus pilot evidence; prints JSON to stdout.

Usage: python scripts/review_stylegan_pilot.py /private/path/pilot_seed142
Exit 0: technical evidence checks pass. Exit 1: incomplete/failed evidence.
Image quality, clinical validity and memorization always require separate review.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

DATA_SHA = '028216f4780719022b0200463c0e411a5cc18b64b4f4460e55c49a227a287f6d'
SPLIT_SHA = '4f3037e5dc1726e5a534e3a6f155df3e0e0be52229b057659e097483fd0800cd'
SOURCE = 'd72cc7d041b42ec8e806021a205ed9349f87c6a4'


def review(root):
    root = Path(root)
    checks = {}

    def check(name, passed):
        checks[name] = bool(passed)

    def read_json(relative):
        try:
            value = json.loads((root / relative).read_text())
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    state = read_json('pilot_state.json')
    check('state_complete', state.get('status') == 'complete')
    check('training_launched', state.get('training_launched') is True)
    check('dataset_hash', state.get('dataset_sha256') == DATA_SHA)
    check('split_hash', state.get('split_sha256') == SPLIT_SHA)
    check('upstream_pin', state.get('upstream_commit') == SOURCE)
    check('fixed_seed_batch_budget_gpus', all(
        state.get(k) == v for k, v in dict(seed=142, batch=16, kimg=20, gpus=1).items()))
    check('class_mapping', state.get('classes') == {'0': 'benign', '1': 'malignant'})
    check('no_pretrained_weights', 'pretrained_weights' in state and state['pretrained_weights'] is None)
    command = state.get('command', [])
    expected = ['--gpus=1', '--cfg=auto', '--cond=1', '--aug=ada', '--augpipe=bg',
                '--mirror=0', '--batch=16', '--kimg=20', '--snap=1',
                '--metrics=none', '--seed=142', '--workers=2']
    check('recorded_command', isinstance(command, list) and all(x in command for x in expected)
          and not any(str(x).startswith('--resume') for x in command))
    runtime = read_json('runtime/stylegan_gpu_runtime_report.json')
    check('runtime_status', runtime.get('status') == 'gpu_and_targeted_runtime_checks_passed')
    try:
        path_check = runtime['network_check']['tests']['path_length_double_backward']
    except (KeyError, TypeError):
        path_check = False
    check('runtime_path_length_check', path_check is True)
    for name in ['training.log', 'official_dry_run.log', 'pip_freeze.txt',
                 'generate_class0.log', 'generate_class1.log']:
        try:
            check('present_' + name, (root / name).stat().st_size > 0)
        except OSError:
            check('present_' + name, False)
    try:
        dry = (root / 'official_dry_run.log').read_text(errors='replace')
        log = (root / 'training.log').read_text(errors='replace')
    except OSError:
        dry, log = '', ''
    check('official_dry_run_exited', 'Dry run; exiting.' in dry)
    reached = re.findall(r'\bkimg\s+(\d+(?:\.\d+)?)', log)
    reached_kimg = max(map(float, reached), default=None)
    check('logged_budget_reached', reached_kimg is not None and reached_kimg >= 20)
    snapshots = list((root / 'runs').rglob('network-snapshot-000020.pkl'))
    check('one_final_checkpoint', len(snapshots) == 1)
    digest = None
    if len(snapshots) == 1:
        try:
            h = hashlib.sha256()
            with snapshots[0].open('rb') as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(chunk)
            digest = h.hexdigest()
        except OSError:
            pass
    check('final_checkpoint_hash', digest is not None and digest == state.get('final_checkpoint_sha256'))
    # Never unpickle the checkpoint or execute returned code.
    try:
        from PIL import Image
    except ImportError:
        Image = None
    check('pillow_available', Image is not None)

    def image_ok(path, size):
        if Image is None:
            return False
        try:
            with Image.open(path) as im:
                if im.size != size or im.mode != 'RGB':
                    return False
                im.load()
            return True
        except (OSError, ValueError):
            return False

    for target in (0, 1):
        folder = root / f'samples_class{target}'
        expected_names = {f'seed{seed:04d}.png' for seed in range(16)}
        check(f'class{target}_exact_seed_set', {p.name for p in folder.glob('seed*.png')} == expected_names)
        check(f'class{target}_images_decode', all(image_ok(folder / n, (128, 128)) for n in expected_names))
        check(f'class{target}_grid_decodes', image_ok(root / f'grid_class{target}_final.png', (512, 512)))
    passed = all(checks.values())
    return dict(technical_evidence_complete=passed, checks=checks,
                failed_checks=[k for k, v in checks.items() if not v],
                logged_kimg=reached_kimg, final_checkpoint_sha256=digest,
                interpretation='Technical file checks only; review the full logs and images.',
                image_quality='not assessed', diversity='not assessed',
                memorization='not assessed', classifier_benefit='not evaluated')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    result = review(args.results)
    print(json.dumps(result, indent=2))
    return 0 if result['technical_evidence_complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
