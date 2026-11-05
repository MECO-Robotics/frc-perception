"""Curated, local video samples available to the dashboard source selector."""
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEDIA_ROOT = PROJECT_ROOT / 'media'
SAMPLES_FILE = MEDIA_ROOT / 'samples.json'


def load_samples():
    with SAMPLES_FILE.open(encoding='utf-8') as source:
        samples = json.load(source)
    if not isinstance(samples, list):
        raise ValueError('Video sample catalog must be a list')
    ids = set()
    for sample in samples:
        if not isinstance(sample, dict) or not all(
                isinstance(sample.get(key), str) and sample[key]
                for key in ('id', 'name', 'file')):
            raise ValueError('Each video sample needs an id, name, and file')
        if sample['id'] in ids:
            raise ValueError(f'Duplicate video sample id: {sample["id"]}')
        ids.add(sample['id'])
        path = (MEDIA_ROOT / sample['file']).resolve()
        if MEDIA_ROOT.resolve() not in path.parents or path.suffix.lower() != '.mp4':
            raise ValueError(f'Invalid video sample path: {sample["file"]}')
        if not path.is_file():
            continue
        sample['path'] = str(path)
    return samples


def find_sample(source):
    try:
        path = Path(source).resolve()
    except (TypeError, OSError):
        return None
    for sample in load_samples():
        if path == Path(sample['path']):
            return sample
    return None
