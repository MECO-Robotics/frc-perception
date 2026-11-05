import unittest
from pathlib import Path

from scout.samples import MEDIA_ROOT, load_samples


class SampleCatalogTests(unittest.TestCase):
    def test_catalog_entries_are_unique_local_mp4_files(self):
        samples = load_samples()
        self.assertGreaterEqual(len(samples), 4)
        self.assertEqual(len({sample['id'] for sample in samples}), len(samples))
        for sample in samples:
            path = Path(sample['path'])
            self.assertTrue(path.is_file())
            self.assertEqual(path.suffix.lower(), '.mp4')
            self.assertIn(MEDIA_ROOT.resolve(), path.parents)


if __name__ == '__main__':
    unittest.main()
