import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReproductionTests(unittest.TestCase):
    def test_runner_and_fixed_scope_exist(self):
        spec = importlib.util.spec_from_file_location('reproduce', ROOT/'scripts/reproduce.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(sum(map(len, module.PERIODS.values())), 10)
        self.assertEqual(module.SEEDS, (42, 43, 44))

    def test_no_path_traversal_in_download(self):
        spec = importlib.util.spec_from_file_location('fetch', ROOT/'scripts/fetch_reproduction.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for path in ('../outside', '/absolute', 'repro/../../outside'):
            with self.assertRaises(ValueError):
                module.safe_path(ROOT, path)
        self.assertEqual(module.safe_path(ROOT, 'repro/data/test.csv'), ROOT/'repro/data/test.csv')


if __name__ == '__main__':
    unittest.main()
