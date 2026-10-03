import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from assemble_officehome_colab import assemble


class AssemblyTests(unittest.TestCase):
    def fixture(self, root):
        parts = [b'abc', b'defg']
        records = []
        for index, data in enumerate(parts):
            name = f'part{index}'
            (root / name).write_bytes(data)
            records.append(dict(file=name, index=index, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        manifest = dict(file='real_world_0-64_test.zip', bytes=7,
                        sha256=hashlib.sha256(b'abcdefg').hexdigest(), parts=records)
        (root / 'officehome-parts-v1.json').write_text(json.dumps(manifest))

    def test_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            assemble(root)
            self.assertEqual((root / 'real_world_0-64_test.zip').read_bytes(), b'abcdefg')
            with self.assertRaises(FileExistsError):
                assemble(root)

    def test_corruption_stops_before_assembly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / 'part1').write_bytes(b'xxxx')
            with self.assertRaises(AssertionError):
                assemble(root)
            self.assertFalse((root / 'real_world_0-64_test.zip').exists())


if __name__ == '__main__':
    unittest.main()
