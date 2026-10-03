"""Pure stdlib protocol tests plus optional real Office-Home input audit."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task_protocol import (OFFICE31_A2W, OFFICEHOME_PR2RW, TaskProtocol,
                           audit_lists, read_list, visda_protocol, macro_open_set_metrics)


class ProtocolTests(unittest.TestCase):
    def test_macro_unknown_not_pooled(self):
        protocol = TaskProtocol("toy", (0,), (7, 9), 2, "test")
        metrics = macro_open_set_metrics(protocol, [0, 7, 9, 9, 9], [0, 1, 0, 0, 0])
        self.assertEqual(metrics["OS_star"], 1.0)
        self.assertEqual(metrics["UNK"], 0.5)  # pooled unknown accuracy would be .25
        self.assertAlmostEqual(metrics["HOS"], 2 / 3)
        with self.assertRaises(ValueError):
            macro_open_set_metrics(protocol, [0, 7], [0, 1])
        with self.assertRaises(ValueError):
            macro_open_set_metrics(protocol, [0, 7, 9], [0, 1, 2])

    def test_office_labels(self):
        self.assertEqual(OFFICE31_A2W.source_label(9), 9)
        self.assertEqual(OFFICE31_A2W.evaluation_label(20), 10)
        self.assertEqual(OFFICE31_A2W.evaluation_label(30), 10)
        with self.assertRaises(ValueError):
            OFFICE31_A2W.evaluation_label(15)
        with self.assertRaises(ValueError):
            OFFICE31_A2W.source_label(20)
        self.assertEqual(OFFICEHOME_PR2RW.evaluation_label(64), 25)

    def test_visda_not_contiguous(self):
        names = ("aeroplane", "bicycle", "bus", "car", "horse", "knife",
                 "motorcycle", "person", "plant", "skateboard", "train", "truck")
        protocol = visda_protocol(dict(enumerate(names)))
        self.assertEqual(protocol.known_ids, (1, 2, 3, 6, 10, 11))
        self.assertEqual(protocol.source_label(11), 5)
        self.assertEqual(protocol.evaluation_label(0), 6)
        with self.assertRaises(ValueError):
            visda_protocol({0: "bicycle"})

    def test_overlap(self):
        with self.assertRaises(ValueError):
            TaskProtocol("invalid", (0,), (0,), 2, "x")

    def test_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "image with spaces.jpg").touch()
            listing = root / "list.txt"
            listing.write_text("image with spaces.jpg 0\n", encoding="utf-8")
            self.assertEqual(read_list(listing, root), [("image with spaces.jpg", 0)])
            listing.write_text("../outside.jpg 0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_list(listing, root)
            listing.write_text("image with spaces.jpg 0\nimage with spaces.jpg 0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_list(listing, root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--officehome-root", type=Path)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProtocolTests))
    if not result.wasSuccessful():
        sys.exit(1)
    if args.officehome_root:
        root = args.officehome_root
        print(json.dumps(audit_lists(OFFICEHOME_PR2RW,
              root / "product_0-24_train_all.txt", root / "real_world_0-64_test.txt", root), indent=2))
