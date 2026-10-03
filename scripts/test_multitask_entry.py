"""Dependency-free tests of actual trainer transforms; does not run training."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task_protocol import OFFICE31_A2W, OFFICEHOME_PR2RW, macro_open_set_metrics

ENTRY = Path(__file__).resolve().parents[1] / 'experiments/rta_multitask_baseline_v1/main.py'
TREE = ast.parse(ENTRY.read_text(encoding='utf-8'))


class FakeTransforms:
    def __getattr__(self, name):
        if name == 'Compose':
            return lambda _: lambda image: image
        return lambda *args: None


class EntryTests(unittest.TestCase):
    def test_actual_transforms(self):
        functions = [node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == 'transform']
        self.assertEqual(len(functions), 3)
        for protocol in (OFFICE31_A2W, OFFICEHOME_PR2RW):
            namespace = dict(protocol=protocol,
                             args=SimpleNamespace(shared_classes=protocol.known_classes,
                                 all_classes=protocol.known_classes + protocol.default_slots),
                             one_hot=lambda size, label: (size, label), transforms=FakeTransforms())
            instantiated = []
            for function in functions:
                exec(compile(ast.Module(body=[function], type_ignores=[]), str(ENTRY), 'exec'), namespace)
                instantiated.append(namespace['transform'])
            source, target_train, target_test = instantiated
            self.assertEqual(source('image', protocol.known_ids[-1], True)[1],
                             (protocol.known_classes + protocol.default_slots, protocol.known_classes - 1))
            target_outputs = [target_train('image', label, True)[1]
                              for label in protocol.known_ids + protocol.unknown_ids]
            self.assertEqual(len(set(target_outputs)), 1)
            self.assertEqual(target_test('image', protocol.unknown_ids[-1], False)[1][1],
                             protocol.unknown_ids[-1])

    def test_original_office31_metric_parity(self):
        labels, predictions = [], []
        original_per_class = []
        for index, label in enumerate(OFFICE31_A2W.known_ids + OFFICE31_A2W.unknown_ids):
            expected = OFFICE31_A2W.evaluation_label(label)
            # Vary class counts so accidental micro averaging cannot pass.
            row = [expected] * (index + 1) + [(expected + 1) % 11]
            labels.extend([label] * len(row))
            predictions.extend(row)
            original_per_class.append((index + 1) / len(row))
        metrics = macro_open_set_metrics(OFFICE31_A2W, labels, predictions)
        self.assertAlmostEqual(metrics['OS_star'], sum(original_per_class[:10]) / 10)
        self.assertAlmostEqual(metrics['UNK'], sum(original_per_class[10:]) / 11)

    def test_target_bookkeeping_not_used(self):
        text = ENTRY.read_text(encoding='utf-8')
        self.assertNotIn("ProbRecorder['lt']", text)
        self.assertNotIn('range(20,31)', text)
        self.assertNotIn('one_hot(31', text)


if __name__ == '__main__':
    unittest.main()
