"""Execute the actual optimizer manager class without importing GPU dependencies."""
import ast
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'experiments/rta_multitask_baseline_v1/utilities.py'
node = next(node for node in ast.parse(path.read_text(encoding='utf-8')).body
            if isinstance(node, ast.ClassDef) and node.name == 'OptimizerManager')
namespace = {}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
Manager = namespace['OptimizerManager']


class Optimizer:
    def __init__(self):
        self.steps = 0
    def zero_grad(self):
        pass
    def step(self):
        self.steps += 1


class Tests(unittest.TestCase):
    def test_exception_does_not_step(self):
        optimizer = Optimizer()
        with self.assertRaises(FloatingPointError):
            with Manager([optimizer]):
                raise FloatingPointError('nonfinite')
        self.assertEqual(optimizer.steps, 0)

    def test_normal_path_steps_once(self):
        optimizer = Optimizer()
        with Manager([optimizer]):
            pass
        self.assertEqual(optimizer.steps, 1)


if __name__ == '__main__':
    unittest.main()
