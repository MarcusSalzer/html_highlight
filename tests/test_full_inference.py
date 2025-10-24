"""Test easy examples. Inference should never fail on these"""

import sys
import unittest

sys.path.append(".")

from src import inference, text_process

infer = inference.Inference("model_inference")


class TestShort(unittest.TestCase):
    def test_assign(self):
        text = "x = 3"
        tags_pred = infer.run(*text_process.process(text))
        self.assertListEqual(["va", "ws", "opas", "ws", "nu"], tags_pred)

    def test_loop_fun(self):
        text = "for x in range(3):\n    print(x)"
        tp = infer.run(*text_process.process(text))
        self.assertEqual("kwfl", tp[0])  # for
        self.assertEqual("va", tp[2])  # x
        self.assertEqual("fnfr", tp[6])  # range
        self.assertEqual("fnfr", tp[13])  # print


if __name__ == "__main__":
    unittest.main()
