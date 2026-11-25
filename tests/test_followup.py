import unittest
import numpy as np
from timing.kokoro_trace import punctuation_index
from timing.fsmn_probe import posterior_gap


class FollowupTests(unittest.TestCase):
    def test_punctuation_identity_not_last_comma_heuristic(self):
        self.assertEqual(punctuation_index('甲，乙，', '甲，乙，丙，丁。', 'a, b, c, d.'), 4)
        with self.assertRaises(ValueError):
            punctuation_index('甲，', '甲，乙。', 'a. b.')

    def test_neural_missing_zero_and_ambiguity_differ(self):
        self.assertEqual(posterior_gap(np.zeros(100), .2, .8)['gap_ms'], 0)
        probability = np.zeros(100)
        probability[35:45] = 1
        result = posterior_gap(probability, .2, .8)
        self.assertAlmostEqual(result['gap_ms'], 100)
        self.assertAlmostEqual(result['start'], .3575)
        probability[60:70] = 1
        self.assertIsNone(posterior_gap(probability, .2, .8)['gap_ms'])
        self.assertIsNone(posterior_gap(np.ones(100), .2, .8)['gap_ms'])


if __name__ == '__main__':
    unittest.main()
