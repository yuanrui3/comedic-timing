import itertools
import tempfile
import unittest
from pathlib import Path
import numpy as np
from timing.common import Manifest
from timing.measurement import ctc_align, energy_gap, paired_bootstrap

class MeasurementTests(unittest.TestCase):
    def test_ctc_matches_exhaustive_paths(self):
        # Independent exhaustive oracle includes repeated labels requiring a blank.
        rng=np.random.default_rng(9)
        for target in ([1,2],[1,1]):
            lp=np.log(rng.dirichlet([1,1,1],size=6))
            candidates=[]
            for path in itertools.product(range(3),repeat=6):
                collapsed=[a for i,a in enumerate(path) if a and (i==0 or a!=path[i-1])]
                if collapsed==target:
                    candidates.append((sum(lp[t,a] for t,a in enumerate(path)),path))
            _,winner=max(candidates)
            runs=[(a,list(g)) for a,g in itertools.groupby(enumerate(winner),lambda z:z[1])]
            expected=[(g[0][0],g[-1][0]+1) for a,g in runs if a]
            self.assertEqual(ctc_align(lp,target,0),expected)

    def test_missing_is_distinct_from_zero(self):
        self.assertIsNone(energy_gap(np.zeros(16000),16000,.5)['gap_ms'])
        self.assertEqual(energy_gap(np.ones(16000)*.1,16000,.5)['gap_ms'],0)

    def test_pause_does_not_include_file_padding(self):
        x=np.r_[np.zeros(8000),np.ones(8000)*.1]
        self.assertEqual(energy_gap(x,16000,.25)['gap_ms'],0)

    def test_constant_difference_has_no_standardized_effect(self):
        result=paired_bootstrap([20,20,20])
        self.assertIsNone(result['dz'])
        self.assertEqual(result['ci95_ms'],[20,20])

    def test_resume_refuses_changed_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'manifest.json'
            m=Manifest(path,{'seed':1})
            m.record('interrupted',status='running')
            self.assertFalse(Manifest(path,{'seed':1}).complete('interrupted'))
            with self.assertRaises(ValueError):
                Manifest(path,{'seed':2})

if __name__ == '__main__':
    unittest.main()
