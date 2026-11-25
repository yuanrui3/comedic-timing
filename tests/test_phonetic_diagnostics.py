import unittest
import tempfile
from pathlib import Path
import numpy as np
from timing.charsiu_adapter import separate,occurrence_alignment,boundary_from_occurrences,textless_gap,transcript_phones
from timing.ac_rms import ac_track
from timing.detrended_rms import linear_track
from timing.common import Manifest
from timing.quiet_selector_v2 import select

class PhoneticDiagnosticsTests(unittest.TestCase):
    def test_subminimum_boundary_blip_cannot_censor_search(self):
        centers=(np.arange(100)+1)*.005;energy=np.ones(100)
        energy[37:40]=0  # 15ms run touches the left edge of the bracket.
        self.assertEqual(select(centers,energy,.2,.4)['gap_ms'],0.)
        energy[34:40]=0  # 30ms run really is censored at that edge.
        self.assertIsNone(select(centers,energy,.2,.4)['gap_ms'])

    def test_tuple_parameter_grid_resumes_without_erasing_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'manifest.json'
            first=Manifest(path,{'grid':{'padding':(7,100,500)}})
            first.record('case',status='complete',artifacts={})
            resumed=Manifest(path,{'grid':{'padding':(7,100,500)}})
            self.assertTrue(resumed.complete('case'))
            with self.assertRaises(ValueError):Manifest(path,{'grid':{'padding':(7,100,600)}})

    def test_mandarin_special_syllables(self):
        for syllable,phones in {'yuan2':['van2'],'zhi1':['zh','iii1'],'ri4':['r','iii4'],
                               'lv4':['l','v4'],'liu2':['l','iou2'],'er2':['e2','rr'],'weng1':['u1','eng1']}.items():
            self.assertEqual(separate(syllable),phones)

    def test_repeated_phone_occurrences_survive_silence(self):
        # Two /a/ occurrences separated by /b/, then silence before repeated /a/.
        cost=np.full((12,4),.01);cost[:3,2]=.97;cost[3:5,3]=.97;cost[5:9,1]=.97;cost[9:,2]=.97
        occurrence,silent,_=occurrence_alignment(cost,[2,3,2])
        np.testing.assert_array_equal(occurrence,[0,0,0,1,1,-1,-1,-1,-1,2,2,2])
        result=boundary_from_occurrences(occurrence,2)
        self.assertAlmostEqual(result['start'],.05);self.assertAlmostEqual(result['end'],.09)
        self.assertEqual(result['boundary_phone_interrupted_by_silence'],[])

    def test_interrupted_boundary_phone_is_disclosed(self):
        result=boundary_from_occurrences(np.array([0,0,-1,0,1,1,-1,2]),1)
        self.assertEqual(result['boundary_phone_interrupted_by_silence'],[0])
        self.assertEqual(result['gap_ms'],0)

    def test_textless_ambiguity_is_missing_not_zero(self):
        silent=np.zeros(100,dtype=bool);silent[30:40]=True;silent[60:70]=True
        self.assertIsNone(textless_gap(silent,.2,.8)['gap_ms'])
        self.assertEqual(textless_gap(np.zeros(100,dtype=bool),.2,.8)['gap_ms'],0.)

    def test_uncovered_transcript_is_rejected(self):
        with self.assertRaises(ValueError):transcript_phones(lambda text:['jia3','?'],'甲乙','甲',{'j':1,'ia3':2})

    def test_frame_corrections_obey_their_invariances(self):
        rng=np.random.default_rng(8);sr=24000;x=rng.normal(size=sr)*.02
        np.testing.assert_allclose(ac_track(x,sr)[1],ac_track(x+.3,sr)[1],atol=1e-10)
        ramp=np.linspace(-.2,.4,len(x))
        np.testing.assert_allclose(linear_track(x,sr)[1],linear_track(x+ramp,sr)[1],atol=2e-9)
        # Different implementation: direct least-squares projection in each frame.
        starts=np.arange(0,len(x)-240+1,120);design=np.c_[np.ones(240),np.arange(240)]
        direct=[]
        for start in starts:
            frame=x[start:start+240];beta=np.linalg.lstsq(design,frame,rcond=None)[0]
            direct.append(np.sqrt(np.mean((frame-design@beta)**2)))
        np.testing.assert_allclose(linear_track(x,sr)[1],direct,atol=1e-10)

if __name__=='__main__':unittest.main()
