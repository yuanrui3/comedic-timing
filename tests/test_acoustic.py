import unittest
import numpy as np
from timing.acoustic import acoustic_gap,close_interruptions

class AcousticTests(unittest.TestCase):
    def signal(self,pause_ms,sr=24000):
        t=np.arange(sr//3)/sr
        speech=.2*np.sin(2*np.pi*440*t+.3)
        return np.r_[speech,np.zeros(round(sr*pause_ms/1000)),speech]

    def test_known_gap_and_gain_invariance(self):
        for duration in (0,30,100,300,600,1000):
            for gain in (1,.01):
                x=self.signal(duration)*gain
                result=acoustic_gap(x,24000,.2,len(x)/24000-.2,hysteresis=True)
                self.assertIsNotNone(result['gap_ms'])
                self.assertLessEqual(abs(result['gap_ms']-duration),10+1e-6)

    def test_multiple_pauses_are_missing_not_zero_or_nearest_fragment(self):
        x=np.ones(24000)*.2
        x[6000:8400]=0
        x[12000:14400]=0
        r=acoustic_gap(x,24000,.1,.9)
        self.assertEqual(r['status'],'ambiguous_multiple_intervals')
        self.assertIsNone(r['gap_ms'])

    def test_gap_cut_off_by_bracket_is_missing(self):
        x=self.signal(300)
        r=acoustic_gap(x,24000,.4,.8)
        self.assertEqual(r['status'],'censored_at_bracket')
        self.assertIsNone(r['gap_ms'])

    def test_closing_does_not_join_real_speech_or_padding(self):
        x=np.array([0,0,1,1,0,1,1,0,0,0,1,1,0],dtype=bool)
        self.assertEqual(close_interruptions(x,1).astype(int).tolist(),[0,0,1,1,1,1,1,0,0,0,1,1,0])

if __name__=='__main__':
    unittest.main()
