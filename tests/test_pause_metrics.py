import unittest
from timing.pause_audit import hmm_boundary
from timing.measure_pauses import metric

class PauseMetricTests(unittest.TestCase):
    def make(self,phone='sil',words=None):
        return {'tiers':{'words':{'entries':words or [[0,.4,'甲'],[.4,.7,'<eps>'],[.7,1,'乙']]},
                         'phones':{'entries':[[.4,.7,phone]]}}}

    def test_phone_gap_not_ctc_proxy(self):
        prepared={'tokens':['甲','乙'],'setup_token_count':1}
        hmm=hmm_boundary(self.make(),prepared)
        self.assertAlmostEqual(hmm['gap_ms'],300)
        row={'key':'fixture','script_id':'x','voice':'stock','kind':'joke','hmm':hmm,
             'rms_grid':{'35':{'gap_ms':300,'status':'ok'}},'legacy_ctc':{'gap_ms':550},
             'legacy_rms':{'gap_ms':300},'spectral_grid':{'35':{'gap_ms':300}}}
        result=metric(row)
        self.assertAlmostEqual(result['pause_ms'],300)
        self.assertEqual(result['diagnostics']['ctc_emission_gap_ms_not_silence'],550)

    def test_unknown_noise_is_not_silence(self):
        result=hmm_boundary(self.make('spn'),{'tokens':['甲','乙'],'setup_token_count':1})
        self.assertIsNone(result['gap_ms'])
        self.assertEqual(result['status'],'non_silence_phone_in_gap')

    def test_transcript_mismatch_is_missing(self):
        result=hmm_boundary(self.make(),{'tokens':['甲','丙'],'setup_token_count':1})
        self.assertIsNone(result['gap_ms'])

    def test_touching_words_are_true_zero(self):
        obj=self.make(words=[[0,.4,'甲'],[.4,1,'乙']])
        obj['tiers']['phones']['entries']=[]
        result=hmm_boundary(obj,{'tokens':['甲','乙'],'setup_token_count':1})
        self.assertEqual(result['gap_ms'],0)

if __name__=='__main__':
    unittest.main()
