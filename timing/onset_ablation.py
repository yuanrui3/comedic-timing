"""Test whether textless frame classification anticipates deliberately muted onsets."""
from .common import ROOT,read_json,atomic_json,digest,Manifest
from .charsiu_adapter import occurrence_alignment,boundary_from_occurrences,textless_gap
import importlib.util,os
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

def main():
    os.environ['HF_HUB_OFFLINE']='1'
    import torch
    from transformers import Wav2Vec2FeatureExtractor
    cases=[r for r in read_json(ROOT/'runs/phonetic-evidence-v1/cases.json') if r['category']=='RMS_endpoint_inside_first_HMM_phone']
    audit={r['key']:r for r in read_json(ROOT/'runs/pause-audit-v3/measurements.json')}
    original=read_json(ROOT/'runs/kokoro-diagnostic-v1/manifest.json')['items'];out=ROOT/'runs/onset-ablation-v1'
    locks={n:read_json(ROOT/'runs/model-locks'/f'{n}.json') for n in ('charsiu_zh','charsiu-code','charsiu-tokenizer')}
    for n,folder in (('charsiu_zh','charsiu_zh'),('charsiu-code','charsiu_code'),('charsiu-tokenizer','charsiu_tokenizer')):
        for file,sha in locks[n]['sha256'].items():assert digest(ROOT/'models'/folder/file)==sha
    state=Manifest(out/'manifest.json',{'models':locks,'code_sha256':digest(__file__),'target_keys':[r['key'] for r in cases],
        'left_guard_ms':20,'end_extensions_ms':[0,40],'convolution_receptive_field_samples':400,'frame_stride_samples':160,
        'purpose':'context anticipation challenge; no model retraining or stimulus replacement'})
    code=ROOT/'models/charsiu_code/src/models.py';spec=importlib.util.spec_from_file_location('charsiu_upstream_models',code)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    torch.set_num_threads(4);torch.manual_seed(20261115)
    model,info=module.Wav2Vec2ForFrameClassification.from_pretrained(str(ROOT/'models/charsiu_zh'),local_files_only=True,output_loading_info=True)
    assert not any(info[k] for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    model=model.cuda().eval();extractor=Wav2Vec2FeatureExtractor(feature_size=1,sampling_rate=16000,padding_value=0.,do_normalize=True,return_attention_mask=False)
    for c in cases:
        key=c['key'];item=original[key];path=ROOT/next(iter(item['artifacts']));assert digest(path)==next(iter(item['artifacts'].values()))
        native,sr=sf.read(path,dtype='float32');assert sr==24000
        original16=resample_poly(native,2,3)
        prepared=read_json(ROOT/'runs/charsiu-diagnostic-v1/aligned'/f'{key}.json')['prepared']
        baseline=np.load(ROOT/'data/audio/charsiu-diagnostic-v1'/f'{key}.npz')['posterior']
        for extension in (0,40):
            name=f'{key}__mute_plus{extension:02}ms'
            if state.complete(name):continue
            a=round((c['mfa2']['end']-.02)*16000);b=round((c['raw']['end']+extension/1000)*16000)
            x=original16.copy();x[a:b]=0
            # Mute after resampling so the exact-zero span has no filter ringing.
            features=extractor(x,sampling_rate=16000,return_tensors='pt').input_values.cuda()
            with torch.no_grad():cost=torch.softmax(model(features).logits,dim=-1)[0].cpu().numpy()
            frame_start=np.arange(len(cost))*160;interior=(frame_start>=a)&(frame_start+400<=b)
            p=cost[interior];pb=baseline[interior]
            occ,silent,raw=occurrence_alignment(cost,prepared['phone_ids'])
            result={'key':key,'extension_ms':extension,'mute_start':a/16000,'mute_end':b/16000,'n_entirely_zero_RF_frames':len(p),
                'baseline_silence_argmax_fraction':float(np.mean(pb.argmax(1)==1)) if len(p) else None,
                'muted_silence_argmax_fraction':float(np.mean(p.argmax(1)==1)) if len(p) else None,
                'baseline_mean_silence_posterior':float(pb[:,1].mean()) if len(p) else None,
                'muted_mean_silence_posterior':float(p[:,1].mean()) if len(p) else None,
                'forced':boundary_from_occurrences(occ,prepared['split_phone_index']),
                'textless':textless_gap(silent,*audit[key]['bracket']),
                'note':'Convolutional input is zero in selected frames; transformer still sees the entire utterance.'}
            result['forced_onset_before_unmuted_audio_ms']=1000*(b/16000-result['forced']['end'])
            dest=out/'cases'/f'{name}.json';atomic_json(dest,result)
            trace=ROOT/'data/audio/onset-ablation-v1'/f'{name}.npz';trace.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(trace,posterior=cost,occurrence=occ,interior_mask=interior)
            state.record(name,status='complete',artifacts={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in (dest,trace)})
            print(name,result['muted_silence_argmax_fraction'],result['forced_onset_before_unmuted_audio_ms'],flush=True)
    rows=[read_json(p) for p in (out/'cases').glob('*.json')]
    summary={str(e):{'n':sum(r['extension_ms']==e for r in rows),
        'zero_RF_frames':sum(r['n_entirely_zero_RF_frames'] for r in rows if r['extension_ms']==e),
        'mean_muted_silence_fraction':float(np.mean([r['muted_silence_argmax_fraction'] for r in rows if r['extension_ms']==e])),
        'mean_baseline_silence_fraction':float(np.mean([r['baseline_silence_argmax_fraction'] for r in rows if r['extension_ms']==e])),
        'early_onset_over30ms':sum(r['extension_ms']==e and r['forced_onset_before_unmuted_audio_ms']>30+1e-6 for r in rows)} for e in (0,40)}
    atomic_json(out/'summary.json',summary);print(summary)

if __name__=='__main__':main()
