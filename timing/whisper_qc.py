"""Unprompted local Whisper transcript QC on every original and transfer clip."""
from .common import ROOT,read_json,atomic_json,digest,Manifest
import os,math,time,shutil
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

def main():
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch,transformers
    from transformers import AutoProcessor,AutoModelForSpeechSeq2Seq
    torch.set_num_threads(4);torch.manual_seed(20261115)
    lock=read_json(ROOT/'runs/model-locks/whisper_turbo.json')
    for name,sha in lock['sha256'].items():assert digest(ROOT/'models/whisper_turbo'/name)==sha
    if torch.cuda.mem_get_info()[0]<5*1024**3:raise RuntimeError('Need 5 GiB free GPU; do not disturb other jobs')
    runs=('kokoro-diagnostic-v1','melo-diagnostic-v1')
    generation={'language':'zh','task':'transcribe','do_sample':False,'num_beams':1,
        'temperature':0.0,'return_timestamps':False,'max_new_tokens':224,'condition_on_prev_tokens':False}
    out=ROOT/'runs/whisper-transcript-qc-v1'
    state=Manifest(out/'manifest.json',{'code_sha256':digest(__file__),'model_lock':lock,
        'source_manifests_sha256':{r:digest(ROOT/'runs'/r/'manifest.json') for r in runs},
        'plan_sha256':digest(ROOT/'docs/context-transfer-plan.md'),'generation':generation,
        'torch':torch.__version__,'transformers':transformers.__version__,'dtype':'float16',
        'device':'cuda','batch_size':1,'text_prompt':None,'timing_reference':False})
    shutil.copyfile(ROOT/'docs/context-transfer-plan.md',out/'declared-plan.md')
    processor=AutoProcessor.from_pretrained(str(ROOT/'models/whisper_turbo'),local_files_only=True)
    model,info=AutoModelForSpeechSeq2Seq.from_pretrained(str(ROOT/'models/whisper_turbo'),
        local_files_only=True,torch_dtype=torch.float16,output_loading_info=True)
    assert not any(info[k] for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),info
    model=model.cuda().eval()
    for run in runs:
        source=read_json(ROOT/'runs'/run/'manifest.json')['items']
        for key,item in source.items():
            if state.complete(key):continue
            path=ROOT/next(iter(item['artifacts']));assert digest(path)==next(iter(item['artifacts'].values()))
            x,sr=sf.read(path,dtype='float32');assert x.ndim==1 and len(x)/sr<30
            g=math.gcd(sr,16000);x=resample_poly(x,16000//g,sr//g).astype(np.float32)
            inputs=processor(x,sampling_rate=16000,return_tensors='pt',return_attention_mask=True)
            inputs={k:v.to('cuda',dtype=torch.float16 if k=='input_features' else v.dtype) for k,v in inputs.items()}
            start=time.monotonic();torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():ids=model.generate(**inputs,**generation)
            text=processor.batch_decode(ids,skip_special_tokens=True)[0]
            dest=out/'transcripts'/f'{key}.json'
            atomic_json(dest,{'key':key,'family':'Kokoro' if run.startswith('kokoro') else 'Melo',
                'source_audio':str(path.relative_to(ROOT)).replace('\\','/'),'audio_sha256':digest(path),
                'target_text':item['text'],'transcript':text,'token_ids':ids[0].cpu().tolist(),
                'reached_token_limit':int(ids.shape[-1])>=224,'elapsed_seconds':time.monotonic()-start,
                'peak_gpu_mib':torch.cuda.max_memory_allocated()/1024**2,'generation':generation})
            state.record(key,status='complete',artifacts={str(dest.relative_to(ROOT)).replace('\\','/'):digest(dest)})
            print(len(state.data['items']),key,text,flush=True)
    assert len(state.data['items'])==100

if __name__=='__main__':main()
