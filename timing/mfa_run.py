"""Run a pinned phonetic aligner without training, adaptation, or a server."""
from .common import ROOT,read_json,atomic_json,digest,Manifest
from .corpus import han
import argparse
import math
import os
from pathlib import Path
import subprocess
import time
import soundfile as sf
from scipy.signal import resample_poly

def lexicon():
    lex={}
    for line in (ROOT/'models/mfa_zh/mandarin_china_mfa.dict').read_text(encoding='utf-8').splitlines():
        parts=line.split()
        if not parts:
            continue
        # Four dictionary probability fields precede the phones. Keep phones;
        # use the acoustic model's global silence prior, not word-specific priors.
        lex.setdefault(parts[0],[]).append(parts[5:])
    return lex

def segment(text,lex):
    """Exact text coverage with minimum dictionary tokens; deterministic tie rule."""
    text=''.join(han(text))
    paths={len(text):[]}
    for i in range(len(text)-1,-1,-1):
        candidates=[]
        for j in range(i+1,min(len(text),i+30)+1):
            if text[i:j] in lex and j in paths:
                candidates.append([text[i:j]]+paths[j])
        if candidates:
            paths[i]=min(candidates,key=lambda c:(len(c),tuple(c)))
    if 0 not in paths:
        raise ValueError(f'Cannot cover transcript with dictionary: {text}')
    return paths[0]

def environment():
    env=os.environ.copy()
    prefix=ROOT/'.runtime/mfa'
    env['PATH']=os.pathsep.join(str(prefix/p) for p in ('Library/bin','Scripts','.'))+os.pathsep+env['PATH']
    env['MFA_ROOT_DIR']=str(ROOT/'.cache/mfa')
    env['PYTHONIOENCODING']='utf-8'
    env['PYTHONUTF8']='1'
    env['OMP_NUM_THREADS']='2'
    env['MKL_NUM_THREADS']='2'
    env['OPENBLAS_NUM_THREADS']='2'
    return env

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='mfa-diagnostic-v1')
    parser.add_argument('--source',default='kokoro-diagnostic-v1')
    parser.add_argument('--limit',type=int,default=40)
    parser.add_argument('--input-manifest',type=Path)
    args=parser.parse_args()
    source=args.input_manifest or ROOT/'runs'/args.source/'manifest.json'
    source_data=read_json(source)
    items={k:v for k,v in list(source_data['items'].items())[:args.limit] if v['status']=='complete'}
    corpus=ROOT/'.cache/mfa-corpora'/args.run
    out=ROOT/'runs'/args.run
    corpus.mkdir(parents=True,exist_ok=True)
    out.mkdir(parents=True,exist_ok=True)
    spec={'source_sha256':digest(source),'limit':args.limit,'code_sha256':digest(__file__),
          'acoustic_model':read_json(ROOT/'runs/model-locks/mfa_zh.json'),
          'configuration_sha256':digest(ROOT/'config/mfa-alignment.yaml'),
          'alignment':'speaker independent; no fMLLR, training or fine tuning',
          'lexicon':'minimum dictionary tokens within each marked span; equal variant weights; global silence prior',
          'resampling':'scipy.resample_poly, 16000 Hz PCM16 copy',
          'software':'MFA 2.2.17; Kaldi 5.5.1068; CPU inference',
          'runtime_lock':read_json(ROOT/'runs/model-locks/mfa-runtime.json')}
    state=Manifest(out/'manifest.json',spec)
    if state.complete('alignment'):
        print('Resume: all MFA outputs verified; skipping alignment.')
        return
    lex=lexicon()
    used=set()
    prepared={}
    for key,item in items.items():
        setup=segment(item['setup'],lex)
        ending=segment(item['ending'],lex)
        tokens=setup+ending
        used.update(tokens)
        path=ROOT/next(iter(item['artifacts']))
        if digest(path)!=next(iter(item['artifacts'].values())):
            raise ValueError('Input waveform checksum mismatch')
        x,sr=sf.read(path,dtype='float32')
        g=math.gcd(sr,16000)
        x=resample_poly(x,16000//g,sr//g)
        target=corpus/item['voice']/key
        target.parent.mkdir(parents=True,exist_ok=True)
        sf.write(target.with_suffix('.wav'),x,16000,subtype='PCM_16')
        target.with_suffix('.lab').write_text(' '.join(tokens),encoding='utf-8')
        prepared[key]={'script_id':item['script_id'],'voice':item['voice'],'kind':item['kind'],
                       'tokens':tokens,'setup_token_count':len(setup),'setup_han_count':len(han(item['setup'])),
                       'source_audio':str(path.relative_to(ROOT)).replace('\\','/'),
                       'input_wav_sha256':digest(target.with_suffix('.wav'))}
    dictionary=out/'lexicon.dict'
    dictionary.write_text(''.join(w+'\t'+' '.join(p)+'\n' for w in sorted(used) for p in lex[w]),encoding='utf-8')
    atomic_json(out/'inputs.json',prepared)
    command=[str(ROOT/'.runtime/mfa/Scripts/mfa.exe'),'align',str(corpus),str(dictionary),
             str(ROOT/'models/mfa_zh/mandarin_mfa.zip'),str(out/'aligned'),
             '--config_path',str(ROOT/'config/mfa-alignment.yaml'),
             '--temporary_directory',str(ROOT/'.cache/mfa-work'/args.run),
             '--output_format','json','--no_use_mp','--no_use_postgres','--no_textgrid_cleanup','--num_jobs','2']
    state.record('alignment',status='running',command=command)
    start=time.perf_counter()
    with (out/'process.log').open('w',encoding='utf-8') as log:
        result=subprocess.run(command,env=environment(),stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
    files=list((out/'aligned').rglob('*.json'))
    success=result.returncode==0 and len(files)==len(prepared)
    state.record('alignment',status='complete' if success else 'failed',returncode=result.returncode,
                 elapsed_seconds=time.perf_counter()-start,output_count=len(files),expected_count=len(prepared),
                 artifacts={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in files})
    print(f'MFA returncode={result.returncode}, aligned={len(files)}/{len(prepared)}, log={out/"process.log"}')
    if not success:
        print((out/'process.log').read_text(encoding='utf-8')[-6000:])
        raise SystemExit(1)

if __name__=='__main__':
    main()
