"""MFA 3.0.7 with official v3 Mandarin weights/dictionary in an isolated runtime."""
from .common import ROOT,read_json,atomic_json,digest,Manifest
from .mfa_run import segment
from .corpus import han
import argparse,os,subprocess,time,math
import soundfile as sf
from scipy.signal import resample_poly


def environment():
    env=os.environ.copy();prefix=ROOT/'.runtime/mfa3'
    env['PATH']=os.pathsep.join(str(prefix/p) for p in ('Library/bin','Scripts','.'))+os.pathsep+env['PATH']
    env['MFA_ROOT_DIR']=str(ROOT/'.cache/mfa3')
    env['PYTHONIOENCODING']='utf-8';env['PYTHONUTF8']='1'
    for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):env[n]='2'
    return env


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',default='kokoro-diagnostic-v1')
    parser.add_argument('--run',default='mfa3-diagnostic-v1');args=parser.parse_args()
    source_path=ROOT/'runs'/args.source/'manifest.json';source=read_json(source_path)['items']
    model=ROOT/'models/mfa3_zh/mandarin_mfa.zip';dictionary=ROOT/'models/mfa3_zh/mandarin_china_mfa.dict'
    lex={}
    for line in dictionary.read_text(encoding='utf-8').splitlines():
        parts=line.split()
        if parts:lex.setdefault(parts[0],[]).append(parts[5:])
    supplement=ROOT/'runs/mfa3-g2p/supplement.dict'
    for line in supplement.read_text(encoding='utf-8').splitlines():
        parts=line.split()
        if parts:lex.setdefault(parts[0],[]).append(parts[1:])
    out=ROOT/'runs'/args.run;corpus=ROOT/'.cache/mfa-corpora'/args.run
    lock={'acoustic_release':'acoustic-mandarin_mfa-v3.0.0','dictionary_release':'dictionary-mandarin_china_mfa-v3.0.0',
          'license':'CC-BY-4.0','attribution':'Michael McAuliffe and Morgan Sonderegger',
          'sha256':{p.name:digest(p) for p in (model,dictionary,ROOT/'models/mfa3_zh/mandarin_china_g2p.zip')},
          'g2p_supplement_sha256':digest(supplement),'g2p_scope':'automatic pronunciations for 仆人 and 蘸 only'}
    atomic_json(ROOT/'runs/model-locks/mfa3_zh.json',lock)
    state=Manifest(out/'manifest.json',{'source_sha256':digest(source_path),'model':lock,
        'code_sha256':digest(__file__),'segment_code_sha256':digest(ROOT/'timing/mfa_run.py'),
        'configuration_sha256':digest(ROOT/'config/mfa3-alignment.yaml'),
        'runtime':'MFA 3.0.7; Kalpy 0.6.4 CPU; Kaldi 5.5.1112 CPU; hybrid native Conda + Python wheels',
        'speaker_adaptation':False,'fine_tune':False,'normalization_groups':'per stock voice',
        'lexicon':'v3 pronunciations, minimum tokens within marked spans; equal variants; global model silence prior'})
    if state.complete('alignment'):print('Verified completed MFA 3 alignment');return
    prepared={};used=set()
    for key,item in source.items():
        if item['status']!='complete':raise ValueError('Source incomplete')
        setup=segment(item['setup'],lex);ending=segment(item['ending'],lex);tokens=setup+ending;used.update(tokens)
        path=ROOT/next(iter(item['artifacts']))
        if digest(path)!=next(iter(item['artifacts'].values())):raise ValueError('Input checksum mismatch')
        x,sr=sf.read(path,dtype='float32');g=math.gcd(sr,16000);x=resample_poly(x,16000//g,sr//g)
        dest=corpus/item['voice']/key;dest.parent.mkdir(parents=True,exist_ok=True)
        sf.write(dest.with_suffix('.wav'),x,16000,subtype='PCM_16');dest.with_suffix('.lab').write_text(' '.join(tokens),encoding='utf-8')
        prepared[key]={'tokens':tokens,'setup_token_count':len(setup),'setup_han_count':len(han(item['setup'])),
                      'voice':item['voice'],'kind':item['kind'],'script_id':item['script_id'],
                      'source_audio':str(path.relative_to(ROOT)).replace('\\','/'),'input_wav_sha256':digest(dest.with_suffix('.wav'))}
    atomic_json(out/'inputs.json',prepared)
    lex_path=out/'lexicon.dict';lex_path.write_text(''.join(w+'\t'+' '.join(p)+'\n' for w in sorted(used) for p in lex[w]),encoding='utf-8')
    command=[str(ROOT/'.runtime/mfa3/Scripts/mfa.exe'),'align',str(corpus),str(lex_path),str(model),str(out/'aligned'),
             '--config_path',str(ROOT/'config/mfa3-alignment.yaml'),'--temporary_directory',str(ROOT/'.cache/mfa3-work'/args.run),
             '--output_format','json','--no_use_mp','--no_use_postgres','--no_textgrid_cleanup','--num_jobs','2']
    state.record('alignment',status='running',command=command);start=time.perf_counter()
    with (out/'process.log').open('w',encoding='utf-8') as log:result=subprocess.run(command,env=environment(),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    files=list((out/'aligned').rglob('*.json'));success=result.returncode==0 and len(files)==len(source)
    state.record('alignment',status='complete' if success else 'failed',returncode=result.returncode,
        elapsed_seconds=time.perf_counter()-start,output_count=len(files),
        artifacts={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in files+[out/'inputs.json',lex_path,out/'process.log']})
    print(f'MFA3 code={result.returncode}; outputs={len(files)}/{len(source)}')
    if not success:print((out/'process.log').read_text(encoding='utf-8')[-5000:]);raise SystemExit(1)


if __name__=='__main__':main()
