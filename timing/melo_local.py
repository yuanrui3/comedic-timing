"""Whole-script native Melo Chinese, pinned local weights and stock speaker only."""
from .common import ROOT,read_json,atomic_json,digest,Manifest
from .corpus import han
import argparse,ast,os,re,sys,time
import numpy as np
import soundfile as sf
import torch

def assets():
    os.environ['HF_HUB_OFFLINE']='1'
    sys.path.insert(0,str(ROOT/'models/melo_code'))
    locks={name:read_json(ROOT/'runs/model-locks'/f'{name}.json') for name in ('melo_zh','melo-code','melo_bert')}
    for name,folder in (('melo_zh','melo_zh'),('melo-code','melo_code'),('melo_bert','melo_bert')):
        for file,sha in locks[name]['sha256'].items():assert digest(ROOT/'models'/folder/file)==sha
    return locks

def reference_frontend():
    # Execute the pinned mixed frontend's pure functions, avoiding imports for
    # unused English/Japanese/Korean dependencies. English branches must not run.
    from melo.text import chinese
    from melo.text.symbols import language_tone_start_map
    path=ROOT/'models/melo_code/melo/text/chinese_mix.py';tree=ast.parse(path.read_text(encoding='utf-8'))
    names={'g2p','_g2p_v2','text_normalize','replace_punctuation'}
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assignments=[node for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'punctuation','rep_map'} for t in node.targets)]
    def no_english(*args,**kwargs):raise ValueError('Unexpected English frontend path')
    ns={'re':re,'cn2an':__import__('cn2an'),'_chinese_g2p':chinese._g2p,'g2p_en':no_english,
        'language_tone_start_map':language_tone_start_map,'tokenizer':None}
    exec(compile(ast.Module(body=assignments+nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns

def prepare():
    locks=assets();torch.set_num_threads(4)
    from transformers import AutoTokenizer,AutoModelForMaskedLM,BertTokenizer
    from melo.text import chinese,cleaned_text_to_sequence
    from melo import commons
    if torch.cuda.mem_get_info()[0]<2*1024**3:raise RuntimeError('Need 2 GiB free for text conditioning')
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/melo_bert',local_files_only=True)
    slow_tokenizer=BertTokenizer.from_pretrained(ROOT/'models/melo_bert',local_files_only=True)
    encoder,info=AutoModelForMaskedLM.from_pretrained(ROOT/'models/melo_bert',local_files_only=True,output_loading_info=True)
    # Pooling and next-sentence classification do not enter token hidden states.
    assert not info['missing_keys'] and not info['mismatched_keys'] and not info['error_msgs'],info
    assert all(k.startswith(('cls.seq_relationship.','bert.pooler.')) for k in info['unexpected_keys']),info
    encoder=encoder.cuda().eval();reference=reference_frontend();config=read_json(ROOT/'models/melo_zh/config.json')
    symbols={s:i for i,s in enumerate(config['symbols'])};out=ROOT/'runs/melo-inputs-v1'
    state=Manifest(out/'manifest.json',{'code_sha256':digest(__file__),'models':locks,'corpus_sha256':digest(ROOT/'data/scripts/open_zh_diagnostic.json'),
        'frontend':'native Chinese branch of ZH_MIX_EN, checked against pinned mixed frontend functions for every input',
        'bert':'native multilingual uncased hidden layer -3; one full text; blank expansion unchanged'})
    atomic_json(out/'bert-load-info.json',info)
    for row in read_json(ROOT/'data/scripts/open_zh_diagnostic.json'):
        for kind in ('joke','control'):
            key=row['id']+'_'+kind
            if state.complete(key):continue
            text=row['setup']+row[kind];assert not re.search('[A-Za-z]',text)
            normalized=chinese.text_normalize(text);assert han(normalized)==han(text)
            phones,tones,word2ph=chinese.g2p(normalized)
            assert normalized==reference['text_normalize'](text)
            assert (phones,tones,word2ph)==reference['g2p'](normalized)
            ids,tone_ids,language_ids=cleaned_text_to_sequence(phones,tones,'ZH_MIX_EN',symbols)
            ids=commons.intersperse(ids,0);tone_ids=commons.intersperse(tone_ids,0);language_ids=commons.intersperse(language_ids,0)
            word2ph=[n*2 for n in word2ph];word2ph[0]+=1
            encoded=tokenizer(normalized,return_tensors='pt');assert tokenizer.unk_token_id not in encoded.input_ids[0]
            assert encoded.input_ids.tolist()==slow_tokenizer(normalized,return_tensors='pt').input_ids.tolist()
            assert len(encoded.input_ids[0])==len(word2ph)
            with torch.inference_mode():hidden=encoder(**{k:v.cuda() for k,v in encoded.items()},output_hidden_states=True).hidden_states[-3][0].cpu()
            features=torch.cat([hidden[i].repeat(n,1) for i,n in enumerate(word2ph)]).T.contiguous().numpy()
            assert features.shape==(768,len(ids))
            path=ROOT/'data/audio/melo-inputs-v1'/f'{key}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,phone=np.array(ids),tone=np.array(tone_ids),language=np.array(language_ids),ja_bert=features)
            state.record(key,status='complete',script_id=row['id'],kind=kind,setup=row['setup'],ending=row[kind],text=text,
                normalized_text=normalized,phones=phones,word2ph=word2ph,frontend_parity=True,fast_slow_tokenizer_parity=True,unknown_tokens=0,
                artifacts={str(path.relative_to(ROOT)).replace('\\','/'):digest(path)})
            print('Prepared native text features',key,len(ids),flush=True)

def synthesize(limit):
    locks=assets();torch.set_num_threads(4)
    if torch.cuda.mem_get_info()[0]<2*1024**3:raise RuntimeError('Need 2 GiB free for Melo diagnostic synthesis')
    from melo.models import SynthesizerTrn
    config=read_json(ROOT/'models/melo_zh/config.json')
    model=SynthesizerTrn(len(config['symbols']),config['data']['filter_length']//2+1,
        config['train']['segment_size']//config['data']['hop_length'],n_speakers=config['data']['n_speakers'],
        num_tones=config['num_tones'],num_languages=config['num_languages'],**config['model']).cuda().eval()
    checkpoint=torch.load(ROOT/'models/melo_zh/checkpoint.pth',map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['model'],strict=True);del checkpoint
    src=ROOT/'runs/melo-inputs-v1/manifest.json';source=read_json(src)['items'];assert len(source)==20
    out=ROOT/'runs/melo-diagnostic-v1';seeds=[20261114,20261115,20261116]
    state=Manifest(out/'manifest.json',{'code_sha256':digest(__file__),'models':locks,'input_features_sha256':digest(src),
        'seeds':seeds,'speaker':'official stock ZH ID 1','speed':1.,'sdp_ratio':.2,'noise_scale':.6,'noise_scale_w':.8,
        'chunks_per_clip':1,'wrapper_added_silence_samples':0,'sample_rate':44100,'purpose':'diagnostic transfer only, not main study'})
    n=0
    for item_key,item in source.items():
        data=np.load(ROOT/next(iter(item['artifacts'])))
        phone=torch.from_numpy(data['phone'][None]).long().cuda();tone=torch.from_numpy(data['tone'][None]).long().cuda()
        language=torch.from_numpy(data['language'][None]).long().cuda();ja_bert=torch.from_numpy(data['ja_bert'][None]).cuda()
        length=torch.tensor([phone.shape[-1]],device='cuda');speaker=torch.tensor([1],device='cuda')
        bert=torch.zeros(1,1024,phone.shape[-1],device='cuda')
        for take,seed in enumerate(seeds,1):
            if n>=limit:print('Execution limit reached; resumable scientific configuration still contains all 60 cases');return
            n+=1;key=f'{item["script_id"]}_melo_ZH_{item["kind"]}_t{take}'
            if state.complete(key):continue
            torch.manual_seed(seed);torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            with torch.inference_mode():audio,attn,mask,latent=model.infer(phone,length,speaker,tone,language,bert,ja_bert,
                sdp_ratio=.2,noise_scale=.6,noise_scale_w=.8,length_scale=1.)
            x=audio[0,0].cpu().numpy();assert x.ndim==1 and np.isfinite(x).all() and np.abs(x).max()>1e-5
            assert len(x)==int(mask.sum())*config['data']['hop_length']
            path=ROOT/'data/audio/melo-diagnostic-v1'/f'{key}.wav';path.parent.mkdir(parents=True,exist_ok=True);sf.write(path,x,44100,subtype='FLOAT')
            state.record(key,status='complete',script_id=item['script_id'],voice='melo_ZH',kind=item['kind'],take=take,seed=seed,
                text=item['text'],setup=item['setup'],ending=item['ending'],source_input=item_key,seconds=len(x)/44100,sample_rate=44100,
                elapsed_seconds=time.perf_counter()-start,peak_gpu_mib=torch.cuda.max_memory_allocated()/1024**2,
                chunks=1,appended_samples=0,artifacts={str(path.relative_to(ROOT)).replace('\\','/'):digest(path)})
            print(key,round(len(x)/44100,2),'seconds',flush=True)
            del audio,attn,mask,latent

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','synthesize']);p.add_argument('--limit',type=int,default=60);args=p.parse_args()
    prepare() if args.action=='prepare' else synthesize(args.limit)
