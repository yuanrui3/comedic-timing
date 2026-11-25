"""Build a self-contained, prediction-blind six-trial owner practice pack."""
from .common import ROOT,read_json,atomic_json,digest
import base64,io,json,random,hashlib
import numpy as np
import soundfile as sf
from scipy.signal import stft
from PIL import Image
from matplotlib import colormaps

def main():
    out=ROOT/'evaluation/human-practice-v1';private=ROOT/'runs/human-practice-v1';out.mkdir(parents=True,exist_ok=True)
    source=read_json(ROOT/'runs/kokoro-diagnostic-v1/manifest.json')['items']
    categories={r['key']:r['category'] for r in read_json(ROOT/'runs/phonetic-evidence-v1/cases.json')}
    rng=random.Random(20261116);keys=[]
    for category,n in [('RMS_endpoint_inside_first_HMM_phone',3),('fragmented_RMS_subsonic_energy',2),('prior_duration_agreement',1)]:
        keys+=rng.sample(sorted(k for k in source if categories[k]==category),n)
    rng.shuffle(keys);trials=[];mapping=[]
    for i,key in enumerate(keys,1):
        item=source[key];path=ROOT/next(iter(item['artifacts']));assert digest(path)==next(iter(item['artifacts'].values()))
        x,sr=sf.read(path);assert x.ndim==1 and sr==24000
        hop=24;frame=120;nfft=256
        f,t,z=stft(x,fs=sr,window='hann',nperseg=frame,noverlap=frame-hop,nfft=nfft,boundary='zeros',padded=True)
        keep=f<=8000;z=z[keep,:];d=20*np.log10(np.maximum(np.abs(z),1e-12));d=np.clip((d-d.max()+50)/50,0,1)
        pixels=(colormaps['magma'](d[::-1])[...,:3]*255).astype(np.uint8)
        buf=io.BytesIO();Image.fromarray(pixels).save(buf,format='PNG')
        padded=np.pad(x,(0,(-len(x))%hop));blocks=padded.reshape(-1,hop)
        trial={'id':f'P{i:02d}','setup':item['setup'],'ending':item['ending'],'duration':len(x)/sr,
            'sampleRate':sr,'audio':'data:audio/wav;base64,'+base64.b64encode(path.read_bytes()).decode(),
            'waveStep':hop/sr,'waveMin':blocks.min(1).round(7).tolist(),'waveMax':blocks.max(1).round(7).tolist(),
            'peak':float(max(abs(x.min()),abs(x.max()))),
            'spectrogram':'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode(),
            'spectrumT0':float(t[0]-hop/(2*sr)),'spectrumT1':float(t[-1]+hop/(2*sr))}
        trials.append(trial);mapping.append({'trial_id':trial['id'],'original_key':key,'category':categories[key],
            'source_audio':str(path.relative_to(ROOT)).replace('\\','/'),'sha256':digest(path)})
    packid=hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()[:16]
    data={'version':'human-practice-v1','packId':packid,'mode':'practice','declaredLanguageAbility':'fluent_or_native',
        'plotSettings':{'window':'Hann','window_ms':5,'hop_ms':1,'nfft':256,'frequency_max_Hz':8000,'dynamic_range_dB':50},'trials':trials}
    template=ROOT/'timing/human_practice_template.html';text=template.read_text(encoding='utf-8')
    text=text.replace('/*PACK_DATA*/null',json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'))
    assert '/*PACK_DATA*/' not in text
    # Scientific predictions and identifying source keys must not enter the client.
    for key in source:assert key not in text
    publicmeta=json.dumps([{k:v for k,v in t.items() if k not in ('audio','spectrogram')} for t in trials],ensure_ascii=False)
    # Random base64 payload characters can coincidentally spell a short model name.
    for forbidden in ('mfa2','mfa3','charsiu','RMS_endpoint_inside','fragmented_RMS_subsonic'):
        assert forbidden not in publicmeta and forbidden not in template.read_text(encoding='utf-8')
    dest=out/'START.html'
    if dest.exists() and dest.read_text(encoding='utf-8')!=text:
        archive=private/'build-archive'/f'{digest(dest)}.html';archive.parent.mkdir(parents=True,exist_ok=True)
        if not archive.exists():archive.write_bytes(dest.read_bytes())
    dest.write_text(text,encoding='utf-8')
    atomic_json(private/'analyst-key.json',{'seed':20261116,'packId':packid,'mode':'practice_only','items':mapping,
        'hide_from_annotator_until_practice_is_complete':True})
    atomic_json(private/'build.json',{'code_sha256':digest(__file__),'template_sha256':digest(template),'html_sha256':digest(dest),
        'bytes':dest.stat().st_size,'trials':6,'source_manifest_sha256':digest(ROOT/'runs/kokoro-diagnostic-v1/manifest.json'),
        'audio_embedded_byte_identically':True,'predicted_timestamps_in_client':False,'user_annotations_collected':False,
        'plot_settings':data['plotSettings']})
    # QA uses its own localStorage namespace and clearly tagged exports.
    qa=text.replace('"mode":"practice"','"mode":"software_qa"',1)
    (private/'QA.html').write_text(qa,encoding='utf-8')
    print('Prepared six anonymous practice trials; HTML bytes:',dest.stat().st_size)

if __name__=='__main__':main()
