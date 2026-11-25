"""Mandarin Charsiu inference adapter with phone-occurrence boundary accounting.

Pinyin normalization follows lingjzhu/charsiu src/processors.py (MIT), pinned in
runs/model-locks/charsiu-code.json. The acoustic network and DTW are unmodified.
We retain DTW occurrence IDs instead of upstream's greedy phone-string word map,
which can lose repeated phone/word identities. No human pronunciation labels.
"""
import re
import numpy as np
from .acoustic import runs
from .corpus import han

CONSONANTS=set('b p m f d t n l g k h j q x zh ch sh r z c s'.split())
TRANSFORMS=dict(zip(
    'ju qu xu jue que xue quan xuan juan qun xun jun yuan yue yun you yan yin wa wo wai weng wang wu yu yi yo ya ye yao yang ying yong yvan yve yvn wei wan wen yv wuen wuo wuang wuan wua wuai zhi chi shi zi ci si'.split(),
    'jv qv xv jve qve xve qvan xvan jvan qvn xvn jvn van ve vn iou ian in ua uo uai ueng uang u v i io ia ie iao iang ing iong van ve vn ui uan un v un uo uang uan ua uai zhiii chiii shiii zii cii sii'.split()))


def separate(syllable):
    stem,tone=syllable[:-1],syllable[-1]
    stem=TRANSFORMS.get(stem,stem)
    if stem in ('er','r'):return ['e'+tone,'rr']
    if stem=='ueng':return ['u'+tone,'eng'+tone]
    if stem=='io':return ['i'+tone,'o'+tone]
    if stem=='ri' and tone=='4':return ['r','iii4']
    onset=stem[:2] if stem[:2] in CONSONANTS else stem[:1] if stem[:1] in CONSONANTS else ''
    rhyme=stem[len(onset):]
    rhyme={'iu':'iou','u:e':'ve','u:':'v'}.get(rhyme,rhyme)
    return ([onset] if onset else [])+[rhyme+tone]


def transcript_phones(g2p,text,setup,vocab):
    pronunciations=g2p(text)
    if len(pronunciations)!=len(text):raise ValueError('G2pM length mismatch')
    phones=[];words=[];word_phone_counts=[]
    for char,pinyin in zip(text,pronunciations):
        if re.fullmatch(r'\w+:?\d',pinyin):
            ph=separate(pinyin)
            words.append(char);word_phone_counts.append(len(ph));phones.extend(ph)
    if words!=han(text):raise ValueError('Incomplete transcript coverage')
    unknown=[p for p in phones if p not in vocab]
    if unknown:raise ValueError(f'Unknown target phones: {unknown}')
    split_words=len(han(setup));split_phones=sum(word_phone_counts[:split_words])
    return {'phones':phones,'phone_ids':[vocab[p] for p in phones],'words':words,
            'word_phone_counts':word_phone_counts,'split_phone_index':split_phones}


def occurrence_alignment(cost,target_ids,sil_idx=1,sil_threshold=4):
    """Original Charsiu silence masking + DTW, retaining occurrence indices."""
    from librosa.sequence import dtw
    raw=np.argmax(cost,axis=1)
    silent=np.zeros(len(raw),dtype=bool)
    for a,b in runs(raw==sil_idx):
        if b-a>=sil_threshold:silent[a:b]=True
    indices=np.flatnonzero(~silent)
    if len(indices)<len(target_ids):raise ValueError('Insufficient non-silence frames for transcript')
    _,path=dtw(C=-cost[indices][:,target_ids],step_sizes_sigma=np.array([[1,1],[1,0]]))
    occurrence=np.full(len(raw),-1,dtype=int)
    for frame,phone in path:occurrence[indices[frame]]=phone
    mapped=occurrence[indices]
    if mapped[0]!=0 or mapped[-1]!=len(target_ids)-1 or not np.all(np.isin(np.diff(mapped),(0,1))):
        raise ValueError('DTW coverage or monotonicity failure')
    return occurrence,silent,raw


def boundary_from_occurrences(occurrence,split,step=.01):
    left=np.flatnonzero((occurrence>=0)&(occurrence<split))
    right=np.flatnonzero(occurrence>=split)
    if not len(left) or not len(right):return {'gap_ms':None,'status':'missing_span'}
    a,b=int(left[-1]+1),int(right[0])
    if a>b or np.any(occurrence[a:b]>=0):return {'gap_ms':None,'status':'invalid_boundary_path'}
    # A silence within either boundary phone is separate from between-word silence.
    interrupted=[]
    for phone in (split-1,split):
        support=np.flatnonzero(occurrence==phone)
        if len(support) and np.any(occurrence[support[0]:support[-1]+1]==-1):interrupted.append(phone)
    return {'start':a*step,'end':b*step,'gap_ms':(b-a)*step*1000,'status':'ok',
            'boundary_phone_interrupted_by_silence':interrupted,'timebase':'upstream 10 ms frame cells, no fitted offset'}


def textless_gap(silent,left,right,step=.01):
    candidates=[];censored=False
    for a,b in runs(silent):
        start,end=a*step,b*step
        if end<left or start>right:continue
        if start<left or end>right or a==0 or b==len(silent):censored=True;continue
        candidates.append({'start':start,'end':end,'gap_ms':(b-a)*step*1000})
    if len(candidates)==1:return {**candidates[0],'status':'ok','candidates':candidates}
    return {'gap_ms':None if candidates or censored else 0.,'status':'ambiguous' if candidates else 'censored' if censored else 'no_silence', 'candidates':candidates}
