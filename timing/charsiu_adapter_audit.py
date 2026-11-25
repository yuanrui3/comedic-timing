"""Compare corpus pronunciations/DTW to the pinned upstream implementations."""
from .common import ROOT,read_json,atomic_json,digest
from .charsiu_adapter import separate,occurrence_alignment
import ast,types,re,importlib.util
import numpy as np

def main():
    from g2pM import G2pM
    path=ROOT/'models/charsiu_code/src/processors.py';tree=ast.parse(path.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CharsiuPreprocessor_zh')
    constructor=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    mapping_names={'consonant_list','transform_dict','er_mapping','rhyme_mapping'};obj=types.SimpleNamespace()
    for statement in constructor.body:
        if isinstance(statement,ast.Assign) and isinstance(statement.targets[0],ast.Attribute) and statement.targets[0].attr in mapping_names:
            exec(compile(ast.Module(body=[statement],type_ignores=[]),str(path),'exec'),{'self':obj})
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_separate_syllable');namespace={}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace);fn=namespace['_separate_syllable']
    g2p=G2pM();syllables=set();mismatch=[]
    for item in read_json(ROOT/'runs/kokoro-diagnostic-v1/manifest.json')['items'].values():
        syllables.update(p for p in g2p(item['text']) if re.fullmatch(r'\w+:?\d',p))
    for p in sorted(syllables):
        upstream=list(fn(obj,obj.transform_dict.get(p[:-1],p[:-1])+p[-1]));ours=separate(p)
        if upstream!=ours:mismatch.append({'syllable':p,'upstream':upstream,'adapter':ours})
    upstream_path=ROOT/'models/charsiu_code/src/utils.py';spec=importlib.util.spec_from_file_location('charsiu_upstream_utils',upstream_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);dtw_mismatch=[];comparisons=0
    for path in sorted((ROOT/'runs/charsiu-diagnostic-v1/aligned').glob('*.json')):
        r=read_json(path);saved=np.load(ROOT/'data/audio/charsiu-diagnostic-v1'/f'{r["key"]}.npz');cost=saved['posterior']
        occurrence,silent,_=occurrence_alignment(cost,r['prepared']['phone_ids'])
        expected=np.asarray(module.forced_align(cost[~silent],r['prepared']['phone_ids']))
        if not np.array_equal(expected,occurrence[~silent]):dtw_mismatch.append(r['key'])
        comparisons+=1
    result={'observed_syllables':len(syllables),'pronunciation_mismatches':mismatch,'DTW_cases':comparisons,'DTW_mismatches':dtw_mismatch,
        'upstream_sha256':{p.name:digest(p) for p in (ROOT/'models/charsiu_code/src/processors.py',upstream_path)},
        'adapter_sha256':digest(ROOT/'timing/charsiu_adapter.py'),'difference':'Retained phone occurrence IDs instead of upstream greedy string-to-word map; original DTW paths unchanged.'}
    atomic_json(ROOT/'runs/charsiu-adapter-audit-v1/summary.json',result);print(result)
    assert not mismatch and not dtw_mismatch

if __name__=='__main__':main()
