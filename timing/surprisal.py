from .common import ROOT, read_json, digest, Manifest, atomic_json, source_hash
from .corpus import han
import argparse
import math
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',default='surprisal-v1')
    args = parser.parse_args()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    corpus = ROOT/'data/scripts/open_zh_diagnostic.json'
    out = ROOT/'runs'/args.run
    config = {'corpus_sha256': digest(corpus), 'code_sha256': source_hash(),
              'models': {n:read_json(ROOT/'runs/model-locks'/f'{n}.json') for n in ('surprisal_small','surprisal_large')},
              'torch':torch.__version__, 'dtype':'float32', 'criterion':'ending_sum_bits_higher_in_both',
              'tokenization': 'separately tokenize setup and ending, concatenate token IDs; no cross-boundary token',
              'note':'Two sizes of the same family; not independent model-family replication.'}
    manifest = Manifest(out/'manifest.json',config)
    torch.set_num_threads(4)
    rows = read_json(corpus)
    for name in config['models']:
        tokenizer = AutoTokenizer.from_pretrained(ROOT/'models'/name,local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(ROOT/'models'/name,local_files_only=True,
                                                     torch_dtype=torch.float32).to('cuda').eval()
        for row in rows:
            for variant in ('original','reduced'):
                texts = row if variant=='original' else row['reduced']
                for kind in ('joke','control'):
                    key = f'{row["id"]}_{name}_{variant}_{kind}'
                    if manifest.complete(key):
                        continue
                    context = tokenizer(texts['setup'],add_special_tokens=False)['input_ids']
                    ending = tokenizer(texts[kind],add_special_tokens=False,return_offsets_mapping=True)
                    ids = context+ending['input_ids']
                    if not context or not ending['input_ids']:
                        raise ValueError('Empty context or ending')
                    with torch.inference_mode():
                        logits = model(torch.tensor([ids],device='cuda')).logits[0,len(context)-1:-1].float()
                        lp = logits.log_softmax(-1)
                        bits = (-lp[torch.arange(len(ending['input_ids']),device='cuda'),ending['input_ids']]/math.log(2)).cpu().tolist()
                    # Offsets can overlap for byte-level tokens. Preserve them instead of inventing word alignment.
                    record = {'id':row['id'],'model':name,'kind':kind,'variant':variant,
                              'ending_bits':sum(bits),'bits_per_token':sum(bits)/len(bits),
                              'bits_per_han':sum(bits)/len(han(texts[kind])),
                              'tokens':[{'id':i,'offset':off,'bits':b} for i,off,b in zip(ending['input_ids'],ending['offset_mapping'],bits)]}
                    dest = out/f'{key}.json'
                    atomic_json(dest,record)
                    manifest.record(key,status='complete',artifacts={str(dest.relative_to(ROOT)).replace('\\','/'):digest(dest)})
        del model
        torch.cuda.empty_cache()
        print(f'{name}: completed',flush=True)
    summary=[]
    for row in rows:
        for variant in ('original','reduced'):
            differences={}
            for name in config['models']:
                j=read_json(out/f'{row["id"]}_{name}_{variant}_joke.json')
                c=read_json(out/f'{row["id"]}_{name}_{variant}_control.json')
                differences[name]=j['ending_bits']-c['ending_bits']
            summary.append({'id':row['id'],'variant':variant,'joke_minus_control_bits':differences,
                            'retained':all(x>0 for x in differences.values())})
    atomic_json(out/'retention.json',summary)
    print(f'Retained original pairs: {sum(r["retained"] for r in summary if r["variant"]=="original")}/{len(rows)}')

if __name__ == '__main__':
    main()
