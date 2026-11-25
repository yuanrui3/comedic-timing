"""Reproducible figures and local audio/transcript review for the transfer probe."""
from .common import ROOT,read_json,atomic_json,digest
from pathlib import Path
import html,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    out=ROOT/'reports';figdir=out/'context-transfer-figures';figdir.mkdir(parents=True,exist_ok=True)
    transfer=read_json(ROOT/'runs/transfer-analysis-v1/cases.json')
    qc=read_json(ROOT/'runs/context-transfer-qc-v2/cases.json');qs=read_json(ROOT/'runs/context-transfer-qc-v2/summary.json')
    future=read_json(ROOT/'runs/future-context-analysis-v1/summary.json')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(2,2,figsize=(12.8,9.4),layout='constrained')
    ax=axes[0,0];modes=['mute_g100','mute_g500','reverse_g100','reverse_g500']
    for norm,shift,color in [('ordinary',-.18,'#2865a8'),('fixed_original',.18,'#cc703a')]:
        vals=[future[m][norm]['4']['end']['max_abs_ms'] for m in modes]
        ax.bar(np.arange(4)+shift,vals,.34,label=norm.replace('_',' '),color=color)
    ax.axhline(30,color='#777',linestyle='--',linewidth=1);ax.set(ylim=(0,35),ylabel='Largest onset change (ms)',title='A. Intact boundary, changed future context',xticks=range(4),xticklabels=['Mute\n+100 ms','Mute\n+500 ms','Reverse\n+100 ms','Reverse\n+500 ms'])
    ax.legend(fontsize=9);ax.text(.02,.51,'39 measurable baselines; one short gap\ncrosses the 40 ms minimum in three edits.\n20 ms sensitivity: all 40 stay within 10 ms.',transform=ax.transAxes,fontsize=9)
    ax=axes[0,1]
    for method,color,marker in [('MFA3','#28765b','o'),('Charsiu','#bf6c24','s'),('raw','#7259ad','x')]:
        rows=[r for r in transfer if r['methods'][method]['gap_ms'] is not None]
        ax.scatter([r['methods']['MFA2']['gap_ms'] for r in rows],[r['methods'][method]['gap_ms'] for r in rows],s=27,alpha=.65,label=f'{method} (n={len(rows)})',color=color,marker=marker)
    lim=max(r['methods']['MFA2']['gap_ms'] for r in transfer)+20;ax.plot([0,lim],[0,lim],color='#999',linestyle='--')
    ax.set(xlabel='Original MFA gap (ms)',ylabel='Comparator gap (ms)',title='B. Melo transfer: 60 complete-script clips');ax.legend(fontsize=9)
    ax.text(.03,.94,'Raw RMS: 19 zero / 6 ambiguous',va='top',transform=ax.transAxes,fontsize=9)
    ax=axes[1,0];labels=[];vals=[];colors=[]
    for family in ('Kokoro','Melo'):
        for model,color in [('CTC','#7f8ca3'),('Whisper','#2865a8')]:
            labels.append(f'{family}\n{model}');vals.append(qs[family]['recognizers'][model]['character_error_rate']*100);colors.append(color)
    bars=ax.bar(range(4),vals,color=colors);ax.bar_label(bars,fmt='%.1f%%',padding=3)
    ax.set(xticks=range(4),xticklabels=labels,ylabel='Target character mismatch (%)',ylim=(0,28),title='C. Unprompted local transcription')
    ax.text(.02,.93,'ASR output is a QC signal, not ground truth.',va='top',transform=ax.transAxes,fontsize=9)
    ax=axes[1,1]
    groups=[[r['HP20_gap_level']['ratio'] for r in qc if r['original_category']=='fragmented_RMS_subsonic_energy'],[r['HP20_gap_level']['ratio'] for r in qc if r['family']=='Melo']]
    for i,values in enumerate(groups):
        ax.scatter(np.full(len(values),i)+np.linspace(-.08,.08,len(values)),values,s=24,alpha=.6,color=['#cc703a','#2865a8'][i])
        ax.plot([i-.2,i+.2],[np.median(values)]*2,color='#222',linewidth=2)
    ax.axhline(1,color='#999',linestyle='--');ax.set(xticks=[0,1],xticklabels=['Kokoro: five artifacts','Melo: all 60'],ylabel='RMS after 20 Hz high-pass / original',ylim=(.35,1.12),title='D. Different energy in modeled gap windows')
    ax.text(.02,.95,'Full-waveform filtering; MFA windows trimmed 10 ms.\nExploratory diagnostic, not a promoted correction.',va='top',transform=ax.transAxes,fontsize=9)
    fig.suptitle('Context and model transfer: all 34 original flags remain',fontsize=17,fontweight='bold')
    fig.savefig(figdir/'overview.png',dpi=160);fig.savefig(figdir/'overview.svg');plt.close(fig)
    tr={r['key']:r for r in transfer};cards=[]
    for r in qc:
        k=r['key'];esc=html.escape
        timing='Original flag category: '+str(r['original_category'])
        if k in tr:
            timing=' | '.join(m+': '+(f'{tr[k]["methods"][m]["gap_ms"]:.1f} ms' if tr[k]['methods'][m]['gap_ms'] is not None else tr[k]['methods'][m]['status']) for m in ('MFA2','MFA3','Charsiu','raw','AC','linear'))
        cards.append(f'<article data-family="{r["family"]}"><details><summary>{esc(k)} · Whisper character edits {r["ASR"]["Whisper"]["char_edits"]}/{r["characters"]}</summary><p><b>Target:</b> {esc(r["target"])}</p><p><b>CTC:</b> {esc(r["ASR"]["CTC"]["text"])}</p><p><b>Whisper:</b> {esc(r["ASR"]["Whisper"]["text"])}</p><p>{esc(timing)}</p><audio controls preload="none" src="../{r["audio"]}"></audio></details></article>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Context and transfer diagnostics</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;background:#f6f8fb;color:#182738}h1{line-height:1.15}img{width:100%}article{background:white;border:1px solid #dce2eb;border-radius:8px;padding:12px;margin:10px 0}summary{cursor:pointer;font-weight:650}audio{width:100%}button{padding:8px 16px;margin:5px}p{margin:8px 0}</style><h1>Context and model-transfer diagnostics</h1><p>All 100 recordings are retained. Original 34 flags remain. Transcripts are machine QC; model agreement does not establish true phonetic boundaries.</p><p><a href="context-transfer-investigation.md">Decision brief and methods</a> · <a href="second-aligner-clips.html">Original 40 boundary plots</a></p><img src="context-transfer-figures/overview.png" alt="Four-panel summary of context stability, gap disagreement, ASR mismatch and high-pass diagnostic"><p>Filter: <button onclick="show('all')">All 100</button><button onclick="show('Kokoro')">Original Kokoro 40</button><button onclick="show('Melo')">Melo transfer 60</button></p>'''+''.join(cards)+'''<script>function show(f){document.querySelectorAll('article').forEach(x=>x.hidden=f!=='all'&&x.dataset.family!==f)}</script></html>'''
    (out/'context-transfer-clips.html').write_text(page,encoding='utf-8')
    atomic_json(ROOT/'runs/context-transfer-report-v1/artifacts.json',{str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in (figdir/'overview.png',figdir/'overview.svg',out/'context-transfer-clips.html')})
    print('Saved overview figure and 100-clip local transcript/audio viewer')

if __name__=='__main__':main()
