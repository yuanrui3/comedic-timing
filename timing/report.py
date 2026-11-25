from .common import ROOT, read_json, atomic_json
from .measurement import agreement, rms_track
import argparse
import html
import numpy as np
import soundfile as sf

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='kokoro-diagnostic-v1')
    parser.add_argument('--analysis',default='alignment-v2')
    parser.add_argument('--surprisal',default='surprisal-v1')
    args=parser.parse_args()
    base=ROOT/'runs'/args.run
    synthesis=read_json(base/'manifest.json')
    align=read_json(base/args.analysis/'manifest.json')
    measurements=[read_json(ROOT/next(iter(v['artifacts']))) for v in align['items'].values() if v['status']=='complete']
    clips=[v for v in synthesis['items'].values() if v['status']=='complete']
    retention=read_json(ROOT/'runs'/args.surprisal/'retention.json')
    original=[r for r in retention if r['variant']=='original']
    validation=read_json(base/'silence-validation/summary.json')
    fixtures=read_json(ROOT/'runs/validation/tone-fixtures.json')
    by_voice={v:agreement([m for m in measurements if m['voice']==v]) for v in sorted({m['voice'] for m in measurements})}
    summary={'run':args.run,'n_clips':len(clips),'n_aligned':len(measurements),
             'n_failed_alignment':sum(v['status']=='failed' for v in align['items'].values()),
             'agreement':agreement(measurements),'by_voice':by_voice,
             'original_retained_ids':[r['id'] for r in original if r['retained']],
             'audio_seconds':sum(c['seconds'] for c in clips),
             'synthesis_seconds_including_first_inference':sum(c['elapsed_seconds'] for c in clips),
             'peak_synthesis_gpu_mib':max(c['peak_gpu_mib'] for c in clips),
             'silence_validation_pass':validation['all_pass'],
             'status':'diagnostic_only_gate_1_not_accepted',
             'reason':'No absolute boundary reference; inspect detector disagreement; human semantic review deferred.'}
    atomic_json(base/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=ROOT/'reports/figures'
    figures.mkdir(parents=True,exist_ok=True)
    chosen=[m for m in measurements if m['script_id']=='zh001']
    for m in chosen:
        item=synthesis['items'][m['key']]
        x,sr=sf.read(ROOT/next(iter(item['artifacts'])))
        t=np.arange(len(x))/sr
        center=(m['setup_end']+m['ending_start'])/2
        fig,axes=plt.subplots(2,1,figsize=(11,5),sharex=True,gridspec_kw={'height_ratios':[2,1]})
        axes[0].plot(t,x,color='#425675',linewidth=.5)
        axes[0].axvspan(m['setup_end'],m['ending_start'],color='#eb6a48',alpha=.3,label='CTC character support gap')
        e=m['energy']
        if e['status']=='ok':
            axes[0].axvspan(e['start'],e['end'],color='#16a085',alpha=.4,label='Low-energy interval')
        axes[0].set_title(f'{m["key"]}: CTC {m["ctc_gap_ms"]:.0f} ms; RMS {e["gap_ms"]:.0f} ms (unreviewed)')
        axes[0].set_ylabel('Amplitude')
        axes[0].legend(loc='upper right',fontsize=8)
        times,rms=rms_track(x,sr)
        axes[1].plot(times,20*np.log10(np.maximum(rms,1e-8)),color='#425675')
        axes[1].axhline(20*np.log10(e['threshold_rms']),color='#16a085',linestyle='--',label='RMS threshold')
        axes[1].set_ylabel('RMS dBFS')
        axes[1].set_xlabel('Seconds from start of native waveform')
        axes[1].set_xlim(max(0,center-1.2),min(t[-1],center+1.2))
        for ax in axes:
            ax.axvline(m['setup_end'],color='#d84e31',linestyle=':',linewidth=1)
            ax.axvline(m['ending_start'],color='#d84e31',linestyle=':',linewidth=1)
            ax.grid(alpha=.15)
        fig.tight_layout()
        fig.savefig(figures/f'{m["key"]}.png',dpi=150)
        plt.close(fig)
    differences=np.array([m['ctc_gap_ms']-m['energy']['gap_ms'] for m in measurements])
    averages=np.array([(m['ctc_gap_ms']+m['energy']['gap_ms'])/2 for m in measurements])
    a=summary['agreement']
    fig,ax=plt.subplots(figsize=(8,4))
    ax.scatter(averages,differences,color='#425675')
    ax.axhline(a['bias_ctc_minus_energy_ms'],color='#d84e31',label='Mean difference')
    for limit in a['loa_ms']:
        ax.axhline(limit,color='#d84e31',linestyle='--')
    ax.axhspan(-30,30,color='#16a085',alpha=.12,label='+/-30 ms band')
    ax.set(xlabel='Mean of two measurements (ms)',ylabel='CTC minus RMS (ms)',title='Detector agreement: diagnostic clips, not independent scripts')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures/'detector-agreement.png',dpi=150)
    plt.close(fig)
    doc=[
        '# Local exploration — 2026-11-14', '',
        '**GPU inference works; this is a diagnostic run, not a completed study or an accepted Gate 1.**', '',
        f'Generated {len(clips)} Mandarin clips: ten public-domain joke adaptations and their matched controls, each spoken by two stock voices from **one** Kokoro model. No human annotation, paid inference, cloned voice or downloaded performance media was used. All lengths and punctuation positions match; semantic adequacy remains unreviewed.', '',
        f'Two local Qwen2.5 language models retained **{len(summary["original_retained_ids"])}/10** original pairs under the rule that the joke ending must have greater summed conditional surprisal in both models. Retained IDs: {", ".join(summary["original_retained_ids"])}. These are two sizes of one family, so agreement is not independent-family replication. Audio for all candidates is retained solely for pipeline diagnostics.', '',
        f'The aligner produced {len(measurements)}/{len(clips)} measurements. CTC minus RMS bias was **{a["bias_ctc_minus_energy_ms"]:.1f} ms**, MAE **{a["mae_ms"]:.1f} ms**, and **{a["fraction_within_30ms"]:.1%}** were within 30 ms. Bland–Altman limits were {a["loa_ms"][0]:.1f} to {a["loa_ms"][1]:.1f} ms. Neither detector is ground truth; these clips are paired/repeated observations, not independent validation subjects.', '',
        '| Stock voice | Clips | CTC − RMS bias, ms | MAE, ms | Within 30 ms |',
        '|---|---:|---:|---:|---:|']
    for voice,a_v in by_voice.items():
        doc.append(f'| {voice} | {a_v["n"]} | {a_v["bias_ctc_minus_energy_ms"]:.1f} | {a_v["mae_ms"]:.1f} | {a_v["fraction_within_30ms"]:.1%} |')
    doc += ['', f'RMS tone fixtures: {sum(r["pass_30ms"] for r in fixtures["rows"])}/{len(fixtures["rows"])} within 30 ms. Actual-speech insertion tests: {sum(r["ctc_pass_30ms"] and r["rms_pass_30ms"] for r in validation["rows"])}/{validation["expected_n"]} passed both detectors. Durations were 0, 30, 100, 300, 600 and 1000 ms, on the first joke in each stock voice. These tests measure **incremental gap recovery**, not absolute phonetic-boundary accuracy. The insertion point was selected automatically inside a low-energy interval; no human reference is available.', '',
            f'GPU: RTX 4090 Laptop, 16,376 MiB; PyTorch 2.8.0+cu128, CUDA runtime 12.8, existing driver 581.95. Generated {summary["audio_seconds"]:.1f} s audio in {summary["synthesis_seconds_including_first_inference"]:.1f} s timed inference, including the first-call warmup and excluding model load/download. Peak synthesis allocation: {summary["peak_synthesis_gpu_mib"]:.0f} MiB; this is PyTorch allocation, not whole-device usage. This does not project runtime for other speech models.', '',
            '![Detector agreement](figures/detector-agreement.png)', '',
            'Methods: complete-script Mandarin phonemization followed by one inference, fixed speed 1.0, seed 20261114 and one diagnostic take. CTC Viterbi alignment uses 20 ms frame stride and explicit repeated-label constraints. Character-emission support is only an acoustic-boundary proxy. RMS uses 10 ms frames, 5 ms hop, a threshold 35 dB below the clip’s 95th-percentile RMS, a 1e-6 floor, and an internal interval within 50 ms of the CTC gap midpoint. Thus energy values are independent of alignment emissions, but the search location is alignment-assisted. Terminal padding is excluded.', '',
            'All original setup boundaries are commas in this small set. The diagnostic cannot identify punctuation interactions, and no normalized-punctuation audio or performed-style condition has been run. The two voices share one TTS model. The run does not test 60 retained pairs, English, four to six speech models, multi-take variability, surprise coupling, or funniness. Do not interpret a joke-minus-control estimate before resolving measurement validity.', '',
            'Next automated work: investigate/refine the acoustic boundary definition and threshold sensitivity, enlarge the candidate corpus before applying the fixed LM filter, and add a distinct speech-model family with style control. Human split/boundary review remains deferred. Gate 2 and the main run have not started.', '',
            'Reproduction: see [README](../README.md). Exact model commits and artifact hashes are in [model locks](../runs/model-locks/); raw audio remains local and ignored by Git. [Static audio comparisons](diagnostic-audio.html), [silence tests](../runs/'+args.run+'/silence-validation/summary.json), [retention table](../runs/'+args.surprisal+'/retention.json), [related work](../docs/related.md), and [novelty search](../docs/novelty-search.md).', '']
    (ROOT/'reports/exploration.md').write_text('\n'.join(doc),encoding='utf-8')
    page=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Speech timing diagnostics</title>',
          '<style>body{font:17px system-ui;max-width:1050px;margin:40px auto;padding:0 20px;color:#172638}article{border-top:1px solid #ccd5dd;padding:20px 0}audio{width:100%;max-width:420px}img{width:100%}.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}@media(max-width:700px){.pair{grid-template-columns:1fr}}small{color:#536478}</style>',
          '<h1>Mandarin timing diagnostics</h1><p>AI-generated stock voices, local inference. Provisional script splits; no human annotation. Timing marks are automatic estimates, not validated punchline boundaries. Ten candidate pairs, two stock voices from one model.</p>']
    corpus=read_json(ROOT/'data/scripts/open_zh_diagnostic.json')
    lookup={m['key']:m for m in measurements}
    for row in corpus:
        page.append(f'<article><h2>{row["id"]} · {html.escape(row["source_title"])}</h2><p>{html.escape(row["setup"])}</p>')
        page.append(f'<small>AI adaptation of public-domain <a href="{html.escape(row["source_url"],quote=True)}">historical source</a>. Human review deferred.</small>')
        for voice in synthesis['config']['voices']:
            page.append(f'<h3>{voice}</h3><div class="pair">')
            for kind in ('joke','control'):
                key=f'{row["id"]}_{voice}_{kind}'
                item=synthesis['items'].get(key)
                if not item or item['status']!='complete':
                    continue
                audio=next(iter(item['artifacts']))
                m=lookup.get(key)
                details=f'CTC {m["ctc_gap_ms"]:.0f} ms · RMS {m["energy"]["gap_ms"]:.0f} ms' if m else 'Alignment unavailable'
                page.append(f'<div><b>{kind.title()}</b><p lang="zh">{html.escape(row[kind])}</p><audio controls preload="none" src="../{audio}"></audio><p><small>{details}</small></p></div>')
            page.append('</div>')
        if row['id']=='zh001':
            for m in chosen:
                page.append(f'<img alt="Annotated waveform for {m["key"]}" src="figures/{m["key"]}.png">')
        page.append('</article>')
    page.append('</html>')
    (ROOT/'reports/diagnostic-audio.html').write_text('\n'.join(page),encoding='utf-8')
    print(summary)

if __name__ == '__main__':
    main()
