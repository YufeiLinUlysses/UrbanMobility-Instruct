"""Build a reproducible offline audit from commit-pinned downloaded predictions."""
from pathlib import Path
import hashlib
import json
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / 'data/08_reporting/in_domain_finetuned_audit_2026-09-12'
METRICS = ['rougeL', 'bertscore_f1', 'nli_entailment_score']
ROUGE_TASKS = {'short_answer', 'mc_explanation', 'short_answer_explanation', 'paraphrase'}


def main():
    sources = json.loads((ROOT / 'sources.json').read_text())
    frames = []
    for source in sources:
        path = ROOT / 'sources' / source['model'] / 'predictions.parquet'
        frame = pd.read_parquet(path)
        assert frame['checkpoint_stage'].eq('fine_tuned').all()
        assert frame['evaluation_scope'].eq('in_domain').all()
        frame['source_row'] = range(len(frame))
        frame['audit_id'] = [f"{source['model']}:{i}" for i in range(len(frame))]
        frame['source_url'] = source['url']
        source['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        source['rows'] = len(frame)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    groups, selected, coverage = [], set(), []
    for model in ['ALL'] + sorted(data.model_key.unique().tolist()):
        frame = data if model == 'ALL' else data[data.model_key.eq(model)]
        for metric in METRICS:
            applicable = frame.evaluation_task.isin(ROUGE_TASKS) if metric == 'rougeL' else pd.Series(True, index=frame.index)
            eligible = frame[applicable & frame[metric].notna()]
            missing = frame[applicable & frame[metric].isna()]
            coverage.append(dict(model=model, metric=metric, total=len(frame), scored=len(eligible), missing_applicable=len(missing), not_applicable=int((~applicable).sum())))
            for direction in ['lowest', 'highest', 'missing']:
                ranked = missing.sort_values(['model_key', 'evaluation_task', 'example_id', 'source_row']) if direction == 'missing' else eligible.sort_values([metric, 'model_key', 'evaluation_task', 'example_id', 'source_row'], ascending=[direction == 'lowest', True, True, True, True])
                sample = ranked.head(15)
                ids = sample.audit_id.tolist()
                selected.update(ids)
                groups.append(dict(model=model, metric=metric, direction=direction, available=len(ranked), ids=ids))
                assert len(ids) == min(15, len(ranked))
                if direction != 'missing':
                    values = sample[metric].tolist()
                    assert values == sorted(values, reverse=direction == 'highest')
    records = json.loads(data[data.audit_id.isin(selected)].to_json(orient='records', force_ascii=False))
    for record in records:
        record.update(audit_judgment='', audit_notes='', reference_issue='', metric_issue='')
    payload = dict(sources=sources, coverage=coverage, groups=groups, records=records)
    (ROOT / 'audit_samples.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    readme = f'''# In-domain fine-tuned evaluation audit

Open `audit.html` in a browser. Choose a model, metric, and lowest/highest/missing group. Each nonempty score group contains 15 evaluation rows. ALL ranks across all five models; model-specific groups rank within that model. The same row can occur in multiple groups; annotations are shared by audit ID. Export notes before closing the page.

Sources: {len(data):,} rows from five revision_v5 runs, downloaded from the private prediction repository at commit `{sources[0]['commit']}`. Chat-v2 runs are used for Llama and Qwen; three-task-v1 is used for T5. Earlier revisions and older non-chat runs are excluded. Full original Parquet files, available manifests and metrics are included under `sources/`. Source URLs and file hashes are recorded in `audit_samples.json`.

Selection: 15 lowest and 15 highest **per metric**, both pooled and per model. ROUGE ranking uses ROUGE-L; ROUGE-1 and ROUGE-2 remain visible. BERT uses BERTScore F1; NLI uses the continuous entailment score, not its boolean threshold result. Scores are copied without recomputation or rounding for sorting. Ties are resolved by model, task, example ID, then original zero-based source row. These are evaluation rows, not necessarily distinct questions. Tasks are pooled within each ranking; task labels are shown because scores across task types can have different interpretations.

“nil” is interpreted as NLI; a separate missing-score view also covers null values. Expected ROUGE nulls on mc_answer and answer_verification are not failures and are excluded from missing-score groups. Zeros remain real scores. Missing applicable scores: {sum(c['missing_applicable'] for c in coverage if c['model'] != 'ALL')}. Expected non-applicable ROUGE rows: {sum(c['not_applicable'] for c in coverage if c['model'] != 'ALL')}.

The current evaluation implementation computes semantic metrics against reference_text when present, otherwise reference_answer plus reference_explanation. The downloaded values are retained; prompts and all reference fields are available for auditing metric/target mismatches. Score extremes are sampling aids, not judgments of answer quality. Check factual correctness, completeness, reference quality, and score usefulness independently.

The package contains {len(records)} unique selected model/task rows across {sum(bool(g['ids']) for g in groups)} nonempty ranked lists ({sum(len(g['ids']) for g in groups)} total memberships). JSON preserves full source fields and blank annotation fields. The HTML page makes no network requests. Notes stay in memory until exported; the exported JSON can be imported again.
'''
    (ROOT / 'README.md').write_text(readme, encoding='utf-8')
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>In-domain fine-tuned audit</title>
<style>body{font:16px system-ui;margin:30px auto;max-width:1150px;background:#f4f6fa;color:#182338;padding:0 20px}header{position:sticky;top:0;background:#f4f6fa;padding:12px 0;border-bottom:1px solid #ccd}select,button,input{font:inherit;padding:8px;margin:4px}article{background:white;border:1px solid #ccd;border-radius:10px;padding:22px;margin:20px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 system-ui}textarea{width:98%;min-height:75px;font:inherit}h3{font-size:15px;color:#456}small{overflow-wrap:anywhere} .scores{background:#eef3fa;padding:10px}a{color:#175fa4}</style>
<h1>In-domain fine-tuned audit</h1><p>15 lowest and highest rows per metric. Low/high scores do not establish bad/good quality. ROUGE means ROUGE-L. “Missing” excludes ROUGE values that do not apply to the task.</p>
<header><select id="model"></select><select id="metric"></select><select id="direction"><option>lowest</option><option>highest</option><option>missing</option></select><button id="export">Export notes</button><label>Import notes <input id="import" type="file" accept=".json"></label><div id="status"></div></header><p>Annotations are kept in memory. Export before closing or refreshing. Full source data and selection rules are included in the package.</p><main id="rows"></main>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent), records=Object.fromEntries(data.records.map(r=>[r.audit_id,r]));
const $=id=>document.getElementById(id), esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
for(const m of ['ALL',...data.sources.map(s=>s.model)])$('model').add(new Option(m,m));
for(const m of ['rougeL','bertscore_f1','nli_entailment_score'])$('metric').add(new Option(m,m));
function render(){const g=data.groups.find(g=>g.model===$('model').value&&g.metric===$('metric').value&&g.direction===$('direction').value);
$('status').textContent=`Showing ${g.ids.length} of ${g.available} eligible rows · ${data.records.length} unique audit rows`;
$('rows').innerHTML=g.ids.map((id,i)=>{const r=records[id];return `<article><h2>${i+1}. ${esc(r.model_key)} · ${esc(r.evaluation_task)}</h2><small>${esc(id)} · example ${esc(r.example_id)} · ${esc(r.source_name)} · ${esc(r.Level)} · ${esc(r.region)}</small><p class="scores">${['rouge1','rouge2','rougeL','bertscore_f1','nli_entailment_score'].map(k=>`${k}: <b>${r[k]==null?'N/A':Number(r[k]).toFixed(6)}</b>`).join(' &nbsp; ')}</p>${['prompt','reference_answer','reference_explanation','reference_text','raw_prediction','parsed_answer','review_reasons'].filter(k=>r[k]!=null&&r[k]!=='').map(k=>`<h3>${k.replaceAll('_',' ')}</h3><pre>${esc(r[k])}</pre>`).join('')}<p><a href="${esc(r.source_url)}">Source prediction file</a> · row ${r.source_row}</p><label>Judgment <select data-id="${id}" data-field="audit_judgment">${['','good','bad','mixed','uncertain'].map(v=>`<option ${r.audit_judgment===v?'selected':''}>${v}</option>`).join('')}</select></label>${['audit_notes','reference_issue','metric_issue'].map(k=>`<h3>${k.replaceAll('_',' ')}</h3><textarea data-id="${id}" data-field="${k}">${esc(r[k])}</textarea>`).join('')}</article>`}).join('')||'<p>No missing applicable scores in this group.</p>';}
for(const id of ['model','metric','direction'])$(id).onchange=render;
$('rows').addEventListener('input',e=>{if(e.target.dataset.field)records[e.target.dataset.id][e.target.dataset.field]=e.target.value;});
$('export').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({...data,records:Object.values(records)},null,2)],{type:'application/json'}));a.download='annotated_audit.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};
$('import').onchange=async e=>{try{const input=JSON.parse(await e.target.files[0].text());if(!Array.isArray(input.records))throw Error('Expected audit records');for(const r of input.records)if(records[r.audit_id])for(const k of ['audit_judgment','audit_notes','reference_issue','metric_issue'])records[r.audit_id][k]=String(r[k]??'');render();}catch(err){alert('Cannot import notes: '+err.message);}};
render();</script></html>'''
    (ROOT / 'audit.html').write_text(html.replace('__DATA__', json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c')), encoding='utf-8')
    with zipfile.ZipFile(ROOT.with_suffix('.zip'), 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in ROOT.rglob('*'):
            if path.is_file():
                archive.write(path, str(Path(ROOT.name) / path.relative_to(ROOT)))
    print(json.dumps({'rows': len(data), 'unique_selected': len(records), 'memberships': sum(len(g['ids']) for g in groups), 'missing': [c for c in coverage if c['missing_applicable']], 'output': str(ROOT), 'zip_bytes': ROOT.with_suffix('.zip').stat().st_size}, indent=2))


if __name__ == '__main__':
    main()
