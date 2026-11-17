"""Build a reproducible audit package from the five final in-domain runs."""

from pathlib import Path
import hashlib
import json

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/08_reporting/in_domain_finetuned_audit_prompt_cleaned_2026-09-17'
EVIDENCE = ROOT / 'data/08_reporting/manuscript_refresh_2026-09-17/evidence.json'
REVIEWED = ROOT / 'scripts/prompt_cleaned_reviewed_cases.json'
REPO = 'UlyssesLynne/urban-planning-llm-predictions-v3'
METRICS = ('rougeL', 'bertscore_f1', 'nli_entailment_score')


def main() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding='utf-8'))
    revision = evidence['prediction_repo_commit']
    frames = []
    sources = []
    for model, (version, run) in evidence['runs'].items():
        path = f'evaluations/{version}/{model}/fine_tuned/in_domain/{run}-finetuned-in-domain/predictions.parquet'
        local = Path(hf_hub_download(REPO, path, repo_type='dataset', revision=revision))
        frame = pd.read_parquet(local)
        assert len(frame) == 5552 and frame.model_key.eq(model).all()
        assert frame.checkpoint_stage.eq('fine_tuned').all()
        assert frame.evaluation_scope.eq('in_domain').all()
        frame['source_row'] = range(len(frame))
        frame['audit_id'] = model + ':' + frame.source_row.astype(str)
        frame['source_path'] = path
        frames.append(frame)
        sources.append({'model': model, 'path': path, 'sha256': hashlib.sha256(local.read_bytes()).hexdigest(), 'rows': len(frame)})

    data = pd.concat(frames, ignore_index=True)
    selected = set()
    groups = []
    for model in ['ALL', *evidence['runs']]:
        model_data = data if model == 'ALL' else data[data.model_key.eq(model)]
        for metric in METRICS:
            for task in sorted(model_data.evaluation_task.unique()):
                task_data = model_data[model_data.evaluation_task.eq(task) & model_data[metric].notna()]
                if task_data.empty:
                    continue
                for direction in ('lowest', 'highest'):
                    ranked = task_data.sort_values([metric, 'model_key', 'example_id', 'source_row'],
                        ascending=[direction == 'lowest', True, True, True]).head(15)
                    ids = ranked.audit_id.tolist()
                    selected.update(ids)
                    groups.append({'model': model, 'task': task, 'metric': metric, 'direction': direction,
                                   'eligible': len(task_data), 'ids': ids})

    columns = ['audit_id', 'model_key', 'evaluation_task', 'example_id', 'source_name', 'Level',
               'prompt', 'reference_answer', 'reference_explanation', 'reference_text',
               'raw_prediction', 'parsed_answer', 'rougeL', 'bertscore_f1',
               'nli_entailment_score', 'correct', 'normalized_correct', 'answer_preserved',
               'field_preservation', 'source_path', 'source_row']
    columns = [column for column in columns if column in data.columns]
    rows = data[data.audit_id.isin(selected)][columns].copy()
    reviews = json.loads(REVIEWED.read_text(encoding='utf-8'))['cases']
    reviews_by_id = {item['audit_id']: item for item in reviews}
    if len(reviews_by_id) != len(reviews) or set(reviews_by_id) - set(rows.audit_id):
        raise ValueError('Reviewed cases must have unique IDs present in selected corrected-run rows')
    rows['audit_judgment'] = rows.audit_id.map(lambda value: reviews_by_id.get(value, {}).get('assessment', ''))
    rows['audit_notes'] = rows.audit_id.map(lambda value: reviews_by_id.get(value, {}).get('note', ''))
    OUT.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUT / 'audit_samples.csv', index=False, encoding='utf-8-sig')
    rows.to_json(OUT / 'audit_samples.json', orient='records', indent=2, force_ascii=False)
    (OUT / 'manifest.json').write_text(json.dumps({'repo': REPO, 'commit': revision,
        'sources': sources, 'groups': groups, 'unique_rows': len(rows),
        'reviewed_cases': len(reviews),
        'selection': '15 lowest and highest scored rows per model, task, and metric; plus pooled rankings; no missing or inapplicable scores included.'},
        ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'reviewed_cases.json').write_text(REVIEWED.read_text(encoding='utf-8'), encoding='utf-8')
    viewer_data = {'groups': groups,
                   'records': json.loads(rows.to_json(orient='records', force_ascii=False)),
                   'reviews': reviews_by_id}
    embedded = json.dumps(viewer_data, ensure_ascii=False).replace('<', '\\u003c')
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Corrected-run score audit</title>
<style>body{font:16px system-ui;color:#1b2638;background:#f4f6fa;max-width:1120px;margin:24px auto;padding:0 20px}header{position:sticky;top:0;background:#f4f6fa;padding:12px 0;border-bottom:1px solid #ccd;z-index:1}select{font:inherit;padding:7px;margin:3px}article{background:white;border:1px solid #cbd3df;border-radius:9px;margin:16px 0;padding:18px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.5 system-ui}h3{font-size:14px;color:#425771;margin-bottom:2px}.scores{background:#edf3f9;padding:8px}.note{background:#fff5db;padding:10px}small{color:#526174}</style>
<h1>Corrected-run in-domain score audit</h1><p>Fifteen lowest and highest rows for each applicable task and metric, within each model and pooled. Scores locate cases for inspection; the 20 case notes are provisional AI-assisted judgments, not human-validated error rates.</p>
<header><select id="model"></select><select id="task"></select><select id="metric"></select><select id="direction"><option>lowest</option><option>highest</option></select><span id="status"></span></header><main id="rows"></main>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent),records=Object.fromEntries(data.records.map(r=>[r.audit_id,r]));
const $=id=>document.getElementById(id),esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
for(const [id,items] of [['model',['ALL',...new Set(data.groups.map(g=>g.model).filter(x=>x!=='ALL'))]],['task',[...new Set(data.groups.map(g=>g.task))]],['metric',[...new Set(data.groups.map(g=>g.metric))]]])for(const x of items)$(id).add(new Option(x,x));
function render(){const g=data.groups.find(g=>g.model===$('model').value&&g.task===$('task').value&&g.metric===$('metric').value&&g.direction===$('direction').value);$('status').textContent=g?` ${g.ids.length} of ${g.eligible} eligible rows`: 'No applicable scores';$('rows').innerHTML=g?g.ids.map((id,i)=>{const r=records[id],n=data.reviews[id],fields=['prompt','reference_answer','reference_explanation','reference_text','raw_prediction','parsed_answer'];return `<article><h2>${i+1}. ${esc(r.model_key)} · ${esc(r.evaluation_task)}</h2><small>${esc(id)} · ${esc(r.source_name)} · Level ${esc(r.Level)} · source row ${esc(r.source_row)}</small><p class="scores">${['rougeL','bertscore_f1','nli_entailment_score'].map(k=>`${k}: <b>${r[k]==null?'N/A':Number(r[k]).toFixed(6)}</b>`).join(' &nbsp; ')}</p>${n?`<p class="note"><b>Provisional assessment: ${esc(n.assessment)}</b><br>${esc(n.note)}</p>`:''}${fields.filter(k=>r[k]!=null&&r[k]!=='').map(k=>`<h3>${esc(k.replaceAll('_',' '))}</h3><pre>${esc(r[k])}</pre>`).join('')}</article>`}).join(''):'<p>No applicable rows for this combination.</p>';}
for(const id of ['model','task','metric','direction'])$(id).onchange=render;render();</script></html>'''
    (OUT / 'audit.html').write_text(html.replace('__DATA__', embedded), encoding='utf-8')
    print(f'{len(rows)} unique audit rows from {len(data)} corrected predictions; {len(groups)} rank lists; output {OUT}')


if __name__ == '__main__':
    main()
