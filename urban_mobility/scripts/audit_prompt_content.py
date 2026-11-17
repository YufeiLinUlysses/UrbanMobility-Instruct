"""Read-only lineage audit of embedded answer-key markers; writes separate reports."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'urban_mobility/data'
OUT = DATA / '08_reporting/dataset_prompt_audit_2026-09-14'
OUT.mkdir(exist_ok=True)
PATTERN = re.compile(r'\bcorrect\s+(?:option|answer|choice)\s*[:：]|\b(?:ground[- ]truth\s+answer|answer\s+key)\s*[:：]|\bthe\s+answer\s+to\s+the\s+question\s+is\s*[:：]|(?:^|\n)\s*(?:ANSWER|EXPLANATION|OUTPUT)\s*[:：]', re.I)
summaries, details, errors = [], [], []
generation = {}

def inspect_records(p, records, stage, default_split=''):
    groups = {}
    for i,r in enumerate(records):
        source=r.get('source_name',p.stem)
        split=r.get('split',default_split)
        task=r.get('task_type',r.get('evaluation_task',''))
        key=(source,split,task)
        g=groups.setdefault(key,dict(total=0,flagged=0,seeds=set()))
        g['total']+=1
        # Paraphrase intentionally contains a completed answer; inspect only the question there.
        if 'Question' in r:
            field='Question';value=str(r[field])
        elif 'prompt' in r:
            field='prompt';value=str(r[field])
            if 'paraphrase' in task or '/paraphrase/' in p.as_posix():
                m=re.search(r'Question:\s*(.*?)(?=\n\s*(?:Options|Answer|Explanation|ANSWER|EXPLANATION):|\Z)',value,re.S)
                if not m: continue
                value=m.group(1);field='question_inside_paraphrase'
        else: continue
        match=PATTERN.search(value)
        if match:
            g['flagged']+=1;g['seeds'].add(r.get('seed_id',''))
            details.append(dict(path=str(p.relative_to(ROOT)),stage=stage,source=source,split=split,task=task,
                record_index=i,example_id=r.get('example_id',''),variant_id=r.get('variant_id',''),
                seed_id=r.get('seed_id',''),field=field,marker=match.group(),text=value,
                reference=r.get('reference_answer',r.get('correct_answer',''))))
        if stage=='experiment/generation':
            generation[r['example_id']]=r
    sha=hashlib.sha256(p.read_bytes()).hexdigest()
    for (source,split,task),g in groups.items():
        summaries.append(dict(path=str(p.relative_to(ROOT)),sha256=sha,stage=stage,source=source,
            split=split,task=task,total=g['total'],flagged=g['flagged'],seeds=sorted(g['seeds'])))

for p in sorted((DATA/'01_raw/seed_instructions').glob('*.json')):
    d=json.loads(p.read_text(encoding='utf-8-sig'))
    rs=[r for ins in d['instructions'] for label in ['Positive Example','Negative Example'] for r in ins.get(label,[])]
    inspect_records(p,rs,'authored_seeds')
for folder,stage in [('02_intermediate/normalized','normalized'),('02_intermediate/canonical','canonical'),
        ('03_primary/validated','validated'),('05_model_input/final','final'),
        ('05_model_input/generation','generation'),('05_model_input/verification','verification'),
        ('05_model_input/paraphrase','paraphrase'),('05_model_input/experiment/generation','experiment/generation'),
        ('05_model_input/experiment/verification','experiment/verification'),('05_model_input/experiment/paraphrase','experiment/paraphrase'),
        ('01_raw/multi_regional_external_v1','external_raw'),('03_primary/multi_regional_external_v1','external_validated'),
        ('05_model_input/multi_regional_external_v1','external_views')]:
    for p in sorted((DATA/folder).rglob('*.json')):
        d=json.loads(p.read_text(encoding='utf-8-sig'))
        rs=d.get('records',d.get('seeds',[]))
        if not rs and 'instructions' in d:
            rs=[r for ins in d['instructions'] for r in ins.get('Positive Example',[])]
        if rs: inspect_records(p,rs,stage,d.get('split',''))
for p in sorted((DATA/'02_intermediate/augmented').glob('*.json')):
    d=json.loads(p.read_text())
    rs=[dict(Question=q,seed_id=c['seed']['seed_id'],source_name=d['source_name'],task_type=c['seed']['task_type'])
        for c in d['seed_components'] for q in c['components']['questions']]
    inspect_records(p,rs,'augmentation_questions')
comparisons=[]
for p in sorted(DATA.rglob('predictions.parquet')):
    try:
        df=pd.read_parquet(p)
        if 'prompt' not in df: continue
        rs=df.fillna('').to_dict('records')
        inspect_records(p,rs,'predictions')
        sub=df[df.evaluation_task.isin(['mc_answer','short_answer'])] if 'evaluation_task' in df else df.iloc[:0]
        same=missing=different=0
        for r in sub.to_dict('records'):
            local=generation.get(r['example_id'])
            if local is None: missing+=1
            elif local['prompt']==r['prompt']: same+=1
            else: different+=1
        comparisons.append(dict(path=str(p.relative_to(ROOT)),same=same,different=different,not_in_primary_map=missing))
    except Exception as e: errors.append(dict(path=str(p),error=str(e)))
for name,obj in [('file_summary',summaries),('flagged_records',details),('prediction_prompt_comparison',comparisons),('errors',errors)]:
    (OUT/f'{name}.json').write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
print('Errors:',errors)
for stage in sorted({s['stage'] for s in summaries}):
    ss=[s for s in summaries if s['stage']==stage]
    print(stage,'total',sum(s['total'] for s in ss),'flagged',sum(s['flagged'] for s in ss))
print('Experiment generation:')
for s in summaries:
    if s['stage']=='experiment/generation': print(s['source'],s['split'],s['total'],s['flagged'],s['seeds'])
