"""Summarize task-specific low-score rates from the downloaded audit sources."""
from pathlib import Path
import hashlib
import html
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/08_reporting/in_domain_finetuned_audit_2026-09-12'
OUT = ROOT / 'data/08_reporting/low_score_distribution_2026-09-13'
METRICS = {'rougeL': 'ROUGE-L', 'bertscore_f1': 'BERTScore F1', 'nli_entailment_score': 'NLI entailment'}
NAMES = {'llama31_70b': 'Llama 3.1 70B', 'llama31_8b': 'Llama 3.1 8B', 'qwen25_14b': 'Qwen 2.5 14B', 'qwen25_7b': 'Qwen 2.5 7B', 't5_base': 'T5 base'}


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = json.loads((SOURCE / 'sources.json').read_text(encoding='utf-8'))
    frames = []
    for source in sources:
        path = SOURCE / 'sources' / source['model'] / 'predictions.parquet'
        frame = pd.read_parquet(path)
        assert frame.model_key.eq(source['model']).all()
        assert frame.checkpoint_stage.eq('fine_tuned').all() and frame.evaluation_scope.eq('in_domain').all()
        source['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    details, thresholds = [], []
    for metric in METRICS:
        for task, group in data.groupby('evaluation_task', sort=True):
            if metric == 'rougeL' and task in {'mc_answer', 'answer_verification'}:
                assert group[metric].isna().all()
                continue
            scores = group[metric].dropna()
            threshold = float(scores.quantile(0.10, interpolation='linear'))
            thresholds.append(dict(metric=metric, task=task, threshold=threshold, denominator=len(scores), low_count=int((scores < threshold).sum()), tied_count=int((scores == threshold).sum())))
            for model, subset in group.groupby('model_key', sort=True):
                values = subset[metric].dropna()
                count, tied, n = int((values < threshold).sum()), int((values == threshold).sum()), len(values)
                details.append(dict(model=model, metric=metric, task=task, threshold=threshold, low_count=count, denominator=n, low_percent=100*count/n, tied_count=tied, inclusive_percent=100*(count+tied)/n, missing_count=int(subset[metric].isna().sum()), p10=float(values.quantile(.1)), median=float(values.median()), p90=float(values.quantile(.9))))
    summary = []
    for model in NAMES:
        for metric in METRICS:
            rows = [r for r in details if r['model'] == model and r['metric'] == metric]
            n, low, tied = (sum(r[k] for r in rows) for k in ['denominator', 'low_count', 'tied_count'])
            summary.append(dict(model=model, metric=metric, denominator=n, low_count=low, low_percent=100*low/n, tied_count=tied, inclusive_percent=100*(low+tied)/n))
    for t in thresholds:
        rows = [r for r in details if r['metric'] == t['metric'] and r['task'] == t['task']]
        for key in ['denominator', 'low_count', 'tied_count']:
            assert sum(r[key] for r in rows) == t[key]
        assert len({r['denominator'] for r in rows}) == 1, 'Models have unequal task coverage'
    payload = dict(method='Low = score strictly below pooled model/task 10th percentile, linear interpolation; exact ties excluded and reported separately.', sources=sources, thresholds=thresholds, summary=summary, by_task=details)
    (OUT / 'low_score_distribution.json').write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    introduction = f'''# Low-score percentages: in-domain fine-tuned models

Analyzed all {len(data):,} downloaded evaluation rows, not only the 15 audit extremes. For each metric and task, pool the five models and compute the 10th percentile using linear interpolation. A score is low when it is **strictly below** that shared threshold. Percentage = low count / nonmissing applicable scores × 100. Exact ties are excluded; their counts and the alternative inclusive percentage are provided to make boundary effects visible. This is a relative score-tail comparison, not a human-confirmed error rate.

The overview sums task-specific low counts and denominators. Each model has identical task coverage, but this is a row-weighted summary: verification contributes 2,396 of 5,552 semantic-score rows per model. Use the task tables for interpretation. ROUGE-L excludes multiple-choice answer and verification rows where ROUGE is not applicable. Null values are never counted as zeros. BERTScore and NLI are retained as published, including verdict tasks; potential reference/response mismatches remain an audit concern. Separate task thresholds prevent verdict scores from defining the low-score cutoff for other tasks.

Source: the same five revision_v5 runs as the audit, pinned to Hugging Face commit `{sources[0]['commit']}`. No scores were recomputed. Source URLs and SHA-256 hashes are in the JSON companion. Rows are model/task evaluations and can share an underlying question; these percentages are descriptive and are not independent-sample significance tests.

## Overview: task-adjusted low-score rates
'''
    overview = table(['Model'] + list(METRICS.values()), [[NAMES[model]] + [f"{r['low_percent']:.2f}% ({r['low_count']}/{r['denominator']})" for metric in METRICS for r in summary if r['model'] == model and r['metric'] == metric] for model in NAMES])
    sections = [introduction, overview]
    webpage = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Low-score distribution</title><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:35px auto;padding:0 24px;color:#182338}table{border-collapse:collapse;width:100%;margin:20px 0 35px}th,td{text-align:left;padding:10px;border-bottom:1px solid #ddd}th{background:#eef3f9}.bar{display:inline-block;background:#4477aa;height:10px;margin-right:8px}small{color:#536174}</style><h1>Low-score percentages</h1><p>All 27,760 in-domain fine-tuned evaluation rows. Low = score strictly below the shared, task-specific pooled 10th percentile. These are score-tail rates, not error rates. Exact cutoff ties are excluded and reported separately.</p><p>The overview weights tasks by their scored row counts. BERTScore/NLI include verdict tasks; ROUGE-L excludes tasks where it does not apply. See the Markdown report for full methodology and the JSON for source provenance and full precision.</p>']
    def html_table(headers, rows):
        return '<table><thead><tr>' + ''.join('<th>'+html.escape(str(c))+'</th>' for c in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table>'
    webpage.append(html_table(['Model'] + list(METRICS.values()), [[NAMES[m]]+[f"{r['low_percent']:.2f}% ({r['low_count']}/{r['denominator']})" for k in METRICS for r in summary if r['model']==m and r['metric']==k] for m in NAMES]))
    for metric, label in METRICS.items():
        sections.append('\n## '+label)
        webpage.append('<h2>'+label+'</h2>')
        for threshold in [t for t in thresholds if t['metric'] == metric]:
            task = threshold['task']
            heading = f"{task}: cutoff {threshold['threshold']:.12g}"
            sections.append('\n### '+heading+'\n')
            webpage.append('<h3>'+heading+'</h3>')
            headers = ['Model', 'Low / scored', 'Low %', 'At cutoff', '% if ties included', 'Median', 'Missing']
            rows = [[NAMES[r['model']], f"{r['low_count']}/{r['denominator']}", f"{r['low_percent']:.2f}%", r['tied_count'], f"{r['inclusive_percent']:.2f}%", f"{r['median']:.6f}", r['missing_count']] for r in details if r['metric']==metric and r['task']==task]
            sections.append(table(headers, rows))
            webpage.append(html_table(headers, rows))
    (OUT / 'low_score_distribution.md').write_text('\n\n'.join(sections)+'\n', encoding='utf-8')
    (OUT / 'low_score_distribution.html').write_text(''.join(webpage)+'</html>', encoding='utf-8')
    assert len(details) == 80 and len(thresholds) == 16
    assert json.loads((OUT / 'low_score_distribution.json').read_text()) == payload
    print(overview)
    print('Verified 80 model/task/metric groups, 16 pooled cutoffs, identical model coverage, and JSON round-trip.')
    print('Threshold ties:', [(t['metric'], t['task'], t['tied_count']) for t in thresholds if t['tied_count']])


if __name__ == '__main__':
    main()
