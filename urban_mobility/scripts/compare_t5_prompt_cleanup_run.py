"""Compare the completed corrected T5 run with the published revision-v5 run."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'data/07_model_output/evaluations'
NEW = ROOT / 'data/08_reporting/t5_prompt_cleanup_comparison_2026-09-14/downloaded/evaluations'
OUT = ROOT / 'data/08_reporting/t5_prompt_cleanup_comparison_2026-09-14'


def paths(scope, stage):
    label = 'base' if stage == 'base' else 'finetuned'
    if scope == 'multi_regional':
        old = OLD / f'multi_regional_external_v1/t5_base/{stage}/multi_regional/revision-v5-t5-three-task-v1-{label}-multi-regional/predictions.parquet'
        new = NEW / f'multi_regional_external_v1/t5_base/{stage}/multi_regional/revision-v6-t5-prompt-cleaned-v1-{label}-multi-regional/predictions.parquet'
    else:
        old = OLD / f'revision_v5/t5_base/{stage}/{scope}/revision-v5-t5-three-task-v1-{label}-{scope.replace("_", "-")}/predictions.parquet'
        new = NEW / f'revision_v6_prompt_cleaned/t5_base/{stage}/{scope}/revision-v6-t5-prompt-cleaned-v1-{label}-{scope.replace("_", "-")}/predictions.parquet'
    return old, new


def paired(df):
    x = df[df.evaluation_task.eq('answer_verification')].copy()
    ok = x['correct'].fillna(False) & x['candidate_match_correct'].fillna(False) & x['verdict_match_consistent'].fillna(False)
    return float(ok.groupby(x['variant_id']).all().mean() * 100)


def summarize(df):
    result = {}
    specs = {
        'MC normalized accuracy (%)': ('mc_answer', 'normalized_correct', 100),
        'MC strict accuracy (%)': ('mc_answer', 'strict_correct', 100),
        'MC format compliance (%)': ('mc_answer', 'format_compliant', 100),
        'SA exact match (%)': ('short_answer', 'exact_match', 100),
        'SA ROUGE-L': ('short_answer', 'rougeL', 1),
        'SA BERTScore F1': ('short_answer', 'bertscore_f1', 1),
        'SA NLI mean': ('short_answer', 'nli_entailment_score', 1),
        'MC explanation BERTScore': ('mc_explanation', 'bertscore_f1', 1),
        'SA explanation BERTScore': ('short_answer_explanation', 'bertscore_f1', 1),
        'Verification accuracy (%)': ('answer_verification', 'correct', 100),
        'Paraphrase BERTScore': ('paraphrase', 'bertscore_f1', 1),
        'Paraphrase answer preserved (%)': ('paraphrase', 'answer_preserved', 100),
        'Paraphrase field preservation (%)': ('paraphrase', 'field_preservation', 100),
    }
    for name, (task, col, scale) in specs.items():
        part = df[df.evaluation_task.eq(task)]
        if len(part) and col in part:
            result[name] = float(part[col].mean() * scale)
    result['Verification paired consistency (%)'] = paired(df)
    return result


def main():
    rows, identity, affected = [], [], []
    affected_seed = 'japan_mlit_instructions__multiple_choice__l2__0001'
    for scope in ('in_domain', 'cross_regional', 'multi_regional'):
        for stage in ('base', 'fine_tuned'):
            old_path, new_path = paths(scope, stage)
            old, new = pd.read_parquet(old_path), pd.read_parquet(new_path)
            old_s, new_s = summarize(old), summarize(new)
            for metric in sorted(set(old_s) | set(new_s)):
                a, b = old_s.get(metric), new_s.get(metric)
                rows.append({'scope': scope, 'stage': stage, 'metric': metric,
                             'revision_v5': a, 'corrected': b,
                             'change': None if a is None or b is None else b-a})
            keys = ['example_id', 'evaluation_task']
            merged = old[keys+['prompt','raw_prediction']].merge(
                new[keys+['prompt','raw_prediction']], on=keys, suffixes=('_old','_new'), validate='one_to_one')
            same_prompt = merged.prompt_old.eq(merged.prompt_new)
            identity.append({'scope':scope,'stage':stage,'old_rows':len(old),'new_rows':len(new),
                             'common_rows':len(merged),'changed_prompts':int((~same_prompt).sum()),
                             'same_prediction_on_unchanged_prompt_pct':float(merged.loc[same_prompt,'raw_prediction_old'].eq(merged.loc[same_prompt,'raw_prediction_new']).mean()*100)})
            if scope == 'cross_regional':
                for label, df in [('revision_v5',old),('corrected',new)]:
                    part=df[df.seed_id.eq(affected_seed)]
                    affected.append({'stage':stage,'version':label,'task_rows':len(part),
                                     'mc_accuracy_pct':float(part[part.evaluation_task.eq('mc_answer')].normalized_correct.mean()*100),
                                     'verification_accuracy_pct':float(part[part.evaluation_task.eq('answer_verification')].correct.mean()*100)})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUT/'metric_comparison.csv',index=False)
    (OUT/'comparison.json').write_text(json.dumps({'metric_rows':rows,'prediction_identity':identity,
        'affected_japan_seed':affected},indent=2),encoding='utf-8')
    headline = comparison[comparison.metric.isin(['MC normalized accuracy (%)','SA ROUGE-L','Verification accuracy (%)','Verification paired consistency (%)','Paraphrase BERTScore'])]
    print(headline.to_string(index=False))
    print('\nPrediction identity:',json.dumps(identity,indent=2))
    print('\nAffected Japanese seed:',json.dumps(affected,indent=2))


if __name__ == '__main__': main()
