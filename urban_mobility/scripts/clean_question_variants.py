"""Build a separate corrected corpus from saved components, with unchanged split IDs."""
from pathlib import Path
import copy
import hashlib
import json
import re
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from urban_mobility.pipelines.data_augmentation.nodes import (
    assemble_canonical_datasets, validate_and_audit_datasets, materialize_task_views,
)
from urban_mobility.pipelines.data_augmentation.question_content import question_content_errors
from urban_mobility.pipelines.experiment_data.nodes import prepare_experiment_partitions

DATA = ROOT / 'data'
DEST = DATA / 'revisions/revision_v6_prompt_cleaned'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def save_partitions(name, partitions):
    for key, value in partitions.items():
        write(DEST / name / f'{key}.json', value)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if DEST.exists():
        raise SystemExit(f'Refusing to overwrite an existing corrected release: {DEST}')
    source_files = sorted((DATA / '02_intermediate/augmented').glob('*.json'))
    originals = {p.stem: read(p) for p in source_files}
    corrected = copy.deepcopy(originals)
    changes = []
    for source, dataset in corrected.items():
        for item in dataset['seed_components']:
            for index, text in enumerate(item['components']['questions']):
                if not question_content_errors(text):
                    continue
                # Only remove the exact copied block observed in this audit. Unknown
                # formats must be inspected rather than silently truncated.
                match = re.fullmatch(
                    r'(?P<stem>.+?)\s*\n\s*Options:\s*\n(?P<options>\{.*\})'
                    r'(?:\s*\n\s*Correct option:\s*(?P<answer>[A-D]))?\s*', text, re.S,
                )
                if not match:
                    raise ValueError(f'Unrecognized contaminated question: {item["seed"]["seed_id"]}:{index}')
                options = json.loads(match['options'])
                seed = item['seed']
                if (set(options) != set(seed['Selections']) or not all(isinstance(v, str) for v in options.values())
                        or (match['answer'] is not None and match['answer'] != seed['Output'])):
                    raise ValueError('Copied option labels/key do not match the original seed; review required')
                stem = match['stem'].strip()
                if not stem.endswith('?') or question_content_errors(stem):
                    raise ValueError('Cleaned stem failed validation')
                item['components']['questions'][index] = stem
                changes.append(dict(source=source, seed_id=seed['seed_id'], question_index=index,
                                    before=text, after=stem))
    if len(changes) != 30:
        raise ValueError(f'Expected the reviewed 30 question strings, found {len(changes)}')

    params = {'random_seed': 2025}
    canonical, assembly = assemble_canonical_datasets(corrected, params)
    changed_variants = []
    for source, dataset in canonical.items():
        old = {r['variant_id']: r for r in read(DATA / '02_intermediate/canonical' / f'{source}.json')['records']}
        if set(old) != {r['variant_id'] for r in dataset['records']}:
            raise ValueError('Canonical IDs changed')
        for r in dataset['records']:
            prior = old[r['variant_id']]
            changed_fields = {k for k in set(r) | set(prior) if r.get(k) != prior.get(k)}
            if changed_fields - {'Question'}:
                raise ValueError(f'Unexpected changed fields: {changed_fields}')
            if changed_fields:
                changed_variants.append(r['variant_id'])
    if len(changed_variants) != 360:
        raise ValueError(f'Expected 360 canonical changes, got {len(changed_variants)}')
    print('Validated 30 exact block removals; rebuilding 360 descendants.', flush=True)
    validated, quality, duplicates, rejected = validate_and_audit_datasets(canonical)
    if rejected:
        raise ValueError(f'Corrected release has {len(rejected)} rejected records')
    final, generation, verification, paraphrase, statistics = materialize_task_views(validated)
    config = yaml.safe_load((ROOT / 'conf/base/parameters_experiment_data.yml').read_text())['experiment_data']
    config['dataset_version'] = 'revision_v6_prompt_cleaned'
    # Deterministic checks are rerun here. A prior semantic adjudication is never
    # copied as if it certified the newly edited prompts.
    config['semantic_audit']['enabled'] = False
    eg, ev, ep, manifest, audit = prepare_experiment_partitions(generation, verification, paraphrase, config)
    previous_manifest = read(DATA / '08_reporting/experiment/split_manifest.json')
    for key in ['concept_group_assignments', 'concept_group_counts', 'generation_record_counts', 'paraphrase_record_counts']:
        if manifest[key] != previous_manifest[key]:
            raise ValueError(f'Split invariant changed: {key}')
    view_changes = {}
    for name, partitions in [('generation', eg), ('verification', ev), ('paraphrase', ep)]:
        counts = {'prompt': 0, 'target': 0, 'records': 0}
        for key, dataset in partitions.items():
            old = {r['example_id']: r for r in read(DATA / '05_model_input/experiment' / name / f'{key}.json')['records']}
            if set(old) != {r['example_id'] for r in dataset['records']}:
                raise ValueError('Task-view IDs changed')
            for r in dataset['records']:
                changed = {k for k in set(r) | set(old[r['example_id']]) if r.get(k) != old[r['example_id']].get(k)}
                if changed - {'prompt', 'target'}:
                    raise ValueError(f'Unexpected task metadata change: {changed}')
                counts['records'] += bool(changed)
                counts['prompt'] += 'prompt' in changed
                counts['target'] += 'target' in changed
        view_changes[name] = counts
    if not audit['passed']:
        raise ValueError('Cross-split deterministic audit failed')

    for name, partitions in [('augmented', corrected), ('canonical', canonical), ('validated', validated),
            ('final', final), ('generation', generation), ('verification', verification), ('paraphrase', paraphrase),
            ('experiment/generation', eg), ('experiment/verification', ev), ('experiment/paraphrase', ep)]:
        save_partitions(name, partitions)
    for name, obj in [('question_changes', changes), ('assembly', assembly), ('quality_control', quality),
            ('duplicate_audit', duplicates), ('statistics', statistics), ('split_manifest', manifest),
            ('leakage_audit', audit)]:
        write(DEST / 'reporting' / f'{name}.json', obj)
    provenance = {
        'dataset_version': config['dataset_version'], 'parent_dataset_version': 'revision_v5',
        'parent_split_manifest_sha256': sha(DATA / '08_reporting/experiment/split_manifest.json'),
        'source_component_hashes': {p.name: sha(p) for p in source_files},
        'question_strings_changed': len(changes), 'canonical_records_changed': len(changed_variants),
        'changed_variant_ids': changed_variants, 'task_view_changes': view_changes,
        'split_assignments_unchanged': True, 'canonical_content_changed_only_in_question': True,
        'deterministic_leakage_audit_passed': audit['passed'],
        'semantic_audit_status': 'Must be recomputed for edited prompts; previous adjudication not reused',
        'evaluation_status': 'No predictions or checkpoints generated for this corrected release',
        'external_benchmark': 'Unmodified: data/05_model_input/multi_regional_external_v1',
    }
    provenance['output_sha256'] = {str(p.relative_to(DEST)): sha(p) for p in sorted(DEST.rglob('*.json'))}
    write(DEST / 'reporting/correction_manifest.json', provenance)
    print(json.dumps({'destination': str(DEST), 'changes': view_changes,
                      'split_counts': manifest['generation_record_counts'],
                      'quality': {k: v for k, v in quality.items() if k != 'warnings'}}, indent=2))


if __name__ == '__main__':
    main()
