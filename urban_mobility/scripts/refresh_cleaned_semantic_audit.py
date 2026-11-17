"""Refresh the existing semantic screen against corrected primary partitions."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch
import yaml
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from urban_mobility.pipelines.experiment_data.nodes import _semantic_cross_split_audit


def main():
    release = ROOT / 'data/revisions/revision_v6_prompt_cleaned'
    files = sorted((release / 'experiment/generation').glob('*.json'))
    partitions = {p.stem: json.loads(p.read_text(encoding='utf-8')) for p in files}
    config = yaml.safe_load((ROOT / 'conf/base/parameters_experiment_data.yml').read_text())['experiment_data']
    config['semantic_audit']['enabled'] = True
    torch.set_num_threads(4)
    model = SentenceTransformer(config['semantic_audit']['model_id'], device='cpu')

    class UniqueEncoder:
        def encode(self, texts, **kwargs):
            # Repeated strings receive identical vectors; restore original row order
            # before running the unmodified cross-split nearest-neighbor screen.
            unique = list(dict.fromkeys(texts))
            print(f'Encoding {len(unique)} unique prompts from {len(texts)} rows.', flush=True)
            vectors = model.encode(unique, **kwargs)
            lookup = dict(zip(unique, vectors))
            return np.array([lookup[t] for t in texts])

    audit = _semantic_cross_split_audit(partitions, config, encoder=UniqueEncoder())
    audit['input_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    audit['dataset_version'] = 'revision_v6_prompt_cleaned'
    dest = release / 'reporting/semantic_audit_refreshed.json'
    dest.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(f'Saved {dest}; {len(audit["human_review_queue"])} pairs require review.')


if __name__ == '__main__':
    main()
