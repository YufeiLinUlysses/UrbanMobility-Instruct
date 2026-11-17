# Corrected question variants — revision_v6_prompt_cleaned

This local corrected release removes embedded option dictionaries and answer keys from question stems, rebuilding all task views from saved augmentation components. It contains no new model predictions and has not been uploaded. The published revision_v5 corpus and its evaluation artifacts remain preserved as evidence.

## Changes

- Removed copied options and `Correct option:` blocks from the 25 previously identified question variants across NGSIM, TLC, the Urban Flow Survey, and Japan.
- Removed copied options (without an answer key) from five additional highD Level-2 question variants detected by the strengthened validator. This is a separate formatting defect from the 300 answer-key-bearing records in the original audit.
- Preserved each question stem verbatim except trailing whitespace, and preserved canonical facts, final options, correct answers, explanations, lineage, record IDs, and group assignments.
- Rebuilt 360 canonical descendants: 240 training, 60 validation, and 60 regional records. In-domain test records are unchanged.
- Updated 360 generation prompts and 720 verification prompts. Their supervised targets are unchanged.
- Corrected 60 completed-instruction paraphrase targets that also contained the copied blocks. Paraphrase input prompts are unchanged.

## Dataset and paths

| View | Records | Experimental partitions |
|---|---:|---|
| Generation | 14,726 | `experiment/generation/` |
| Candidate verification | 29,452 | `experiment/verification/` |
| Paraphrase | 8,000 | `experiment/paraphrase/` |

The training/validation/test generation counts remain 8,326 / 1,036 / 1,198, with 4,166 regional records. Group counts remain 85 / 11 / 10 under the original 80/10/10 target. The ten-source external benchmark is unchanged and remains at `data/05_model_input/multi_regional_external_v1/`.

Use this release's experimental partitions and `reporting/split_manifest.json` together for subsequent training/evaluation. The default revision_v5 configuration has not been redirected, so existing published runs cannot silently become associated with corrected inputs. A new run must explicitly select these paths and the new dataset version.

## Verification

- All 1,745 generated question strings and 14,726 canonical question fields pass the new embedded-content screen.
- Full canonical validation accepts all 14,726 records with no rejected records or exact duplicates. The existing 4,260 lexical-grounding warnings are retained; they were not converted into factual correctness judgments.
- Recomputed deterministic cross-split concept, seed, and normalized-prompt checks pass.
- Verified identical group assignments, record IDs, and counts against revision_v5.
- Verified that canonical fields other than Question are unchanged, and task-view differences are limited to the prompt/target changes listed above.
- Original component hashes are unchanged; generated artifact hashes are recorded in `reporting/correction_manifest.json`.
- 22 augmentation and split tests pass, including new regression tests for embedded option/answer blocks and cached question responses.

The semantic embedding audit was rerun on the corrected primary partitions using `all-mpnet-base-v2` and the original 0.90 threshold. It flags the same three concept pairs as the earlier audit. Fresh manual comparison of the complete cleaned prompts classifies all three as related but distinct: the paired questions require different answers or reasoning operations. The adjudicated audit therefore contains zero confirmed and zero pending semantic-leakage cases. Use new predictions for every changed evaluation prompt; existing checkpoint behavior and published numerical results have not been fully revalidated by this data cleanup.

## Evidence and reproduction

`reporting/question_changes.json` contains exact before/after strings. `correction_manifest.json` lists changed variant IDs, source/output hashes, and invariants. `final_verification.json` records the residual-content screen and changes by split. Structural and deterministic leakage reports are also included.

From the repository root, the builder is:

```powershell
urban_mobility/.venv/Scripts/python.exe urban_mobility/scripts/clean_question_variants.py
```

It refuses to overwrite an existing corrected release. Future augmentation rejects contaminated strings returned by either the API or cache, and canonical validation rejects embedded options or answer keys in the Question field.

For the T5 confirmation run, use `notebooks/urban_science_t5_prompt_cleanup_rerun.ipynb`. The notebook downloads corrected partitions from the private Hugging Face dataset repository at a pinned commit; no ZIP upload is required. It trains T5-base with the paper's final configuration, runs base and fine-tuned evaluations across all three regimes, and verifies the uniquely named artifacts on Hugging Face.
