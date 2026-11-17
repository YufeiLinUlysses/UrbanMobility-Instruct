"""Record the manual review of the three semantic pairs in the corrected release."""
from pathlib import Path
import json

PATH = Path(__file__).resolve().parents[1] / (
    'data/revisions/revision_v6_prompt_cleaned/reporting/semantic_audit_refreshed.json'
)

NOTES = {
    ('tlc_trip_instructions__short_answer__l1__0001',
     'tlc_trip_instructions__short_answer__l1__0010'):
        'Both concern TLC passenger trips, but one asks whose movements are represented and the other asks the city containing trip origins and destinations.',
    ('ngsim_instructions__multiple_choice__l1__0003',
     'ngsim_instructions__multiple_choice__l2__0003'):
        'Both mention 10 Hz, but the Level 1 item asks for its literal frequency meaning while the Level 2 item asks what motion variables that resolution supports.',
    ('highd_instructions__multiple_choice__l1__0004',
     'highd_instructions__multiple_choice__l2__0003'):
        'Both mention 25 Hz, but the Level 1 item asks for its literal frequency meaning while the Level 2 item asks what motion-variable calculation it supports.',
}


def main():
    audit = json.loads(PATH.read_text(encoding='utf-8'))
    reviewed = 0
    for row in audit['human_review_queue']:
        key = (row['left_seed_id'], row['right_seed_id'])
        if key not in NOTES:
            raise ValueError(f'Unexpected semantic candidate requires independent review: {key}')
        row['human_decision'] = 'related_but_distinct'
        row['human_notes'] = NOTES[key]
        reviewed += 1
    if reviewed != 3:
        raise ValueError(f'Expected three reviewed pairs, received {reviewed}')
    audit['review_required'] = False
    audit['adjudication_summary'] = {
        'reviewed_concept_pairs': reviewed,
        'related_but_distinct': reviewed,
        'confirmed_leakage': 0,
        'pending': 0,
        'review_basis': 'Manual comparison of the complete cleaned prompt pairs.',
        'prior_decisions_reused_without_review': False,
    }
    out = PATH.with_name('semantic_audit_refreshed_adjudicated.json')
    out.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
