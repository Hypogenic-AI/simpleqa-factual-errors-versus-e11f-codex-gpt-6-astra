"""Reproduce frozen SimpleQA resources; no model/API calls. Run from repository root."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA = 'feee3f7e7db3617e94e8fcf1977b756ec420ef8568f4e0fcbbe0e92e9d5fc032'
URL = 'https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv'
SEED = '20261020'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

def main():
    assert Path.cwd() == ROOT and (ROOT / '.git').exists(), 'Run from workspace repository root'
    print(ROOT)
    datafile = ROOT / 'datasets/simple_qa_test_set.csv'
    if not datafile.exists():
        r = requests.get(URL, timeout=90)
        r.raise_for_status()
        assert sha(r.content) == EXPECTED_SHA, 'Source changed; do not resample silently'
        datafile.write_bytes(r.content)
    assert sha(datafile.read_bytes()) == EXPECTED_SHA
    with datafile.open(newline='') as f:
        rows = list(csv.DictReader(f))
    # Evaluate only the pinned pure helper, without importing the evaluation framework.
    source = ROOT / 'code/inspect_evals/src/inspect_evals/utils/deps_utils.py'
    tree = ast.parse(source.read_text())
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'create_stable_id')
    namespace = {'hashlib': hashlib, 'Any': object}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), 'exec'), namespace)
    ranked = []
    for index, row in enumerate(rows):
        assert row['problem'].strip() and row['answer'].strip()
        canonical_id = namespace['create_stable_id'](row['problem'], row['answer'], prefix='simpleqa')
        # Explicit delimiter is part of the frozen seed-plus-ID serialization.
        selection_hash = sha((SEED + ':' + canonical_id).encode('utf-8'))
        metadata = ast.literal_eval(row['metadata'])
        assert isinstance(metadata, dict)
        ranked.append(dict(id=canonical_id, source_row=index, selection_hash=selection_hash,
                           problem=row['problem'], answer=row['answer'], metadata=metadata))
    assert len(rows) == 4326
    assert len({r['id'] for r in ranked}) == len(rows)
    assert len({r['problem'] for r in ranked}) == len(rows)
    ranked.sort(key=lambda r: (r['selection_hash'], r['id']))
    chosen = ranked[:450]
    for rank, row in enumerate(chosen):
        row['selection_rank'] = rank
        row['split'] = 'calibration' if rank < 50 else 'main'
    calibration, main_rows = chosen[:50], chosen[50:]
    assert not ({r['id'] for r in calibration} & {r['id'] for r in main_rows})
    # Preserve all question/answer data under datasets/; lightweight manifest has IDs only.
    for name, data in [('calibration_questions', calibration), ('main_questions', main_rows)]:
        (ROOT / f'datasets/{name}.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in data))
    controls=[]
    for row in calibration:
        for label, answer in [('A',row['answer']), ('C','I do not know'),
                              ('B',f'The answer to this question is not {json.dumps(row["answer"], ensure_ascii=False)}.')]:
            controls.append(dict(control_id=row['id']+'_'+label, question_id=row['id'], question=row['problem'],
                                 gold=row['answer'], response=answer, label=label,
                                 label_provenance='programmatically constructed relative to dataset gold; not human labelled'))
    assert len(controls)==150
    (ROOT/'datasets/calibration_controls.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in controls))
    manifest=dict(dataset_url=URL,dataset_sha256=EXPECTED_SHA,seed=SEED,
                  selection_rule='ascending SHA256(UTF8("20261020:" + canonical_id)), tie-break canonical_id',
                  canonical_id_rule='pinned Inspect create_stable_id(problem, answer, prefix="simpleqa"): MD5 of NUL-separated fields, first 8 hex chars',
                  id_helper_sha256=sha(source.read_bytes()),
                  items=[{k:r[k] for k in ['id','source_row','selection_hash','selection_rank','split']} for r in chosen])
    write_json(ROOT/'artifacts/resource_finder/item_manifest.json',manifest)
    summary=dict(n_source=len(rows),n_selected=len(chosen),n_calibration=50,n_main=400,n_controls=len(controls),
                 control_labels={g:sum(x['label']==g for x in controls) for g in 'ABC'},
                 manifest_sha256=sha((ROOT/'artifacts/resource_finder/item_manifest.json').read_bytes()),
                 files={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in sorted((ROOT/'datasets').glob('*')) if p.suffix in ['.csv','.jsonl']})
    write_json(ROOT/'artifacts/resource_finder/preparation_validation.json',summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
