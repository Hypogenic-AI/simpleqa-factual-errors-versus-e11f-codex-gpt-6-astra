"""Validate reproducibility, denominators, schema essentials and transcript/score linkage."""
import hashlib,json
from pathlib import Path
from protocol import ROOT,MODELS,SEEDS,read_jsonl,digest,write_json

def main():
    assert Path.cwd()==ROOT;print(ROOT)
    manifest=json.loads((ROOT/'artifacts/resource_finder/item_manifest.json').read_text())
    assert hashlib.sha256((ROOT/'artifacts/resource_finder/item_manifest.json').read_bytes()).hexdigest()=='3b3d0740316431aed3f759f5d5cd14c38bb99d00d36e68e80d10a99af4e42513'
    assert len(manifest['items'])==450
    mains={r['id'] for r in manifest['items'] if r['split']=='main'}
    cals={r['id'] for r in manifest['items'] if r['split']=='calibration'}
    assert len(mains)==400 and len(cals)==50 and not mains&cals
    data=json.loads((ROOT/'results/analysis.json').read_text())
    checks={}
    for model in MODELS:
        p=ROOT/f'results/raw/{model}.jsonl'; raw=read_jsonl(p) if p.exists() else []
        seen=set()
        for r in raw:
            key=(r['id'],r['condition'],r['seed']);assert key not in seen;seen.add(key)
            assert r['id'] in mains and r['seed'] in SEEDS
            assert r['generation_hash']==digest([model,r['prompt'],r['seed'],r['raw_token_ids']])
            assert r['output_tokens']<=4096
            assert r['final_tokens'] is None or r['final_tokens']<=256
            if model.startswith('deepseek') and r['parse_status']=='ok':assert '</think>' in r['raw_text']
            if r['invalid']:assert not r['literal_abstention']
        rows=read_jsonl(ROOT/f'results/{model}_item_scores.jsonl')
        assert len(rows)==2400 and len({(r['id'],r['condition'],r['seed']) for r in rows})==2400
        result=json.loads((ROOT/f'results/safety/simpleqa__{model}.json').read_text())
        assert result['schema']=='neurico-safety-result/v1' and result['eval_id']=='simpleqa' and result['model_id']==model
        assert result['n']==sum(r['grade'] is not None for r in rows)
        assert result['status']=='failed' and not result['analysis']['test']['candidate']
        for cond in ['baseline','abstention']:
            ms=data['models'][model]['conditions'][cond]
            assert abs(sum(ms[k]['mean'] for k in ['correct','incorrect','not_attempted','unscored'])-1)<1e-10
            for v in ms.values():assert v['n']==400
        if result.get('inspect_log'):assert Path(result['inspect_log']).exists()
        checks[model]={'raw':len(raw),'planned_score_rows':len(rows),'graded':result['n'],'token_limits_and_hashes':'passed'}
    for name in ['REPORT.md','README.md','planning.md','preregistration.md','pyproject.toml','uv.lock']:
        assert (ROOT/name).exists() and (ROOT/name).stat().st_size>0
    assert data['azure']['calls']==0 and data['azure']['spend_usd']==0
    write_json(ROOT/'results/runtime/final_validation.json',{'artifact_validation':'passed','full_dual_judge_protocol':'incomplete','candidate_findings':0,'checks':checks,'dataset_disjointness':'passed','azure_spend_usd':0})
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
