"""Deterministic diagnostic examples and exploratory truncation-as-unknown bounds.
These diagnostics do not replace preregistered effects or produce candidates.
"""
import json
from pathlib import Path
import numpy as np
from protocol import ROOT,MODELS,SEEDS,read_jsonl,write_json
from analyze import summary

def main():
    assert Path.cwd()==ROOT; print(ROOT)
    questions={r['id']:r for r in read_jsonl(ROOT/'datasets/main_questions.jsonl')}
    ids=list(questions);idx={v:i for i,v in enumerate(ids)}
    bs=np.random.default_rng(20261020).integers(0,400,(10000,400))
    cache={r['key']:r for r in read_jsonl(ROOT/'results/calibration/local_cache.jsonl')}
    mappings=json.loads((ROOT/'results/calibration/local_mapping.json').read_text())
    lookup={(r['model'],r['id'],r['condition'],r['seed']):r['key'] for r in mappings if r['kind']=='main'}
    out={};cases=[]
    for model in MODELS:
        rows=read_jsonl(ROOT/f'results/{model}_item_scores.jsonl')
        known={k:np.zeros((400,2,3)) for k in ['correct','incorrect']};unknown=np.ones((400,2,3))
        for r in rows:
            i,c,s=idx[r['id']],['baseline','abstention'].index(r['condition']),SEEDS.index(r['seed'])
            if r.get('grade') and not r.get('truncated',False):
                unknown[i,c,s]=0
                known['correct'][i,c,s]=r['grade']=='A';known['incorrect'][i,c,s]=r['grade']=='B'
        u=unknown.mean(axis=2);bounds={}
        for k,x in known.items():
            a=x.mean(axis=2);d=a[:,1]-a[:,0]
            bounds[k]={'minimum_effect':summary(d-u[:,0],bs),'maximum_effect':summary(d+u[:,1],bs)}
        out[model]={'exploratory':True,'definition':'Treat both unscored and truncated finals as unknown, adversarially by condition','bounds':bounds}
        for category,predicate in [('truncated',lambda r:r.get('truncated',False)),('invalid_no_final',lambda r:r.get('invalid',False)),('infrastructure_missing',lambda r:r.get('infrastructure_failure',False)),('judge_format_failure',lambda r:r.get('unscored',False) and not r.get('invalid',False) and not r.get('infrastructure_failure',False))]:
            selected=sorted([r for r in rows if predicate(r)],key=lambda r:(r['id'],r['condition'],r['seed']))[:3]
            for r in selected:
                q=questions[r['id']]
                cases.append({'model':model,'category':category,'id':r['id'],'condition':r['condition'],'seed':r['seed'],'question':q['problem'],'dataset_gold':q['answer'],'local_grade':r.get('grade'),'final_excerpt':r.get('final_answer','')[:800],'finish_reason':r.get('finish_reason'),'raw_path':f'results/raw/{model}.jsonl','generation_hash':r.get('generation_hash'),'judge_raw_output':cache.get(lookup.get((model,r['id'],r['condition'],r['seed'])),{}).get('raw_text'),'judge_finish_reason':cache.get(lookup.get((model,r['id'],r['condition'],r['seed'])),{}).get('finish_reason')})
    write_json(ROOT/'results/truncation_sensitivity.json',out)
    write_json(ROOT/'results/diagnostic_cases.json',cases)
    print('Wrote',len(cases),'diagnostic examples and exploratory bounds')

if __name__=='__main__':main()
