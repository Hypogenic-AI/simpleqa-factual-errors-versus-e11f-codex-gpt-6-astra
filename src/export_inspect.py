"""Create genuine Inspect replay logs from cached outputs, with zero model/judge calls.
Logs are explicitly post-hoc replay artifacts, not new generations or contemporaneous logs.
The skill converter supplies the standard base schema; clustered analysis is preserved.
"""
import os, json, subprocess, sys
from pathlib import Path
from protocol import ROOT, MODELS, read_jsonl,write_json
os.environ['XDG_DATA_HOME']=str(ROOT/'results/runtime/inspect_data')
os.environ['XDG_CACHE_HOME']=str(ROOT/'models/.cache')
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.model import ModelOutput,ModelAPI,Model,GenerateConfig,modelapi
from inspect_ai.solver import solver
from inspect_ai.scorer import scorer,Score,mean,stderr

@modelapi(name='cached-replay')
class NoInferenceAPI(ModelAPI):
    """A serialization-only provider that raises if inference is accidentally requested."""
    async def generate(self,*args,**kwargs):
        raise RuntimeError('Replay export must never call an inference provider')

def replay_model(name):
    return Model(NoInferenceAPI('cached-replay/'+name),GenerateConfig())

@solver
def replay():
    async def solve(state,generate):
        state.output=ModelOutput.from_content(model=state.model.name,content=state.metadata.get('final_answer',''))
        return state
    return solve

@scorer(metrics={k:[mean(),stderr()] for k in ['correct','incorrect','not_attempted','unscored','truncated','infrastructure_failure']})
def cached_grade():
    async def score(state,target):
        r=state.metadata
        return Score(value={
            'correct':int(r.get('grade')=='A'),'incorrect':int(r.get('grade')=='B'),
            'not_attempted':int(r.get('grade')=='C'),'unscored':int(r.get('grade') is None),
            'truncated':int(r.get('truncated',False)),
            'infrastructure_failure':int(r.get('infrastructure_failure',False))},
            answer=r.get('grade') or 'UNSCORED',
            explanation='Cached fixed-rubric local grade; synthetic-control calibration reported separately. Replay does not call models.',
            metadata={'original_id':r['id'],'condition':r['condition'],'seed':r['seed'],'judge':'qwen2.5-7b-judge','post_hoc_replay':True})
    return score

def main():
    assert Path.cwd()==ROOT;print(ROOT)
    question={r['id']:r for r in read_jsonl(ROOT/'datasets/main_questions.jsonl')}
    for model in MODELS:
        rows=read_jsonl(ROOT/f'results/{model}_item_scores.jsonl')
        samples=[Sample(id=r['id']+'_'+r['condition']+'_'+str(r['seed']),input=question[r['id']]['problem'],target=question[r['id']]['answer'],metadata={k:r[k] for k in ['id','condition','seed','grade','final_answer','truncated','invalid','infrastructure_failure','generation_hash'] if k in r}) for r in rows]
        task=Task(dataset=MemoryDataset(samples),solver=replay(),scorer=cached_grade(),name='simpleqa_cached_replay',metadata={'post_hoc_replay':True,'source_model':model,'main_data_sha256':'feee3f7e7db3617e94e8fcf1977b756ec420ef8568f4e0fcbbe0e92e9d5fc032','inference_calls':0})
        logs=eval(task,model=replay_model(model),log_dir=str(ROOT/'results/safety/inspect_logs'),log_format='json',display='none',max_connections=64)
        assert logs[0].status=='success'
        logpath=logs[0].location
        cmd=[sys.executable,'.codex/skills/safety-evals/scripts/run_eval.py','--from-log',logpath,'--eval-id','simpleqa','--model-id',model,'--out','results/runtime/converted']
        subprocess.run(cmd,check=True)
        standard=json.loads((ROOT/f'results/runtime/converted/simpleqa__{model}.json').read_text())
        path=ROOT/f'results/safety/simpleqa__{model}.json'; original=json.loads(path.read_text())
        standard.update(original)
        standard['inspect_log']=logpath
        standard['inspect_log_note']='Post-hoc Inspect replay of cached real generations/grades; no new inference. Native log SEs treat completions as independent and must not replace item-cluster analysis.'
        standard['schema_conversion_command']=' '.join(cmd)
        write_json(path,standard)
        print('EXPORTED',model,logpath,flush=True)

if __name__=='__main__':main()
