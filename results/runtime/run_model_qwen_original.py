"""Sequential, resumable vLLM subject generation and blinded local grading.
Run one process at a time on the allocated CUDA_VISIBLE_DEVICES GPU.
"""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HOME']=str(ROOT/'models/.hf')
os.environ['XDG_CACHE_HOME']=str(ROOT/'models/.cache')
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(ROOT/'models/.torchinductor')
os.environ['VLLM_CACHE_ROOT']=str(ROOT/'models/.vllm')
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse, json, time
from protocol import *

def stop_state(ids,boundary_ids):
    """Return final-token count after the first exact native closing boundary."""
    size=len(boundary_ids)
    for index in range(len(ids)-size+1):
        if list(ids[index:index+size])==boundary_ids:
            return len(ids)-index-size,index+size
    return None,None

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('model',choices=MODELS+['qwen2.5-7b-judge'])
    parser.add_argument('--fixture',action='store_true')
    parser.add_argument('--deadline',type=float,default=0)
    args=parser.parse_args()
    assert Path.cwd()==ROOT
    print(ROOT,flush=True)
    import vllm
    from vllm import LLM,SamplingParams
    from vllm.sampling_params import RequestOutputKind
    model=args.model
    cat=next(x for x in json.loads((ROOT/'artifacts/resource_finder/catalog_models.json').read_text()) if x['id']==model)
    start=time.time()
    actual_memory={'qwen3-8b':0.45,'deepseek-r1-distill-llama-8b':0.45,'moonlight-16b-a3b-instruct':0.50,'qwen2.5-7b-judge':0.42}[model]
    llm=LLM(model=str(ROOT/'models'/model),tokenizer=str(ROOT/'models'/model),
            trust_remote_code=model.startswith('moonlight'),dtype='bfloat16',
            tensor_parallel_size=1,gpu_memory_utilization=actual_memory,
            max_model_len=8192,max_num_seqs=64,max_num_batched_tokens=8192,
            enforce_eager=True,enable_prefix_caching=True,seed=20261020,
            generation_config='vllm',disable_log_stats=True)
    tok=llm.get_tokenizer()
    is_deep=model.startswith('deepseek')
    boundary=tok.encode('</think>',add_special_tokens=False) if is_deep else []
    if is_deep:
        assert tok.decode(boundary)=='</think>', (boundary,tok.decode(boundary))
        assert stop_state([7]+boundary+[3]*256,boundary)==(256,1+len(boundary))
        assert stop_state([7],boundary)==(None,None)
        assert stop_state(boundary+[3]*255,boundary)[0]==255
    def prompt(content):
        return tok.apply_chat_template([{'role':'user','content':content}],tokenize=False,
                 add_generation_prompt=True,**({'enable_thinking':False} if model=='qwen3-8b' else {}))
    gen_config=json.loads((ROOT/'models'/model/'generation_config.json').read_text())
    sampling={'temperature':0 if model=='qwen2.5-7b-judge' else 0.6,
              'top_p':gen_config.get('top_p',1.0), 'top_k':gen_config.get('top_k',-1),
              'repetition_penalty':gen_config.get('repetition_penalty',1.0)}
    if model=='qwen2.5-7b-judge': sampling.update(top_p=1.0,top_k=-1,repetition_penalty=1.0)
    write_json(ROOT/f'results/runtime/{model}_resolved.json',{
        'model':model,'sampling':sampling,'dtype':'bfloat16','engine':'vLLM offline step API',
        'vllm_version':vllm.__version__,'max_num_seqs':64,'tensor_parallel_size':1,
        'max_model_len':8192,'memory_utilization':actual_memory,'catalog_memory_utilization':cat['gpu_memory_utilization'],
        'reasoning_boundary_ids':boundary,'enforce_eager':True,'prefix_caching':True,
        'chat_template':tok.chat_template,'example_render':prompt('What is 2 + 2?'),
        'final_budget':256,'total_cap':4096,'generation_config':gen_config,
        'freeze_time':time.time(),'load_seconds':time.time()-start})
    if True:  # Non-main smoke fixture is always completed before main collection.
        out=llm.generate([prompt('What is 2 + 2?')],SamplingParams(max_tokens=16,temperature=0),use_tqdm=False)
        write_json(ROOT/f'results/runtime/{model}_fixture.json',{'text':out[0].outputs[0].text,'token_ids':out[0].outputs[0].token_ids,'model':model,'native_boundary_ids':boundary,'fixture_only':True})
        if args.fixture: return
    if model=='qwen2.5-7b-judge':
        run_judge(llm,prompt,SamplingParams,sampling,args.deadline)
        return
    items=read_jsonl(ROOT/'datasets/main_questions.jsonl')
    outpath=ROOT/f'results/raw/{model}.jsonl'
    done=set()
    if outpath.exists():
        done={(r['id'],r['condition'],r['seed']) for r in read_jsonl(outpath)}
    records=[]
    for item in items:
        for seed in SEEDS:
            for cond in ['baseline','abstention']:
                if (item['id'],cond,seed) not in done:
                    content=item['problem']+(SUFFIX if cond=='abstention' else '')
                    records.append({'id':item['id'],'seed':seed,'condition':cond,'prompt':prompt(content)})
    engine=llm.llm_engine
    # Bounded batches preserve checkpoints and allow final-budget checks at each engine step.
    with outpath.open('a') as sink:
        for offset in range(0,len(records),64):
            if args.deadline and time.time()>args.deadline:
                print('DEADLINE checkpoint',offset,flush=True); break
            active={}
            boundary_ends={}
            previous_lengths={}
            for ix,r in enumerate(records[offset:offset+64]):
                request_id=str(offset+ix)
                params=SamplingParams(**sampling,seed=r['seed'],max_tokens=4096 if is_deep else 256,
                    output_kind=RequestOutputKind.CUMULATIVE,skip_special_tokens=False)
                engine.add_request(request_id,r['prompt'],params)
                active[request_id]=r
            batch_start=time.time()
            while engine.has_unfinished_requests():
                outputs=engine.step()
                for out in outputs:
                    if out.request_id not in active: continue
                    answer=out.outputs[0]
                    ids=list(answer.token_ids)
                    if is_deep:
                        boundary_end=boundary_ends.get(out.request_id)
                        if boundary_end is None:
                            scan_start=max(0,previous_lengths.get(out.request_id,0)-len(boundary)+1)
                            _,relative_end=stop_state(ids[scan_start:],boundary)
                            if relative_end is not None:
                                boundary_end=scan_start+relative_end
                                boundary_ends[out.request_id]=boundary_end
                        previous_lengths[out.request_id]=len(ids)
                        final_count=None if boundary_end is None else len(ids)-boundary_end
                    else:
                        final_count,boundary_end=len(ids),0
                    reached_final=final_count is not None and final_count>=256
                    if not (out.finished or reached_final): continue
                    r=active.pop(out.request_id)
                    if reached_final and not out.finished: engine.abort_request(out.request_id)
                    assert final_count is None or final_count<=256, 'engine step overshot final budget'
                    raw=tok.decode(ids,skip_special_tokens=False)
                    final,status=extract_final(model,raw)
                    # EOS is a structural token, not answer content.
                    if final and ids and ids[-1] in set(gen_config.get('eos_token_id',[]) if isinstance(gen_config.get('eos_token_id'),list) else [gen_config.get('eos_token_id')]):
                        final,status=extract_final(model,tok.decode(ids[:-1],skip_special_tokens=False))
                    truncated=reached_final or (answer.finish_reason=='length')
                    row={**r,'model':model,'raw_text':raw,'raw_token_ids':ids,'final_answer':final,
                         'parse_status':status,'invalid':status!='ok' or not final,
                         'truncated':truncated,'infrastructure_failure':False,
                         'final_tokens':final_count,'reasoning_tokens':boundary_end if is_deep else 0,
                         'output_tokens':len(ids),'prompt_tokens':len(out.prompt_token_ids),
                         'finish_reason':'final_token_cap' if reached_final else answer.finish_reason,
                         'literal_abstention':final=='I do not know','attempt':1,'rerun':False}
                    row['generation_hash']=digest([model,r['prompt'],r['seed'],ids])
                    sink.write(json.dumps(row,ensure_ascii=False)+'\n'); sink.flush()
            assert not active
            print(json.dumps({'completed':min(offset+64,len(records)),'pending_total':len(records),'batch_seconds':time.time()-batch_start,'elapsed':time.time()-start}),flush=True)
    print('FINISHED',model,time.time()-start,flush=True)

def run_judge(llm,prompt,SamplingParams,sampling,deadline):
    judge='qwen2.5-7b-judge'
    rubric=(ROOT/'artifacts/resource_finder/grader_prompt.txt').read_text()
    controls=read_jsonl(ROOT/'datasets/calibration_controls.jsonl')
    tasks={}
    mappings=[]
    for r in controls:
        k=grade_key(r['question'],r['gold'],r['response'],judge)
        tasks[k]=(r['question'],r['gold'],r['response'])
        mappings.append({'key':k,'kind':'control',**r})
    itemmap={r['id']:r for r in read_jsonl(ROOT/'datasets/main_questions.jsonl')}
    for model in MODELS:
        p=ROOT/f'results/raw/{model}.jsonl'
        if not p.exists(): continue
        for r in read_jsonl(p):
            item=itemmap[r['id']]
            # Grade every available final answer; retain truncation separately, never relabel it as clean abstention.
            valid=not r['invalid']
            k=grade_key(item['problem'],item['answer'],r['final_answer'],judge) if valid else None
            if valid: tasks[k]=(item['problem'],item['answer'],r['final_answer'])
            mappings.append({'key':k,'kind':'main','id':r['id'],'model':model,'condition':r['condition'],'seed':r['seed'],'generation_hash':r['generation_hash']})
    write_json(ROOT/'results/calibration/local_mapping.json',mappings)
    cachepath=ROOT/'results/calibration/local_cache.jsonl'
    cache={r['key']:r for r in read_jsonl(cachepath)} if cachepath.exists() else {}
    control_keys={r['key'] for r in mappings if r['kind']=='control'}
    with cachepath.open('a') as f:
        for phase in ['controls','main']:
            pending=[(key,fields) for key,fields in tasks.items() if key not in cache and ((key in control_keys)==(phase=='controls'))]
            for offset in range(0,len(pending),64):
                if deadline and time.time()>deadline: print('DEADLINE local judge checkpoint',flush=True); break
                batch=pending[offset:offset+64]
                prompts=[prompt(rubric.format(question=q,target=g,predicted_answer=a)) for k,(q,g,a) in batch]
                results=llm.generate(prompts,SamplingParams(**sampling,max_tokens=16,seed=20261020),use_tqdm=False)
                for (key,fields),out in zip(batch,results):
                    a=out.outputs[0]
                    row={'key':key,'judge':judge,'raw_text':a.text,'grade':parse_grade(a.text),'input_tokens':len(out.prompt_token_ids),'output_tokens':len(a.token_ids),'finish_reason':a.finish_reason,'attempt':1}
                    cache[key]=row
                    f.write(json.dumps(row,ensure_ascii=False)+'\n'); f.flush()
                print('JUDGED',phase,min(offset+64,len(pending)),len(pending),flush=True)
            if phase=='controls':
                from analyze import calibration
                control_result=calibration(cache,mappings)
                write_json(ROOT/'results/calibration/local_summary.json',control_result)
                print('CONTROL_CALIBRATION',json.dumps(control_result),flush=True)
    print('FINISHED local judge',flush=True)

if __name__=='__main__': main()
