"""Session supervisor: wait for the current subject, then run remaining models sequentially."""
import argparse,datetime,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def alive(pid):
    try: os.kill(pid,0);return True
    except ProcessLookupError:return False

def main():
    assert Path.cwd()==ROOT; print(ROOT,flush=True)
    p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int);p.add_argument('--wait-engine-pid',type=int);a=p.parse_args()
    for pid in [a.wait_pid,a.wait_engine_pid]:
        while pid and alive(pid):time.sleep(3)
    limits={'deepseek-r1-distill-llama-8b':'2026-10-20T11:45:28+00:00',
            'moonlight-16b-a3b-instruct':'2026-10-20T11:51:28+00:00',
            'qwen2.5-7b-judge':'2026-10-20T11:57:28+00:00'}
    runs=[]
    for model,stamp in limits.items():
        deadline=datetime.datetime.fromisoformat(stamp).timestamp()
        cmd=[sys.executable,'src/run_model.py',model,'--deadline',str(deadline)]
        start=time.time(); print('STARTING',model,datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
        with (ROOT/f'logs/{model}.log').open('w') as log:
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            rc=proc.wait()
        runs.append({'model':model,'command':cmd,'returncode':rc,'started':start,'finished':time.time(),'duration_s':time.time()-start})
        (ROOT/'results/runtime/remaining_runs.json').write_text(json.dumps(runs,indent=2)+'\n')
        print('FINISHED',model,'rc',rc,'seconds',time.time()-start,flush=True)
    print('ALL REMAINING PROCESSES EXITED',flush=True)

if __name__=='__main__':main()
