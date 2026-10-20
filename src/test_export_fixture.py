"""Software-only replay plumbing fixture; never interpreted as model evidence."""
import traceback
from pathlib import Path
try:
    from export_inspect import *
    print('imports ready',flush=True)
    task=Task(dataset=MemoryDataset([Sample(id='software-fixture',input='Replay plumbing test',target='',metadata={'id':'software-fixture','condition':'fixture','seed':0,'grade':None,'final_answer':'cached fixture; no inference'})]),solver=replay(),scorer=cached_grade(),name='software_replay_fixture')
    print('task ready',flush=True)
    logs=eval(task,model=replay_model('fixture'),log_dir='results/runtime/inspect_fixture',log_format='json',display='none')
    print(logs[0].status,logs[0].location,flush=True)
    assert logs[0].status=='success'
except BaseException:
    Path('logs/export_fixture_error.log').write_text(traceback.format_exc())
    raise
