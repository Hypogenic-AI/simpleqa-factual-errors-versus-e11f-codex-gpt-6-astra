"""Frozen protocol and final-answer extraction; never classify parser failures as abstention."""
import hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SEEDS=[11,22,33]
SUFFIX='\n\nIf you are uncertain, answer exactly I do not know.'
MODELS=['qwen3-8b','deepseek-r1-distill-llama-8b','moonlight-16b-a3b-instruct']

def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def extract_final(model,text):
    if model.startswith('deepseek'):
        if '</think>' not in text:
            return '', 'missing_final_boundary'
        return text.split('</think>',1)[1].strip(), 'ok'
    return text.strip(),'ok'

def parse_grade(text):
    text=text.strip()
    return text if text in ('A','B','C') else None

def grade_key(question,gold,response,judge):
    return digest([question,gold,response,judge,(ROOT/'artifacts/resource_finder/grader_prompt.txt').read_text(),{'temperature':0,'max_tokens':16}])

def fixtures():
    assert extract_final('deepseek-r1-distill-llama-8b','reasoning only')==('','missing_final_boundary')
    assert extract_final('deepseek-r1-distill-llama-8b','I guess X</think>\nY')==('Y','ok')
    assert extract_final('qwen3-8b','I do not know')==('I do not know','ok')
    for valid in ['A',' B\n','C']:
        assert parse_grade(valid)==valid.strip()
    for invalid in ['', 'A or B', 'CORRECT', 'C because uncertain']:
        assert parse_grade(invalid) is None
    return {'parser_fixtures':'passed','cases':10,'synthetic_software_checks_not_judge_calibration':True}
