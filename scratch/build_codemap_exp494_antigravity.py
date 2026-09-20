"""Additive Exp494 map; previous experiment maps remain untouched."""
import argparse,hashlib,html,json
from build_codemap_exp489_491_antigravity import generate,ROOT

def build():
    out=generate('exp494','experiments/exp494_all_datasets_low_fpr_antigravity');data=json.loads(out['json'])
    for relative,effect in [('experiments/exp494_all_datasets_low_fpr_antigravity/protocol_antigravity.html','frozen evaluation extension protocol'),('docs/CURRENT_LOW_FPR_COMPARISON_antigravity.json','prior 125-family metric replay')]:
        p=ROOT/relative;data['nodes'].append(dict(path=relative,callers=['experiments/exp494_all_datasets_low_fpr_antigravity/run_antigravity.py'],effect=effect,tests=['experiments/exp494_all_datasets_low_fpr_antigravity/test_exp494_antigravity.py'],sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    data['flow']='frozen score caches or normal-TRAIN fits -> all score/descriptor checkpoints -> fixed Exp493 donors with query family excluded -> persisted routes -> retrospective TEST ROC -> full and common-coverage family/dataset macro metrics'
    out['json']=(json.dumps(data,indent=2,sort_keys=True)+'\n').encode();out['html']=('<!doctype html><meta charset="utf-8"><title>Exp494 code map</title><pre>'+html.escape(out['json'].decode())+'</pre>').encode()
    out['lock']=(json.dumps({k+'_sha256':hashlib.sha256(out[k]).hexdigest() for k in ('json','html')},indent=2)+'\n').encode();return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--validate',action='store_true');a=p.parse_args()
    for ext,content in build().items():
        target=ROOT/f'docs/codemap/codemap_exp494_antigravity.{ext}'
        if a.validate:assert target.read_bytes()==content,str(target)
        else:target.write_bytes(content)
