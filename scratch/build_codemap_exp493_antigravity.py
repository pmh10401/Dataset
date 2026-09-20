"""Additive Exp493 map using the existing experiment dependency scanner."""
import argparse,hashlib,html,json
from build_codemap_exp489_491_antigravity import generate,ROOT

def build():
    out=generate('exp493','experiments/exp493_stable_meta_routing_antigravity')
    data=json.loads(out['json']); p=ROOT/'experiments/exp493_stable_meta_routing_antigravity/protocol_antigravity.html'
    if p.exists(): data['nodes'].append(dict(path=str(p.relative_to(ROOT)),callers=['run_antigravity.py'],effect='frozen Exp493 protocol',tests=['test_exp493_antigravity.py'],sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    data['flow']='normal TRAIN caches + paired B OOF -> stable18D -> 24D/old42D/stable42D family-held-out Ridge -> frozen decisions -> current four-method development outcomes; originals immutable'
    out['json']=(json.dumps(data,indent=2,sort_keys=True)+'\n').encode()
    out['html']=('<!doctype html><meta charset="utf-8"><title>Exp493 code map</title><pre>'+html.escape(out['json'].decode())+'</pre>').encode()
    out['lock']=(json.dumps({k+'_sha256':hashlib.sha256(out[k]).hexdigest() for k in ('json','html')},indent=2)+'\n').encode()
    return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--validate',action='store_true');a=p.parse_args()
    for ext,content in build().items():
        target=ROOT/f'docs/codemap/codemap_exp493_antigravity.{ext}'
        if a.validate: assert target.read_bytes()==content,str(target)
        else:target.write_bytes(content)
