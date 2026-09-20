"""Reuse the existing dependency scanner for the additive Exp492 map."""
import argparse, hashlib, html, json
from pathlib import Path
from build_codemap_exp489_491_antigravity import generate, ROOT

def build():
    result = generate('exp492', 'experiments/exp492_stable_normalization_antigravity')
    data = json.loads(result['json'])
    p = ROOT / 'experiments/exp492_stable_normalization_antigravity/protocol_antigravity.html'
    data['nodes'].append(dict(path=str(p.relative_to(ROOT)), callers=['run_antigravity.py'], effect='frozen Exp492 protocol', tests=['test_exp492_antigravity.py'], sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    data['flow'] = 'readonly normal TRAIN and unlabeled TEST -> shared raw encoders -> separate legacy/stable normalization and LocalGap -> all 125 score checkpoints -> TEST labels -> paired evaluation'
    result['json'] = (json.dumps(data, indent=2, sort_keys=True)+'\n').encode()
    result['html'] = ('<!doctype html><meta charset="utf-8"><title>Exp492 code map</title><pre>'+html.escape(result['json'].decode())+'</pre>').encode()
    result['lock'] = (json.dumps({k+'_sha256':hashlib.sha256(result[k]).hexdigest() for k in ('json','html')},indent=2)+'\n').encode()
    return result
if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--validate',action='store_true'); args=parser.parse_args()
    for ext, data in build().items():
        p=ROOT/f'docs/codemap/codemap_exp492_antigravity.{ext}'
        if args.validate:
            assert p.read_bytes()==data, f'stale map: {p}'
        else: p.write_bytes(data)
