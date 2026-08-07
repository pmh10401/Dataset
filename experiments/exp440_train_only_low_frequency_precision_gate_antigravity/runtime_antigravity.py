from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from .config_antigravity import cpu_worker_count_antigravity
def ordered_parallel_map_antigravity(values,function,*,workers):
 if workers!=cpu_worker_count_antigravity():raise ValueError('official CPU workers are fixed at max-minus-one')
 with ThreadPoolExecutor(max_workers=workers) as pool:return tuple(pool.map(function,values))
