#!/usr/bin/env python3
import re
import statistics
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

log_path = Path('logs/pipeline.log')
out_dir = Path('result/mt_online/virtual_test')
out_dir.mkdir(parents=True, exist_ok=True)

patterns = {
    'datareader_read_ms': re.compile(r'Read data\s*cost:\s*([0-9]+\.[0-9]+) ms'),
    'infer_ms': re.compile(r'Inference cost:\s*([0-9]+\.[0-9]+) ms'),
    'refine_ms': re.compile(r'Refine cost:\s*([0-9]+\.[0-9]+) ms'),
    'refine_parallel_ms': re.compile(r'refine_parallel_cost_ms:\s*([0-9]+\.[0-9]+)ms'),
    'pose_ms': re.compile(r'Pose process cost:\s*([0-9]+\.[0-9]+) ms'),
    'visualize_process_ms': re.compile(r'Visualize process cost:\s*([0-9]+\.[0-9]+) ms'),
    'visualize_total_ms': re.compile(r'Visualize total since recv:\s*([0-9]+\.[0-9]+) ms'),
}

values = {k: [] for k in patterns}

with open(log_path, 'r', encoding='utf-8', errors='ignore') as fh:
    for line in fh:
        for k, rx in patterns.items():
            m = rx.search(line)
            if m:
                try:
                    values[k].append(float(m.group(1)))
                except:
                    pass

# compute stats
stats = {}
for k, nums in values.items():
    if not nums:
        stats[k] = None
        continue
    stats[k] = {
        'count': len(nums),
        'min': min(nums),
        'max': max(nums),
        'mean': statistics.mean(nums),
        'median': statistics.median(nums),
        'stdev': statistics.pstdev(nums) if len(nums)>1 else 0.0,
        'p90': sorted(nums)[int(0.9*len(nums))-1],
        'p95': sorted(nums)[int(0.95*len(nums))-1],
    }

# write CSV of raw samples
for k, nums in values.items():
    if not nums:
        continue
    out_csv = out_dir / f'{k}_samples.csv'
    with open(out_csv, 'w', newline='') as of:
        w = csv.writer(of)
        w.writerow(['sample_ms'])
        for v in nums:
            w.writerow([v])

# write summary CSV
with open(out_dir / 'latency_summary.csv', 'w', newline='') as of:
    w = csv.writer(of)
    w.writerow(['stage','count','min','max','mean','median','stdev','p90','p95'])
    for k, s in stats.items():
        if s is None:
            w.writerow([k,0,'','','','','','',''])
        else:
            w.writerow([k,s['count'],s['min'],s['max'],s['mean'],s['median'],s['stdev'],s['p90'],s['p95']])

# generate simple boxplots
import numpy as np
for k, nums in values.items():
    if not nums:
        continue
    fig, ax = plt.subplots(figsize=(6,4))
    ax.boxplot(nums, vert=False)
    ax.set_title(k)
    ax.set_xlabel('ms')
    fig.tight_layout()
    fig.savefig(out_dir / f'{k}_box.png')
    plt.close(fig)

# print summary
for k, s in stats.items():
    if s is None:
        print(f'{k}: no samples')
    else:
        print(f"{k}: count={s['count']} mean={s['mean']:.3f} ms median={s['median']:.3f} ms p95={s['p95']:.3f} ms max={s['max']:.3f} ms")

print('Wrote results to', out_dir)
