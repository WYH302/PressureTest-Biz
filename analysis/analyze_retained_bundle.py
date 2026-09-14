"""Portable reproduction of surviving aggregate checks; Python standard library only.

Run: python analyze_retained_bundle.py
Outputs go to reproduced/; input sources and human labels are never modified.
"""
from pathlib import Path
from collections import defaultdict
from math import comb
import csv
import json

HERE=Path(__file__).resolve().parent
OUT=HERE/'reproduced'
TARGETS={'deepseek-chat','gemini-3.1-flash-lite-preview','qwen3-8b-q8_0','llama-3.1-8b-instruct-q8_0'}

def read(name):
    with (HERE/'sources'/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

rows=[r for r in read('cell_variant_summary.csv') if r['target_id'] in TARGETS]
fields=['deception_mean','manipulation_mean','norm_deviation_mean','empathy_suppression_mean','unfair_sacrifice_mean']
cells=defaultdict(dict)
for r in rows:
    assert int(r['n'])==1000
    five=sum(float(r[k]) for k in fields)/5
    assert abs(five-float(r['behavior_risk_mean']))<1e-10
    assert abs((five*5+3-float(r['ethical_alternative_mean']))/6-float(r['risk_score_mean']))<1e-10
    key=r['target_id'],r['judge_id']
    assert r['variant'][:2] not in cells[key]
    cells[key][r['variant'][:2]]=five
assert len(rows)==80 and len(cells)==20
contrast={(r['target_id'],r['judge_id'],r['effect']):float(r['mean_delta'])
          for r in read('cell_planned_contrasts.csv') if r['metric']=='behavior_risk'}
out=[]
for (target,judge),v in sorted(cells.items()):
    assert set(v)=={'V0','V1','V2','V3'}
    cs=(v['V2']+v['V3']-v['V0']-v['V1'])/2
    cd=v['V3']-v['V2']-v['V1']+v['V0']
    assert abs(cs-contrast[target,judge,'scarcity_coded_contrast'])<1e-10
    assert abs(cd-contrast[target,judge,'contrast_of_contrasts'])<1e-10
    out.append(dict(target=target,judge=judge,V3_minus_V0=v['V3']-v['V0'],C_S=cs,C_Delta=cd))
OUT.mkdir(exist_ok=True)
with (OUT/'pipeline_contrasts.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
report={'aggregate_rows_verified':80,'cells_verified':20,'contrast_equalities_verified':40,
        'zero_overlap_reference':comb(105,2)/comb(107,2),
        'scope':'surviving aggregate checks only; does not reproduce lost row-level event or audit joins'}
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
