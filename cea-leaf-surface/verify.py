"""Rebuild cohort counts and paired-boundary errors from the released records."""
from pathlib import Path
import csv,json,math,statistics
from collections import defaultdict
root=Path(__file__).resolve().parent
def rows(name):
    with (root/'data'/name).open(encoding='utf8') as f:return list(csv.DictReader(f))
def dist(q,a,b):
    dx,dy=b[0]-a[0],b[1]-a[1];dd=dx*dx+dy*dy
    t=max(0,min(1,((q[0]-a[0])*dx+(q[1]-a[1])*dy)/dd)) if dd else 0
    return math.hypot(q[0]-a[0]-t*dx,q[1]-a[1]-t*dy)
groups=rows('group_outcomes.csv')
assert len({v['group_id'] for v in groups})==len(groups)==248
assert sum(v['retained_exploration']=='True' for v in groups)==214
assert sum(v['retained_strict']=='True' for v in groups)==131
views=rows('views.csv');marks=rows('leaf_locators.csv')
assert len(views)==742 and len({(x['group'],x['leaf']) for x in marks})==769
assert sum(bool(x['display_x']) for x in marks)==2268
pairs=defaultdict(dict);edge=[0,1,4,7,10,9,6,3,0]
for row in rows('paired_fits.csv'):
    g,l,c,b=[row[k] for k in ('group_id','leaf_id','condition','branch')]
    record=json.loads((root/'paired_inputs'/g/l/c/(b+'.json')).read_text(encoding='utf8'))
    held=record['heldout'];verts=record['projections'][held['view_id']]['vertices_work']
    path=[verts[i] for i in edge]
    ds=[min(dist(q,a,z) for a,z in zip(path,path[1:])) for q in held['points_work']]
    rms=(sum(x*x for x in ds)/len(ds))**.5
    assert math.isclose(rms,float(row['heldout_RMS_work_px']),rel_tol=2e-7,abs_tol=1e-7),(g,c,b,rms,row['heldout_RMS_work_px'])
    pairs[c].setdefault(g,{})[b]=rms
for c,(a,b,k) in zip(sorted(pairs),[(2.92,3.71,3),(8.05,4.62,6),(19.74,7.26,6)]):
    cases=pairs[c].values();x=[z['first_view_W_only'] for z in cases];y=[z['two_view'] for z in cases]
    assert (round(statistics.median(x),2),round(statistics.median(y),2),sum(j<i for i,j in zip(x,y)))==(a,b,k)
    print(c,a,b,k,'of',len(x))
print('Verified cumulative group counts, identity records and 36 held-out boundary fits.')

