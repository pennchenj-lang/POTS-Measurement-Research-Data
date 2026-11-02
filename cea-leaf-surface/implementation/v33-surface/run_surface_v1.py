"""V30 bounded camera support selection; numerically unchanged V29 leaf surface fit."""
from pathlib import Path
import os,sys,json,hashlib,importlib.util,argparse,time,traceback,copy,ast,types,datetime
from dataclasses import replace
R=Path(__file__).resolve().parent;V=R.parent;P=V.parents[1]
sys.dont_write_bytecode=True;sys.path.insert(0,str(P/'src'));sys.path.append(str(P/'.venv/Lib/site-packages'))
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
for k in ('TEMP','TMP','TMPDIR','PIP_CACHE_DIR','TORCH_HOME','HF_HOME','MPLCONFIGDIR','XDG_CACHE_HOME'):
 d=R/'cache'/k.lower();d.mkdir(parents=True,exist_ok=True);os.environ[k]=str(d)
import numpy as np,cv2
from PIL import Image,ImageOps,ImageDraw
from pots_gs.triangulation import Camera
from pots_gs.geometry import LEAF_FACES,Provenance,mesh_area
from pots_gs.subdivision import subdivision_levels
cv2.setNumThreads(1)
V24=P/'work/annotation_guided_experiment_20261030_v24'
HELPERS=V24/'surface/run_surface_v1.py'
SOLVER=P/'work/marker_guided_recovery_20261025_v9/code/frozen_normal4_scaffold_v1.py'
GEO=P.parent/'measurement_revision_v039/code/frozen_replay/snapshot/geometry_diagnostics.py'
INIT_SOURCE=P/'work/leaf_target_recovery_20261026_v12/code/given_camera_leaf_surface_v1.py'
DEV=['G01', 'G40', 'P079', 'P094', 'P132', 'P183', 'G20', 'P014', 'P017', 'P173'];CONFLICT=['P074','P174','P180','P181']
RULES=dict(regularization=.2,max_nfev=250,sigma_px=2.,differential_weight=1.,max_displacement=1.)
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
core=load('v25_original_normal4',SOLVER);geometry=load('v25_original_geometry',GEO)
# Only pure helper functions are loaded. V24 top-level caches and experiment runners are not executed.
names={'read','sha','plain','dump','transform','mask_read','contours','prepared','check','split_paths','dashed','overlay','run_fit'}
tree=ast.parse(HELPERS.read_text(encoding='utf8'));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
fit=next(n for n in nodes if n.name=='run_fit');changed=0
for n in ast.walk(fit):
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='solver' for t in n.targets):
  assert ast.unparse(n.value)=='guard.build_solver(SOLVER, segments)'
  n.value=ast.parse("load('v25_unmodified_normal4_fit',SOLVER)",mode='eval').body;changed+=1
assert changed==1
provenance_corrections=0
for n in ast.walk(fit):
 if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('Same-leaf image texture initializes depth;'):
  n.value='Retained candidate-frontier centroid rays set a relative-depth prior; no material-point or texture-XYZ initialization. Original normal4 fit without support-growth guard; canonical B/T/Q are model slots, not marked anatomical observations.'
  provenance_corrections+=1
assert provenance_corrections==1
for block in [n for n in fit.body if isinstance(n,ast.Try)]:
 for i,n in enumerate(block.body):
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='fitted' for t in n.targets):
   block.body.insert(i+1,ast.parse("fitted.diagnostics.update(support_guard_enabled_for_optimization=False,initializer_semantics='retained candidate-frontier centroid rays as prior, not material observations')").body[0])
   break
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(HELPERS)+' [original solver loader only]', 'exec'),globals())
itree=ast.parse(INIT_SOURCE.read_text(encoding='utf8'))
inodes=[n for n in itree.body if isinstance(n,ast.FunctionDef) and n.name=='initialization' or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RULES' for t in n.targets)]
initmod=types.ModuleType('v25_retained_centroid_initializer');initmod.__dict__.update(np=np,hashlib=hashlib)
exec(compile(ast.Module(body=inodes,type_ignores=[]),str(INIT_SOURCE)+' [initializer only]','exec'),initmod.__dict__)

def initial(cameras,pp,texture_unused=None):
 actual=[p['view_id'] for p in pp];assert len(actual)==2
 forward=dict(zip(actual,['01','02']));reverse={v:k for k,v in forward.items()}
 cc={forward[v]:Camera(forward[v],c.K,c.R,c.t,unit=c.unit) for v,c in cameras.items()}
 pq=[dict(p,view_id=forward[p['view_id']]) for p in pp]
 xyz,audit=initmod.initialization(cc,pq)
 audit.update(xyz=xyz,actual_to_initializer_view_alias=forward,initializer_to_actual_view=reverse,
  actual_reference_view=actual[0],initializer_source_sha256=sha(INIT_SOURCE),
  canonical_slots_not_observed_B_T=True,Q_observed=False,identity_locator_used_for_depth=False,
  centroids_are_material_points=False,depth_is_estimated_prior=True,support_guard_in_optimization=False,
  retained_W_sha256={p['view_id']:hashlib.sha256(p['W'].astype('<f8').tobytes()).hexdigest() for p in pp})
 return xyz,audit

def protocol():
 return dict(schema='v33-unchanged-surface-on-track-checked-camera',development_groups=DEV,
  solver_rules=RULES,normal_weight=4.,support_guard_enabled=False,strict_work_RMS=8.,exploration_work_RMS=16.,
  topology='11 controls/12 triangles; unchanged shared-midpoint L1/L2 with organ lineage',
  identity_input=str(P/'work/same_leaf_combined_20261030_v23/bundle_v1/identity_constraints.json'),
  segmentation_input=str(V/'segmentation/groups'),
  camera_input=str(V/'camera/results'),
  initializer='exact retained-only W centroid-ray depth prior and first-view PCA plane; centroid correspondence is not material correspondence',
  no_leaf_texture_minimum_for_surface=True,identity_locators_triangulated=False,
  boundary_semantics='predicted visible segmentation frontier; contact-adjacent edges removed; not independently reviewed true margin',
  group_endpoint='any annotated leaf passing complete camera/surface/optimizer checks, with note-conflict exclusion unchanged',
  controlled_fractions=[.2,.4,.6],controlled_primary='lowest natural leaf ID with candidates in both selected views, fixed before fitting',
  controlled_reference='withheld predicted frontier; same retained-two-view initializer for two-view and first-view residual control',
  source_selection='fixed before surface results; never switch camera after a leaf fails',
  inference_scope='posthoc internal consistency; no independent natural hidden truth or metric scale',
  hashes={str(p):sha(p) for p in [Path(__file__),HELPERS,SOLVER,GEO,INIT_SOURCE,V/'RUN_SCOPE_v1.json']})

def run_group(cid,phash):
 dest=R/'results'/cid
 if (dest/'group_result.json').exists():return read(dest/'group_result.json')
 cp=V/'camera/results'/cid/'camera_result.json';sp=V/'segmentation/groups'/cid/'segmentation.json'
 if not cp.exists() or not sp.exists():return None
 camera=read(cp);seg=read(sp)
 rec=dict(group_id=cid,camera_result_path=str(cp),camera_sha256=sha(cp),segmentation_sha256=sha(sp),protocol_sha256=phash,
  selected_camera_source=camera.get('selected_source'),strict_camera_gate_passed=camera.get('strict_camera_gate_passed',False),
  exploration_camera_gate_passed=camera.get('exploration_camera_gate_passed',False),
  strict_complete_pass=False,exploration_complete_pass=False,any_leaf_strict_descriptive=False,any_leaf_exploration_descriptive=False,
  primary_leaf_id=None,leaves=[],note_conflict=cid in CONFLICT,natural_hidden_accuracy=None,
  support_guard_enabled_for_optimization=False)
 if not camera.get('fixed_middle_camera') or not camera.get('exploration_camera_gate_passed'):
  rec.update(status='camera_unavailable',reason=camera.get('failure_reasons',camera.get('status')));dump(dest/'group_result.json',rec);return rec
 metas={v['view_id']:v for v in camera['views']};pair=camera['actual_view_pair']
 camera['cameras_work']={v:dict(K=q['K_work'],R=q['R'],t=q['t']) for v,q in camera['fixed_middle_camera']['poses_by_view_id'].items()}
 sm={v['view_id']:v for v in seg['views']}
 for v in pair:
  assert sm[v]['image_sha256']==metas[v]['source_sha256']
 def order(l):
  digits=''.join(c for c in l['leaf_id'] if c.isdigit());return (int(digits) if digits else 10**9,l['leaf_id'])
 candidates=[]
 for leaf in sorted(seg['leaves'],key=order):
  eligible=all(leaf['views'].get(v,{}).get('status')=='candidate_generated' for v in pair)
  record=dict(leaf_id=leaf['leaf_id'],candidate_both_views=eligible,valid_nonheldout_texture_points=None,
   initialization_support='retained candidate-frontier depth prior, not triangulated locator or texture')
  if eligible:candidates.append((leaf,record))
  else:record.update(status='input_ineligible',reason='candidate mask unavailable in one of selected two views');rec['leaves'].append(record)
 if candidates:rec['primary_leaf_id']=candidates[0][0]['leaf_id']
 selection=dest/'selection_before_surface.json'
 if not selection.exists():dump(selection,dict(primary_leaf_id=rec['primary_leaf_id'],candidate_order=[x[0]['leaf_id'] for x in candidates],camera_sha256=sha(cp),source_chosen_before_any_surface=True))
 for leaf,record in candidates:
  lid=leaf['leaf_id'];base=dest/lid
  try:
   paths={v:contours(leaf,seg,v,metas[v]) for v in pair}
   record['full']=run_fit(base/'full'/'two_view',cid,leaf,camera,metas,paths,None,'full','two_view',phash)
   if lid==rec['primary_leaf_id']:
    rec.update(strict_complete_pass=record['full']['strict_complete_pass'],exploration_complete_pass=record['full']['exploration_complete_pass'])
    record['controlled']=[]
    for fraction in (.2,.4,.6):
     cond=f'withhold_{round(fraction*100)}'
     try:
      kept,held=split_paths(paths,fraction)
      for branch in ('two_view','first_view_W_only'):
       record['controlled'].append(run_fit(base/cond/branch,cid,leaf,camera,metas,kept,None,cond,branch,phash,held))
     except Exception as e:record['controlled'].append(dict(condition=cond,status='input_unavailable',reason=str(e)))
  except Exception as e:record.update(status='frontier_or_initialization_unavailable',reason=str(e),traceback=traceback.format_exc())
  rec['leaves'].append(record)
 rec.update(status='processed',computed_leaf_models=sum(x.get('full',{}).get('status')=='surface_computed' for x in rec['leaves']),
  any_leaf_strict_descriptive=any(x.get('full',{}).get('strict_complete_pass',False) for x in rec['leaves']),
  any_leaf_exploration_descriptive=any(x.get('full',{}).get('exploration_complete_pass',False) for x in rec['leaves']))
 dump(dest/'group_result.json',rec)
 print(json.dumps({k:rec[k] for k in ['group_id','computed_leaf_models','any_leaf_strict_descriptive','any_leaf_exploration_descriptive']},ensure_ascii=False),flush=True)
 return rec

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['freeze','run']);ap.add_argument('--groups',nargs='*');ap.add_argument('--part',type=int,choices=[0,1]);z=ap.parse_args();pp=R/'protocol_v1.json'
 if z.command=='freeze':dump(pp,protocol());print(str(pp));raise SystemExit
 for f,h in read(pp)['hashes'].items():assert sha(f)==h
 groups=z.groups or [g['id'] for g in read(P/'work/same_leaf_combined_20261030_v23/bundle_v1/manifest.json')['groups']]
 if z.part is not None:groups=[g for i,g in enumerate(groups) if i%2==z.part]
 for cid in groups:
  try:run_group(cid,sha(pp))
  except Exception as e:print(json.dumps(dict(group_id=cid,error=str(e),traceback=traceback.format_exc())),flush=True)
