"""Identity-guided candidate surface experiment, with immutable receipts.

Locator clicks identify organs; none are material points. Candidate frontiers
are not reviewed anatomical margins. Texture geometry only seeds a depth prior.
"""
from pathlib import Path
import os, sys, json, hashlib, importlib.util, argparse, time, traceback, copy
from dataclasses import replace
R = Path(__file__).resolve().parent
V = R.parent
P = V.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(P / 'src'))
sys.path.append(str(P / '.venv/Lib/site-packages'))
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
for k in ('TEMP','TMP','TMPDIR','PIP_CACHE_DIR','TORCH_HOME','HF_HOME','MPLCONFIGDIR','XDG_CACHE_HOME'):
    d=R/'cache'/k.lower(); d.mkdir(parents=True,exist_ok=True); os.environ[k]=str(d)
import numpy as np
import cv2
cv2.setNumThreads(1)
from PIL import Image, ImageOps, ImageDraw
from pots_gs.triangulation import Camera
from pots_gs.geometry import LEAF_FACES, Provenance, mesh_area
from pots_gs.subdivision import subdivision_levels

SOLVER=P/'work/marker_guided_recovery_20261025_v9/code/frozen_normal4_scaffold_v1.py'
GUARD=P/'work/reconstruction_repair_20261028_v16/surface_development/code/visible_support_guard_v1.py'
GEO=P.parent/'measurement_revision_v039/code/frozen_replay/snapshot/geometry_diagnostics.py'
DEV=['G03','G12','G35','P008','P014','P072','P090','P091','P117']
CONFLICT=['P074','P174','P180','P181']
RULES=dict(regularization=.2,max_nfev=250,sigma_px=2.,differential_weight=1.,max_displacement=1.)

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m
core=load('v24_surface_original',SOLVER)
guard=load('v24_surface_guard',GUARD)
geometry=load('v24_surface_geometry',GEO)

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plain(v):
    if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    return v
def dump(p,v):
    p=Path(p); assert p.resolve().is_relative_to(R.resolve()); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(plain(v),f,ensure_ascii=False,indent=2,allow_nan=False)
def transform(xy,A):
    xy=np.asarray(xy,float).reshape(-1,2); h=np.c_[xy,np.ones(len(xy))]@np.asarray(A).T; return h[:,:2]/h[:,2,None]
def mask_read(path):return cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)

def protocol():
    return dict(schema='v24-identity-candidate-surface-protocol-v1',development_groups=DEV,
      solver_rules=RULES,normal_weight=4.,strict_work_RMS=8.,exploratory_work_RMS=16.,
      topology='fixed11/12; L1 and L2 midpoint-only with exact face lineage',
      initial_texture_minimum=6,initial_plane='SVD of valid nonheldout same-leaf texture XYZ, including matches not chosen in camera optimization; view-facing plane if second singular value < 0.001 first',
      initialization_view='first actual selected pair view; PCA of currently retained frontier samples',
      initialization_profile=[.68,1.,.68],frontier_samples=64,
      contact_exclusion='exclude contour segments touching another identified leaf candidate mask within 3 work-equivalent pixels',
      frontier_semantics='unreviewed predicted visible segmentation frontier, NOT verified anatomical W; candidate diagnostic only',
      material_observations=0,anatomical_Q_observations=0,identity_clicks_triangulated=False,
      group_primary='lowest natural leaf ID with two candidate masks and >=6 valid training texture points, selected before surface fitting; no outcome-based replacement',
      other_leaves='all eligible leaves get full fit; at-least-one-output separately descriptive, not primary acceptance',
      controlled_fractions=[.2,.4,.6],controlled='central contiguous segment indices of first-view retained longest arc; endpoints not evaluated; rebuild initialization from retained only',
      controlled_reference='heldout candidate frontier, not independent true margin or natural hidden 3D truth',
      controls='two-view vs first-view W-only residual using same retained-two-view depth initialization and fixed pair cameras; not a standalone single-view reconstruction',
      no_parameter_selection_by_all_cohort_results=True,physical_scale=False,natural_hidden_reference=False,
      note_conflict_groups=CONFLICT,
      hashes={str(p):sha(p) for p in [Path(__file__),SOLVER,GUARD,GEO,P/'src/pots_gs/geometry.py',P/'src/pots_gs/subdivision.py',V/'RUN_SCOPE_v1.json']})

def contours(leaf,seg,view,meta):
    item=leaf['views'][view]; mask=mask_read(item['mask_path'])
    if mask is None:raise ValueError('missing native candidate mask')
    h,w=mask.shape; other=np.zeros((h,w),np.uint8)
    for otherleaf in seg['leaves']:
        if otherleaf['leaf_id']==leaf['leaf_id']:continue
        q=otherleaf['views'].get(view,{})
        if q.get('mask_path'):
            m=mask_read(q['mask_path'])
            if m is not None:other|=m
    radius=max(1,round(3/float(np.asarray(meta['native_to_work'])[0,0])))
    other=cv2.dilate(other,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))
    cs,_=cv2.findContours((mask>0).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not cs:raise ValueError('empty candidate mask')
    c=max(cs,key=cv2.contourArea).reshape(-1,2).astype(float)
    c=np.vstack([c,c[0]]); lens=np.r_[0,np.cumsum(np.linalg.norm(np.diff(c,axis=0),axis=1))]
    if lens[-1]<8:raise ValueError('short frontier')
    ts=np.arange(64)/64*lens[-1]; a=np.c_[np.interp(ts,lens,c[:,0]),np.interp(ts,lens,c[:,1])]
    edges=np.stack([a,np.roll(a,-1,axis=0)],axis=1)
    kept=[]; removed=[]
    for i,(u,v) in enumerate(edges):
        q=np.array([u+t*(v-u) for t in np.linspace(0,1,9)])
        q=np.rint(q).astype(int); x=np.clip(q[:,0],0,w-1); y=np.clip(q[:,1],0,h-1)
        if np.any(other[y,x]>0):removed.append(i)
        else:kept.append(i)
    if len(kept)<4:raise ValueError('insufficient noncontact candidate frontier')
    work=transform(edges.reshape(-1,2),meta['native_to_work']).reshape(-1,2,2)
    return dict(segments=work[kept],segment_ids=kept,removed_contact_segment_ids=removed,
      native_segments=edges[kept],mask_path=item['mask_path'],mask_sha256=sha(item['mask_path']),
      boundary_role='candidate_visible_frontier_not_verified_true_W',closed_curve_assumed_true_margin=False)

def prepared(paths):
    return [dict(view_id=v,material=[],Q=np.empty((0,2)),W=np.array([a+t*(b-a) for a,b in p['segments'] for t in (0.,.25,.5,.75,1.)]).reshape(-1,2)) for v,p in paths.items()]

def initial(cameras,pp,texture):
    if len(texture)<6:raise ValueError('fewer than 6 valid training same-leaf texture points')
    origin=np.median(texture,axis=0); centered=texture-origin
    _,s,vh=np.linalg.svd(centered,full_matrices=False)
    first=pp[0]; c=cameras[first['view_id']]
    fallback=bool(s[1]<s[0]*.001)
    normal=c.R[2].copy() if fallback else vh[-1]
    points=np.unique(first['W'],axis=0); center=points.mean(0)
    _,_,vh2=np.linalg.svd(points-center,full_matrices=False);axis=vh2[0]
    if axis[1]<0 or (axis[1]==0 and axis[0]<0):axis=-axis
    side=np.array([-axis[1],axis[0]]); along=(points-center)@axis; across=(points-center)@side
    lo,hi=float(along.min()),float(along.max()); left,right=float(across.min()),float(across.max())
    if min(hi-lo,right-left)<1e-6:raise ValueError('degenerate retained contour extent')
    uv=np.empty((11,2));uv[0]=center+lo*axis;uv[10]=center+hi*axis
    for j,(u,profile) in enumerate(zip((.25,.5,.75),(.68,1.,.68))):
        mid=center+(lo+u*(hi-lo))*axis;uv[1+3*j]=mid+profile*left*side;uv[2+3*j]=mid;uv[3+3*j]=mid+profile*right*side
    xyz=[]
    for q in uv:
        ray=c.R.T@np.linalg.solve(c.K,np.r_[q,1.]); denom=float(ray@normal)
        if abs(denom)<1e-10:raise ValueError('plane ray parallel')
        d=float((origin-c.center)@normal/denom)
        if d<=0:raise ValueError('plane controls behind reference camera')
        xyz.append(c.center+d*ray)
    xyz=np.asarray(xyz)
    if any(np.any(core.project_xyz(c,xyz)[1]<=0) for c in cameras.values()):raise ValueError('initialized controls have negative depth')
    return xyz,dict(xyz=xyz,plane_origin=origin,plane_normal=normal,texture_train_count=len(texture),
      singular_values=s,view_facing_plane_fallback=fallback,reference_view=first['view_id'],
      canonical_slots_not_observed_B_T=True,Q_observed=False,heldout_frontier_used=False,depth_is_estimated_prior=True,
      retained_W_sha256={p['view_id']:hashlib.sha256(p['W'].astype('<f8').tobytes()).hexdigest() for p in pp})

def check(xyz,xyz0,pp,cameras,diag):
    ga=geometry.audit(xyz,np.asarray(LEAF_FACES),xyz0);views=[]
    for p in pp:
        uv,z=core.project_xyz(cameras[p['view_id']],xyz); rr,_=core.polyline_residual(p['W'],uv[core.BOUNDARY])
        views.append(dict(view_id=p['view_id'],W_RMS_work_px=float(np.sqrt(np.mean(np.sum(rr*rr,axis=1)))),positive_depth=bool(np.all(z>0))))
    hard=bool(np.isfinite(xyz).all() and all(v['positive_depth'] for v in views) and not ga['triangle_intersection_pairs'] and not ga['degenerate_faces_at_normalized_area_1e_minus12'] and not ga['nonmanifold_edges'])
    return dict(geometry=ga,views=views,hard_geometry_passed=hard,
      strict_internal_pass=bool(hard and all(v['W_RMS_work_px']<=8 for v in views)),
      exploration_internal_pass=bool(hard and all(v['W_RMS_work_px']<=16 for v in views)),optimizer_success=diag['optimizer_success'])

def split_paths(paths,fraction):
    result=copy.deepcopy(paths);view=next(iter(result));p=result[view];ids=p['segment_ids']
    runs=[];current=[]
    for i,edge_id in enumerate(ids):
        if current and edge_id!=ids[current[-1]]+1:runs.append(current);current=[]
        current.append(i)
    if current:runs.append(current)
    run=max(runs,key=lambda x:(len(x),-x[0]));n=len(run);count=int(np.ceil(fraction*n));start=(n-count)//2
    if start<1 or start+count>=n:raise ValueError('not enough contiguous frontier for controlled interval')
    withheld=run[start:start+count]; held=np.array([a+t*(b-a) for a,b in p['segments'][withheld] for t in (.25,.5,.75)])
    keep=[i for i in range(len(ids)) if i not in withheld]
    p['segments']=p['segments'][keep];p['segment_ids']=[ids[i] for i in keep]
    p['native_segments']=p['native_segments'][keep]
    return result,dict(view_id=view,points_work=held,fraction_of_longest_arc_segments=fraction,
      withheld_segment_ids=[ids[i] for i in withheld],reference='withheld predicted frontier; not independently measured leaf edge')

def dashed(draw,a,b,fill,width=3):
    a=np.array(a);b=np.array(b);dist=np.linalg.norm(b-a)
    for t in np.arange(0,dist,12):
        if dist:draw.line([tuple(a+(b-a)*t/dist),tuple(a+(b-a)*min(t+7,dist)/dist)],fill=fill,width=width)

def overlay(dest,cameras,metas,leaf,xyz,paths):
    saved={};edges=sorted({tuple(sorted((a,b))) for f in LEAF_FACES for a,b in zip(f,(*f[1:],f[0]))})
    for view,c in cameras.items():
        uv,z=core.project_xyz(c,xyz);native=transform(uv,metas[view]['work_to_native'])
        image=ImageOps.exif_transpose(Image.open(metas[view]['image_path'])).convert('RGB')
        mask=mask_read(leaf['views'][view]['mask_path']);h,w=mask.shape
        boxes=np.vstack([native,np.asarray(leaf['views'][view]['boundary_candidate_xy'],float)])
        lo=np.maximum(np.floor(boxes.min(0)-80),0).astype(int);hi=np.minimum(np.ceil(boxes.max(0)+80),[w,h]).astype(int)
        if np.any(hi<=lo):continue
        draw=ImageDraw.Draw(image)
        for a,b in edges:
            mid=np.rint((native[a]+native[b])*.5).astype(int);inside=bool(0<=mid[0]<w and 0<=mid[1]<h and mask[mid[1],mid[0]])
            if inside:draw.line([tuple(native[a]),tuple(native[b])],fill=(0,150,185),width=max(2,round(w/900)))
            else:dashed(draw,native[a],native[b],(230,120,25),max(2,round(w/900)))
        crop=image.crop((*lo,*hi));crop.thumbnail((700,700));crop.save(dest/f'view_{view}_overlay.jpg',quality=91)
        saved[view]=dict(path=str(dest/f'view_{view}_overlay.jpg'),crop_native_xyxy=[*lo.tolist(),*hi.tolist()],solid='projection inside candidate visible mask',dashed='projection outside candidate mask; inferred/unverified, may also be mask error')
    return saved

def run_fit(dest,cid,leaf,camera,metas,paths,texture,condition,branch,phash,held=None):
    receipt=dict(group_id=cid,leaf_id=leaf['leaf_id'],condition=condition,branch=branch,
      protocol_sha256=phash,strict_complete_pass=False,exploration_complete_pass=False,
      natural_hidden_accuracy=None,identity_note_conflict=cid in CONFLICT)
    if (dest/'execution_receipt.json').exists():return read(dest/'execution_receipt.json')
    dest.mkdir(parents=True,exist_ok=False);tic=time.perf_counter()
    try:
        cams={v:Camera(v,c['K'],c['R'],c['t'],unit='relative') for v,c in camera['cameras_work'].items()}
        pp=prepared(paths);xyz0,init=initial(cams,pp,texture)
        fitpp=pp if branch=='two_view' else pp[:1];fitcam={p['view_id']:cams[p['view_id']] for p in fitpp}
        segments={p['view_id']:paths[p['view_id']]['segments'] for p in fitpp}
        solver=guard.build_solver(SOLVER,segments)
        solver.prepare_observations=lambda *a,**k:(fitpp,[],['Identity reviewed; segmentation frontiers are unreviewed candidates, not true anatomical W'])
        fitted=solver.fit_scaffold(xyz0,fitcam,[],organ_id=f'{cid}/{leaf["leaf_id"]}',data_type='real',allow_unreviewed_diagnostic=True,allow_single_view_diagnostic=len(fitcam)==1,**RULES)
        fitted.diagnostics['W_assignment']='candidate visible frontier to projected boundary; contact-adjacent edges excluded; not reviewed true margin'
        fitted.diagnostics['status']='identity_conditioned_candidate_surface_diagnostic'
        fitted.mesh.metadata.update(canonical_slots_not_observed_B_T=True,anatomical_Q_observed=False,
          candidate_frontier_fitting=True,physical_scale_available=False,identity_locators_triangulated=False,full_plant_reconstruction=False)
        fitted.mesh.coordinate_system='estimated two-view relative coordinates; no physical scale'
        for i,vert in enumerate(fitted.mesh.vertices):
            fitted.mesh.vertices[i]=replace(vert,provenance=Provenance('estimated',tuple(cams),(),
              'Same-leaf image texture initializes depth; predicted candidate frontier fit with priors; canonical B/T/Q are model slots, not marked anatomical observations'))
        areas=[]
        for mesh in subdivision_levels(fitted.mesh,2):
            mesh.write_json(dest/f'model_L{mesh.level}.json');mesh.write_obj(dest/f'model_L{mesh.level}.obj');areas.append(mesh_area(mesh))
        assert np.allclose(areas,areas[0],rtol=1e-10,atol=1e-12)
        checks=check(fitted.xyz,xyz0,pp,cams,fitted.diagnostics)
        dump(dest/'initialization.json',init);dump(dest/'solver_diagnostics.json',fitted.diagnostics)
        dump(dest/'internal_consistency.json',checks)
        dump(dest/'actual_fit_input.json',dict(prepared=fitpp,support_segments=segments,cameras={v:camera['cameras_work'][v] for v in fitcam},
             initialization=init,evaluation_coordinates_used=False,frontier_semantics='candidate, not true W'))
        projections={v:dict(vertices_work=core.project_xyz(c,fitted.xyz)[0],vertices_native=transform(core.project_xyz(c,fitted.xyz)[0],metas[v]['work_to_native'])) for v,c in cams.items()}
        dump(dest/'projections.json',projections)
        ev=None
        if held is not None:
            uv=np.asarray(projections[held['view_id']]['vertices_work']); rr,_=core.polyline_residual(held['points_work'],uv[core.BOUNDARY]);ds=np.linalg.norm(rr,axis=1)
            ev=dict(**held,RMS_work_px=float(np.sqrt(np.mean(ds*ds))),distances_work_px=ds,evaluation_only=True)
            dump(dest/'heldout_frontier_evaluation.json',ev)
        ok=checks['optimizer_success'] and not receipt['identity_note_conflict']
        receipt.update(status='surface_computed',model_L0_json=str(dest/'model_L0.json'),optimizer_success=checks['optimizer_success'],
          nfev=fitted.diagnostics['nfev'],strict_internal_pass=checks['strict_internal_pass'],exploration_internal_pass=checks['exploration_internal_pass'],
          strict_complete_pass=bool(ok and camera['strict_camera_gate_passed'] and checks['strict_internal_pass']),
          exploration_complete_pass=bool(ok and camera['exploration_camera_gate_passed'] and checks['exploration_internal_pass']),
          heldout_RMS_work_px=ev['RMS_work_px'] if ev else None)
        if condition=='full':receipt['overlays']=overlay(dest,cams,metas,leaf,fitted.xyz,paths)
    except Exception as e:receipt.update(status='failed_preserved',reason=str(e),traceback=traceback.format_exc())
    receipt['elapsed_seconds']=time.perf_counter()-tic
    receipt['output_sha256']={p.name:sha(p) for p in dest.iterdir() if p.is_file()}
    dump(dest/'execution_receipt.json',receipt);return receipt

def run_group(cid,phash,controlled=True):
    dest=R/'results'/cid
    if (dest/'group_result.json').exists():return read(dest/'group_result.json')
    cp=V/'camera/results'/cid/'camera_result.json';sp=V/'segmentation/groups'/cid/'segmentation.json'
    if not cp.exists() or not sp.exists():return None
    camera=read(cp);seg=read(sp);rec=dict(group_id=cid,camera_result_path=str(cp),camera_sha256=sha(cp),segmentation_sha256=sha(sp),
       strict_camera_gate_passed=camera.get('strict_camera_gate_passed',False),exploration_camera_gate_passed=camera.get('exploration_camera_gate_passed',False),
       strict_complete_pass=False,exploration_complete_pass=False,primary_leaf_id=None,leaves=[],protocol_sha256=phash,
       note_conflict=cid in CONFLICT,natural_hidden_accuracy=None)
    if not camera.get('fixed_middle_camera') or not (cp.parent/'leaf_geometry.npz').exists():
        rec.update(status='camera_unavailable',reason=camera.get('failure_reasons',camera.get('status')));dump(dest/'group_result.json',rec);return rec
    metas={v['view_id']:v for v in camera['views']}; pair=camera['actual_view_pair'];data=np.load(cp.parent/'leaf_geometry.npz')
    cameras={v:dict(K=q['K_work'],R=q['R'],t=q['t']) for v,q in camera['fixed_middle_camera']['poses_by_view_id'].items()}
    camera['cameras_work']=cameras
    labels={x['leaf_id']:x['label'] for x in camera['leaves']}
    def order(leaf):
        digits=''.join(c for c in leaf['leaf_id'] if c.isdigit());return (int(digits) if digits else 10**9,leaf['leaf_id'])
    candidates=[]
    for leaf in sorted(seg['leaves'],key=order):
        lid=leaf['leaf_id'];label=labels.get(lid);eligible=label is not None and all(leaf['views'].get(v,{}).get('status')=='candidate_generated' for v in pair)
        indices=np.flatnonzero((data['leaf_label']==label)&data['valid_for_surface']) if label is not None else np.array([],int)
        record=dict(leaf_id=lid,valid_nonheldout_texture_points=len(indices),valid_nonheldout_texture_indices=indices.tolist(),candidate_both_views=bool(eligible))
        if eligible and len(indices)>=6:candidates.append((leaf,indices,record))
        else:record.update(status='input_ineligible',reason='requires both candidate masks and >=6 valid training texture correspondences');rec['leaves'].append(record)
    if candidates:rec['primary_leaf_id']=candidates[0][0]['leaf_id']
    # This selection is frozen before any surface outcome.
    selection=dest/'selection_before_surface.json'
    if not selection.exists():dump(selection,dict(primary_leaf_id=rec['primary_leaf_id'],candidate_order=[x[0]['leaf_id'] for x in candidates],camera_sha256=sha(cp),selection_uses_surface_outcome=False))
    for leaf,indices,record in candidates:
        lid=leaf['leaf_id'];base=dest/lid
        try:
            paths={v:contours(leaf,seg,v,metas[v]) for v in pair};texture=data['xyz'][indices]
            record['full']=run_fit(base/'full'/'two_view',cid,leaf,camera,metas,paths,texture,'full','two_view',phash)
            if lid==rec['primary_leaf_id']:
                rec.update(strict_complete_pass=record['full']['strict_complete_pass'],exploration_complete_pass=record['full']['exploration_complete_pass'])
                if controlled:
                    record['controlled']=[]
                    for fraction in (.2,.4,.6):
                        cond=f'withhold_{round(fraction*100)}'
                        try:
                            kept,held=split_paths(paths,fraction)
                            for branch in ('two_view','first_view_W_only'):
                                record['controlled'].append(run_fit(base/cond/branch,cid,leaf,camera,metas,kept,texture,cond,branch,phash,held))
                        except Exception as e:record['controlled'].append(dict(condition=cond,status='input_unavailable',reason=str(e)))
        except Exception as e:record.update(status='frontier_or_initialization_unavailable',reason=str(e))
        rec['leaves'].append(record)
    rec.update(status='processed',computed_leaf_models=sum(x.get('full',{}).get('status')=='surface_computed' for x in rec['leaves']),
        any_leaf_strict_descriptive=any(x.get('full',{}).get('strict_complete_pass',False) for x in rec['leaves']),
        any_leaf_exploration_descriptive=any(x.get('full',{}).get('exploration_complete_pass',False) for x in rec['leaves']))
    dump(dest/'group_result.json',rec);print(json.dumps({k:rec[k] for k in ('group_id','status','computed_leaf_models','strict_complete_pass','exploration_complete_pass')},ensure_ascii=False),flush=True);return rec

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['freeze','run']);ap.add_argument('--groups',nargs='*');ap.add_argument('--no-controlled',action='store_true');args=ap.parse_args()
    pp=R/'protocol_v1.json'
    if args.command=='freeze':dump(pp,protocol());print(str(pp));return
    proto=read(pp)
    for p,h in proto['hashes'].items():assert sha(p)==h,('frozen code changed',p)
    groups=args.groups or [p.parent.name for p in sorted((V/'camera/results').glob('*/camera_result.json'))]
    for cid in groups:
        try:run_group(cid,sha(pp),not args.no_controlled)
        except Exception as e:print(json.dumps(dict(group_id=cid,unhandled_error=str(e),traceback=traceback.format_exc())),flush=True)
if __name__=='__main__':main()
