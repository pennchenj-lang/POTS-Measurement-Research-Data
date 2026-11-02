"""V31 numerical adapter with local-mask inputs of frozen V24/V16 solver. Active inputs and rules are bind_frozen_frontend_v1 and camera_numeric_v1 protocols."""
from pathlib import Path
import os,sys,json,hashlib,ast,types,itertools,time,traceback,collections,argparse,importlib.util,datetime
HERE=Path(__file__).resolve(); ROOT=HERE.parent; V30=ROOT.parent; P=V30.parents[1]; V24=V30
for k in ('TEMP','TMP','TMPDIR','PIP_CACHE_DIR','TORCH_HOME','HF_HOME','MPLCONFIGDIR','XDG_CACHE_HOME'):
    d=ROOT/'cache'/k.lower();d.mkdir(parents=True,exist_ok=True);os.environ[k]=str(d)
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True;os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.path.append(str(P/'.venv/Lib/site-packages'))
import numpy as np,cv2
from PIL import Image,ImageOps
try:
    from scipy.optimize import least_squares
    from scipy.sparse import csr_matrix
except ImportError:
    # The pinned CUDA frontend interpreter needs no numerical optimizer.
    least_squares=csr_matrix=None
cv2.setNumThreads(1)
MANIFEST=P/'work/same_leaf_combined_20261030_v23/bundle_v1/manifest.json'
IDENTITIES=MANIFEST.with_name('identity_constraints.json')
SCOPE=V30/'RUN_SCOPE_v1.json'
SOURCE=P/'work/multispecies_replacement_198_20261025_v11/results/pair_frontend_v1/frozen_source/pair_camera_v1.py'
V16=P/'work/reconstruction_repair_20261028_v16/public_camera/code'
HELPER=V16/'training_camera_initializer_v2.py'
PROFILE=P/'work/tolerance_ceiling_20261028_v15/exploration_profile_frozen_v1.json'
DEV=['G01', 'G40', 'P079', 'P094', 'P132', 'P183', 'G20', 'P014', 'P017', 'P173']
LEGACY_UNUSED_RULES=dict(pair='first two manifest views, immutable original pair; no post-fit pair or leaf selection',
    masks='conflict-free native label map; both texture endpoints require equal positive leaf label; no locator triangulation',
    frontend='verified legacy raw texture matches union new CPU RootSIFT; never legacy F mask or camera',
    rootsift=dict(max_long_side=1536,nfeatures=8000,contrastThreshold=.015,ratio=.8,mutual=True,detector='full image inside conflict-free leaf labels'),
    dedup='paired work coordinate quantization .25px; cache first, RootSIFT second',
    partition='spatial4D round robin cap160; deterministic permutation; holdout max(8,n//5); train exclude <=2 workpx from ANY heldout endpoint then spatial cap128',
    source_camera_K_Rt_used=False,physical_scale=False,strict_thresholds_changed=False,exploration_thresholds_changed=False,
    initialization='frozen V16 six training-only seeds; three final dense exact960 starts; common>=20; fixed middle index1',
    leaf_geometry='all raw nonheldout matches distant>2px from heldout endpoints, triangulated by fixed middle BA; positive both/angle>=.25/reprojection each<=4px; geometry valid is not camera accepted',
    heldout='evaluated only, never input to fit, parameter choice, or surface; saved separately',
    data_type='real photographs; annotations confirm identity but candidate masks are not true margins')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def write(p,x):
    p=Path(p);assert p.resolve().is_relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def tf(x,A):return (np.c_[x,np.ones(len(x))]@np.asarray(A).T)[:,:2]
class Dense(ast.NodeTransformer):
    def visit_Constant(self,n):return ast.copy_location(ast.Constant(960),n) if n.value==240 else n
    def visit_Call(self,n):
        n=self.generic_visit(n)
        if isinstance(n.func,ast.Name) and n.func.id=='least_squares':
            n.keywords=[k for k in n.keywords if k.arg!='tr_options']
            for k in n.keywords:
                if k.arg=='tr_solver':k.value=ast.Constant('exact')
        return n
def core():
    assert least_squares is not None, 'Numerical camera stage requires the batch50 scipy environment'
    body=[]
    for n in ast.parse(SOURCE.read_text(encoding='utf-8-sig')).body:
        if isinstance(n,ast.FunctionDef):
            if n.name=='solve':n=Dense().visit(n);n.body=ast.parse('init = prepared_initializations').body+n.body[2:]
            body.append(n)
        elif isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'GROUPS','PAIRS','SEED','PROTOCOL'} for t in n.targets):body.append(n)
    m=types.ModuleType('frozen_pair_core');m.__dict__.update(globals(),RUN=ROOT,PROJECT=P)
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(SOURCE),'exec'),m.__dict__)
    m.jacobian=m.jacobian_dense;m.PROTOCOL['BA']['max_nfev']=960
    sp=importlib.util.spec_from_file_location('frozen_v16_initializer',HELPER);init=importlib.util.module_from_spec(sp);sp.loader.exec_module(init)
    n=next(n for n in ast.parse((V16/'run_public_camera_repair_v2.py').read_text(encoding='utf-8-sig')).body if isinstance(n,ast.FunctionDef) and n.name=='dual_gates')
    env=dict(globals());exec(compile(ast.Module(body=[n],type_ignores=[]),str(V16/'run_public_camera_repair_v2.py'),'exec'),env)
    return m,init,env['dual_gates']
def metadata(v):
    w,h=v['width'],v['height'];s=1000/max(w,h)
    A=np.array([[s,0,499.5-s*(w-1)/2],[0,s,499.5-s*(h-1)/2],[0,0,1.]])
    return dict(view_id=v['id'],source_view_id=v['source_view_id'],image_path=v.get('frozen_experiment_image_path',v['path']),source_sha256=v.get('frozen_experiment_image_sha256',v['display_sha256']),original_source_sha256=v['sha256'],original_path=v.get('original_path'),display_sha256=v['display_sha256'],native_size=[w,h],native_to_work=A,work_to_native=np.linalg.inv(A),work_size=[1000,1000])
def labels_at(x,lab):
    ix=np.rint(x).astype(int);ok=(ix[:,0]>=0)&(ix[:,1]>=0)&(ix[:,0]<lab.shape[1])&(ix[:,1]<lab.shape[0]);r=np.zeros(len(x),int);r[ok]=lab[ix[ok,1],ix[ok,0]];return r
def cached(g,views):
    """Never trust internal view IDs: bind native endpoints by exact source image SHA."""
    candidates=[];receipts=[]
    if g['source']=='self':
        src=Path(g['source_binding']['matching_receipt_path']);m=read(src);vm={q['source_sha256']:q for q in m['views']}
        if all(v['sha256'] in vm for v in views):
            ids=[vm[v['sha256']]['view_id'] for v in views];ordered=sorted(ids);zpath=src.parent/f'pair_{ordered[0]}_{ordered[1]}.npz'
            if zpath.exists():candidates.append((src,zpath,m,ids!=ordered))
    else:
        od=P/'work/multispecies_replacement_198_20261025_v11/results/pair_frontend_v1/results/matching_v3'/g['id']
        rd=P/'work/leaf_target_recovery_20261026_v12/segmentation/native_roi_matching_v1/results'/g['id']
        for d,n,metadata_file in ((od,'matches.npz','matching.json'),(rd,'target_candidate_matches.npz','interface_v1.json')):
            if (d/metadata_file).exists() and (d/n).exists():
                m=read(d/metadata_file);hs=[q['source_sha256'] for q in m['views']];wanted=[v['sha256'] for v in views]
                if hs==wanted or hs==wanted[::-1]:candidates.append((d/metadata_file,d/n,m,hs!=wanted))
    obs=[]
    for mp,zp,m,rev in candidates:
        with np.load(zp,allow_pickle=False) as z:
            a=z['native_a'];b=z['native_b'];arr=np.stack([b,a] if rev else [a,b],1)
            assert np.isfinite(arr).all();obs.append(arr)
        receipts.append(dict(metadata_path=str(mp),metadata_sha256=sha(mp),matches_path=str(zp),matches_sha256=sha(zp),endpoints_reversed=rev,count=len(arr),binding='exact image SHA then stored native coordinates',raw_F_mask_used=False))
    return np.concatenate(obs) if obs else np.empty((0,2,2)),receipts
def rootsift(v,lab,out):
    with Image.open(v['path']) as im:a=np.asarray(ImageOps.exif_transpose(im).convert('L'))
    assert a.shape==(v['height'],v['width']);h,w=a.shape;s=min(1.,RULES['rootsift']['max_long_side']/max(h,w));rw,rh=round(w*s),round(h*s)
    a=cv2.resize(a,(rw,rh),interpolation=cv2.INTER_AREA) if s<1 else a
    mask=cv2.resize((lab>0).astype(np.uint8)*255,(rw,rh),interpolation=cv2.INTER_NEAREST)
    detector=cv2.SIFT_create(nfeatures=8000,contrastThreshold=.015)
    kp,d=detector.detectAndCompute(a,mask);xy=np.array([q.pt for q in kp],float).reshape(-1,2)
    native=(xy+.5)/np.array([rw/w,rh/h])-.5
    d=np.empty((0,128),np.float32) if d is None else np.sqrt(d/np.maximum(d.sum(1,keepdims=True),1e-12)).astype(np.float32)
    label=labels_at(native,lab);keep=label>0;native,d,label=native[keep],d[keep],label[keep]
    np.savez_compressed(out,native_xy=native,descriptors=d,leaf_label=label,extract_size=[rw,rh])
    return native,d,label
def new_matches(a,b):
    out=[];bf=cv2.BFMatcher(cv2.NORM_L2)
    for label in sorted(set(a[2])&set(b[2])):
        ia=np.flatnonzero(a[2]==label);ib=np.flatnonzero(b[2]==label)
        if min(len(ia),len(ib))<2:continue
        ab=bf.knnMatch(a[1][ia],b[1][ib],k=2);ba=bf.knnMatch(b[1][ib],a[1][ia],k=2)
        reverse={q[0].queryIdx:q[0].trainIdx for q in ba if len(q)==2 and q[0].distance<.8*q[1].distance}
        for q in ab:
            if len(q)==2 and q[0].distance<.8*q[1].distance and reverse.get(q[0].trainIdx)==q[0].queryIdx:
                out.append([a[0][ia[q[0].queryIdx]],b[0][ib[q[0].trainIdx]]])
    return np.array(out,float).reshape(-1,2,2)
def partition(c,obs,seed):
    sel=c.spatial_select(np.arange(len(obs)),obs,160);sel=sel[np.random.default_rng(seed).permutation(len(sel))]
    held=sel[:max(8,len(sel)//5)];t=sel[len(held):];protected=np.zeros(len(obs),bool)
    if len(held):
        for k in (0,1):protected|=np.min(np.linalg.norm(obs[:,k,None]-obs[held,k][None],axis=-1),axis=1)<=2
    train=c.spatial_select(t[~protected[t]],obs,128)
    return sel,train,held,protected
def run(g,c,init,gates):
    gid=g['id'];out=ROOT/'results'/gid;rp=out/'camera_result.json'
    if rp.exists():return read(rp)
    out.mkdir(parents=True,exist_ok=True);tic=time.perf_counter();views=g['views'][:2];meta=[metadata(v) for v in views]
    result=dict(schema='pots.train_track_consistency.v33',selected_source='v33_training_track_consistency',group_id=gid,data_type='real',actual_view_pair=[v['id'] for v in views],views=meta,status='started',strict_camera_gate_passed=False,exploration_camera_gate_passed=False,no_identity_anchor_triangulation=True,source_camera_K_Rt_used=False,fixed_middle_camera=None,leaves=[],protocol_sha256=sha(ROOT/'development_protocol.json'))
    try:
        segpath=V24/'segmentation/groups'/gid/'segmentation.json';seg=read(segpath);sv={v['view_id']:v for v in seg['views']}
        labs=[np.array(Image.open(sv[v['id']]['label_mask_path'])) for v in views]
        for v in views:assert sha(v.get('frozen_experiment_image_path',v['path']))==v.get('frozen_experiment_image_sha256',v['display_sha256'])
        for lab,v in zip(labs,views):assert lab.shape==(v['height'],v['width'])
        raw,receipts=cached(g,views);feat=[rootsift(v,lab,out/f"rootsift_{v['id']}.npz") for v,lab in zip(views,labs)];fresh=new_matches(*feat)
        allnative=np.concatenate([raw,fresh]);source=np.r_[np.zeros(len(raw),np.uint8),np.ones(len(fresh),np.uint8)]
        lab0=labels_at(allnative[:,0],labs[0]);lab1=labels_at(allnative[:,1],labs[1]);keep=~((lab0>0)&(lab1>0)&(lab0!=lab1));native=allnative[keep];labels=np.where((lab0>0)&(lab0==lab1),lab0,0)[keep];source=source[keep]
        obs=np.stack([tf(native[:,k],meta[k]['native_to_work']) for k in (0,1)],1)
        _,u=np.unique(np.rint(obs.reshape(-1,4)/.25).astype(np.int64),axis=0,return_index=True);u=np.sort(u);native,obs,labels,source=native[u],obs[u],labels[u],source[u]
        seed=20261030+int(gid[1:]);selected,trainids,heldids,protected=partition(c,obs,seed);train,held=obs[trainids],obs[heldids]
        np.savez_compressed(out/'camera_input.npz',all_xy_native=native,all_xy_work=obs,leaf_label=labels,source_code=source,selected=selected,train_indices=trainids,heldout_indices=heldids,heldout_protected=protected,training_xy=train,heldout_xy=held)
        write(out/'input_receipt.json',dict(group_id=gid,segmentation_path=str(segpath),segmentation_sha256=sha(segpath),legacy_sources=receipts,root_sift_raw=len(fresh),legacy_raw=len(raw),same_leaf_unique=len(obs),train_count=len(train),heldout_count=len(held),identity_anchors_are_never_correspondences=True,leaf_mapping={i+1:l['leaf_id'] for i,l in enumerate(seg['leaves'])}))
        result.update(input_counts=dict(legacy_raw=len(raw),rootsift_raw=len(fresh),same_leaf_unique=len(obs),selected=len(selected),training=len(train),heldout=len(held)),input_path=str(out/'camera_input.npz'))
        if len(selected)<28 or len(train)<20 or len(held)<8:result['status']='rejected_insufficient_training_or_heldout_texture'
        else:
            initial,audit=init.build_initializations(train,c,seed);write(out/'training_initializer_audit.json',audit);c.prepared_initializations=initial
            result.update(c.solve(train,held,meta,out,seed,gid in DEV))
        bp=out/'bundle_adjustment_all_starts.json';runs=read(bp) if bp.exists() else []
        gate=gates(result,runs);result.update(gate);result['strict_camera_gate_passed']=result.get('conditional_pair_camera_gate_passed',False);result['exploration_camera_gate_passed']=gate['exploration_gate_passed']
        xyz=np.full((len(obs),3),np.nan);valid=np.zeros(len(obs),bool);err=np.full((len(obs),2),np.nan);angles=np.full(len(obs),np.nan);pos=np.zeros(len(obs),bool)
        if len(runs)==3 and runs[1].get('status')=='returned':
            q=runs[1];K=np.array(q['K']);R=np.array(q['R']);t=np.array(q['t']);xyz=c.triangulate(obs,K,R,t);err,pos,angles,depth=c.geometries(xyz,obs,K,R,t);track_flags=np.load(ROOT/'inputs'/gid/'selected_track_flags.npy',allow_pickle=False)
            assert len(track_flags)==len(labels)
            valid=pos&(angles>=.25)&(err.max(1)<=4)&~protected&(labels>0)&(track_flags>=0)
            cams={m['view_id']:dict(K_work=K,K_native=np.asarray(m['work_to_native'])@K,R=np.eye(3) if i==0 else R,t=np.zeros(3) if i==0 else t) for i,m in enumerate(meta)}
            result['fixed_middle_camera']=dict(K_work=K,R=R,t=t,poses_by_view_id=cams,unit='estimated_relative_pair_baseline_norm1',accepted_strict=result['strict_camera_gate_passed'],accepted_exploration=result['exploration_camera_gate_passed'])
        np.savez_compressed(out/'leaf_geometry.npz',xyz=xyz,xy_native=native,xy_work=obs,leaf_label=labels,source_code=source,valid_for_surface=valid,positive=pos,ray_angle_deg=angles,reprojection_error_work_px=err,heldout_protected=protected,heldout_indices=heldids,train_indices=trainids)
        for i,leaf in enumerate(seg['leaves'],1):
            ix=np.flatnonzero(labels==i);result['leaves'].append(dict(leaf_id=leaf['leaf_id'],label=i,texture_count=len(ix),valid_texture_count=int(valid[ix].sum()),indices=ix,valid_indices=ix[valid[ix]],geometry_path=str(out/'leaf_geometry.npz'),status='geometry_support_available' if valid[ix].sum()>=3 else 'insufficient_nonheldout_valid_texture'))
        result['identity_conflict_blocked']=gid in {'P074','P174','P180','P181'}
        if result['identity_conflict_blocked']:
            result['strict_camera_gate_passed']=False;result['exploration_camera_gate_passed']=False;result['conditional_pair_camera_gate_passed']=False;result['exploration_gate_passed']=False
            if result['fixed_middle_camera']:result['fixed_middle_camera']['accepted_strict']=False;result['fixed_middle_camera']['accepted_exploration']=False
            result['status']='identity_conflict_requires_resolution'
        result['failure_reasons']=[] if result['strict_camera_gate_passed'] else [result['status'],*gate.get('hard_rejection_reasons',[]),*[k for k,v in gate.get('exploration_soft_checks',{}).items() if not v]]
    except Exception as ex:result.update(status='runtime_failure',error=str(ex),traceback=traceback.format_exc(),failure_reasons=[str(ex)])
    result['elapsed_seconds']=time.perf_counter()-tic;write(rp,result)
    print(json.dumps({k:result.get(k) for k in ('group_id','status','input_counts','strict_camera_gate_passed','exploration_camera_gate_passed','elapsed_seconds')},ensure_ascii=False),flush=True);return result
def freeze_dev():
    pp=ROOT/'development_protocol.json'
    p=dict(schema='v24-camera-development-protocol',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),development_groups=DEV,rules=RULES,sources={str(f):sha(f) for f in (HERE,MANIFEST,IDENTITIES,SCOPE,SOURCE,HELPER,V16/'run_public_camera_repair_v2.py',PROFILE)})
    if pp.exists() and read(pp)['sources']!=p['sources']:
        assert not list((ROOT/'results').glob('*/camera_result.json')),'Development outcomes already exist; preserve a separate revised run'
        write(ROOT/('pre_execution_protocol_'+sha(pp)[:12]+'.json'),read(pp));write(pp,p)
    else:write(pp,p)
def verify():
    for p,h in read(ROOT/'development_protocol.json')['sources'].items():assert sha(p)==h,p
def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['development','freeze','application','summary']);args=ap.parse_args();groups=read(MANIFEST)['groups'];gm={g['id']:g for g in groups}
    if args.mode=='development':freeze_dev()
    verify()
    if args.mode=='freeze':
        results=[ROOT/'results'/gid/'camera_result.json' for gid in DEV];assert all(p.exists() for p in results);assert all(read(p)['status']!='runtime_failure' for p in results)
        write(ROOT/'application_frozen.json',dict(development_result_sha256={str(p):sha(p) for p in results},sources=read(ROOT/'development_protocol.json')['sources'],rules=RULES,method_changed_after_development=False,full_cohort_count=len(groups)));return
    if args.mode in ('development','application'):
        if args.mode=='application':
            f=read(ROOT/'application_frozen.json')
            for p,h in {**f['sources'],**f['development_result_sha256']}.items():assert sha(p)==h
        c,init,gates=core();ids=DEV if args.mode=='development' else list(gm)
        for gid in ids:
            seg=V24/'segmentation/groups'/gid/'segmentation.json'
            while not seg.exists():time.sleep(5)
            run(gm[gid],c,init,gates)
    rows=[read(f) for f in sorted((ROOT/'results').glob('*/camera_result.json'))]
    summary=dict(groups=len(rows),strict=sum(r['strict_camera_gate_passed'] for r in rows),exploration=sum(r['exploration_camera_gate_passed'] for r in rows),status_counts=dict(collections.Counter(r['status'] for r in rows)),rows=[{k:r.get(k) for k in ('group_id','status','strict_camera_gate_passed','exploration_camera_gate_passed','input_counts','failure_reasons','elapsed_seconds')} for r in rows])
    write(ROOT/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='rows'},ensure_ascii=False))
if __name__=='__main__':main()
