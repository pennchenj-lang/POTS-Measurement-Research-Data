"""Final prespecified input adaptation: shared-feature split, training-only single-pair ranking."""
import run_identity_camera_v1 as a
from track_evidence import classify_tracks, tiered_training
from concurrent.futures import ProcessPoolExecutor
import argparse,math,collections,datetime
HERE=a.Path(__file__).resolve();ROOT=HERE.parent
RULES=dict(version='v33_training_only_track_conflict_and_cycle_priority',
    source='Read-only frozen V32 native ROI and full-scene shared arrays; unlabeled support allowed only by motion/source-domain policy; reject explicit different positive leaf IDs; no legacy pair caches',
    pair_candidates='all combinations of at most three manifest views; two views sufficient; no complete triple requirement',
    dedup='each candidate known different leaf exclusion then paired work coordinates quantized .25px, first row retained',
    train_track_rule='Within training only reject repeated-view components then prioritize 3-edge closed tracks inside original label budgets; 2-view edges remain allowed',
    graph='node=(original view ID, feature array index); edges=all retained authorized support matches from all candidate pairs; exact shared-feature connected components',
    holdout='sort components by minimum node, permute with seed20261030+group numeric ID; first in manifest pair order reserve seeded components until each eligible pair with >=28 raw matches has >=8 held edges; then reserve to global >=max(8,ceil(.2*all edges)); no pose or error used; heldout components never rank or fit',
    protection='Training edge excluded if either endpoint is <=2workpx from ANY global heldout endpoint in its original view; protect raw leaf geometry identically',
    train='cap128: positive leaf-ID round robin up to96, unassigned label0 spatial4D up to32; backfill positive first then label0; every source remains >2workpx from all global held endpoints',
    pair_rank='among input-eligible same-leaf-candidate pairs only: descending common valid positive-leaf training count, then total common valid training geometry across all three original V16 starts, minimum per-start valid count, ascending mean selected all-training robust Sampson cost, then capped training count, positive training leaf count, training4Dgrid, manifest combination order',
    chosen_holdout='only after pair rank choose spatial4D cap32 from this pair global-heldout edges; no heldout count/coordinate/error in rank',
    final='one selected pair only; original V16 six training-only seeds and three dense960 BA starts and strict/exploration gates; no best-of or post-failure fallback',
    dataset_reuse='Repeated cohort algorithm development; not an independent held-out test; independent branches never pooled',
    development_groups=a.DEV,no_identity_anchor_triangulation=True,no_source_K_Rt=True)
def w(p,x):a.write(p,x)
def freeze_development():
    pp=ROOT/'development_protocol.json';fs=[HERE,HERE.with_name('run_identity_camera_v1.py'),a.SOURCE,a.HELPER,a.PROFILE,a.V16/'run_public_camera_repair_v2.py',ROOT/'frontend_protocol_v1.json',a.MANIFEST,a.IDENTITIES,a.SCOPE]
    fs += [ROOT/'track_evidence.py',ROOT.parent/'segmentation/protocols/frozen_protocol_v1.json',ROOT.parent/'segmentation/code/run_segmentation_v1.py']
    rec=dict(schema='v31-local-candidate-pair-selection-protocol',rules=RULES,sources={str(p):a.sha(p) for p in fs},created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if pp.exists():assert a.read(pp)['sources']==rec['sources']
    else:w(pp,rec)
def verify():
    for p,h in a.read(ROOT/'development_protocol.json')['sources'].items():assert a.sha(p)==h,p
def semantic_training(c,indices,obs,labels):
    indices=a.np.asarray(indices,int)
    leaf_ids=sorted(set(labels[indices][labels[indices]>0].tolist()))
    pools=[list(c.spatial_select(indices[labels[indices]==label],obs,len(indices[labels[indices]==label]))) for label in leaf_ids]
    positive=[];offset=0
    while any(offset<len(pool) for pool in pools):
        for pool in pools:
            if offset<len(pool):positive.append(pool[offset])
        offset+=1
    zero=list(c.spatial_select(indices[labels[indices]==0],obs,len(indices[labels[indices]==0])))
    pos_take=min(96,len(positive));zero_take=min(32,len(zero));selected=positive[:pos_take]+zero[:zero_take]
    remain=128-len(selected);extra=positive[pos_take:pos_take+remain];selected+=extra;remain-=len(extra)
    selected+=zero[zero_take:zero_take+remain]
    return a.np.asarray(selected,int)
def prepare(g,c,init):
    out=ROOT/'inputs'/g['id'];receipt=out/'input_selection.json'
    if receipt.exists():return a.read(receipt)
    out.mkdir(parents=True,exist_ok=True);front=ROOT/'frontend'/g['id'];mr=a.read(front/'matching.json');vm={v['id']:v for v in g['views']};seg=a.read(a.V24/'segmentation/groups'/g['id']/'segmentation.json');sv={v['view_id']:v for v in seg['views']}
    labs={v:a.np.asarray(a.Image.open(x['label_mask_path'])) for v,x in sv.items() if x.get('label_mask_path')};pairs=[];parent={}
    def find(x):
        parent.setdefault(x,x)
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(x,y):
        x,y=find(x),find(y)
        if x!=y:parent[max(x,y)]=min(x,y)
    for order,pr in enumerate(mr['pairs']):
        pair=pr['actual_view_pair'];assert a.sha(pr['pair_npz'])==pr['pair_sha256'];z=a.np.load(pr['pair_npz'],allow_pickle=False);native=z['all_native'];labels=z['leaf_label'];features=z['feature_indices']
        assert native.shape==(len(labels),2,2) and features.shape==(len(labels),2)
        keep=a.np.ones(len(native),bool)
        if all(v in labs for v in pair):
            l0=a.labels_at(native[:,0],labs[pair[0]]);l1=a.labels_at(native[:,1],labs[pair[1]]);keep=~((l0>0)&(l1>0)&(l0!=l1));labels=a.np.where((l0>0)&(l0==l1),l0,0)
        native,labels,features=native[keep],labels[keep],features[keep];meta=[a.metadata(vm[v]) for v in pair];obs=a.np.stack([a.tf(native[:,i],meta[i]['native_to_work']) for i in (0,1)],1)
        _,ix=a.np.unique(a.np.rint(obs.reshape(-1,4)/.25).astype(a.np.int64),axis=0,return_index=True);ix=a.np.sort(ix);native,labels,features,obs=native[ix],labels[ix],features[ix],obs[ix]
        nodes=[]
        for label,ff in zip(labels,features):
            nn=tuple((v,int(fi)) for v,fi in zip(pair,ff));union(*nn);nodes.append(nn)
        eligible_leaves=[l['leaf_id'] for l in seg['leaves'] if all(l['views'].get(v,{}).get('status')=='candidate_generated' for v in pair)]
        pairs.append(dict(eligible=bool(eligible_leaves),eligible_leaves=eligible_leaves,order=order,pair=pair,native=native,labels=labels,features=features,obs=obs,nodes=nodes,source_pair_npz=pr['pair_npz'],source_pair_sha256=pr['pair_sha256']))
    components={}
    for pi,p in enumerate(pairs):
        for ei,nodes in enumerate(p['nodes']):components.setdefault(find(nodes[0]),[]).append((pi,ei))
    keys=sorted(components);seed=20261030+int(g['id'][1:]);order=a.np.random.default_rng(seed).permutation(len(keys));permuted=[keys[i] for i in order];target=max(8,math.ceil(.2*sum(len(p['obs']) for p in pairs)));heldset=set();actual=0
    pair_held_counts=a.np.zeros(len(pairs),int);allocation=[]
    def reserve(key,stage,pair_order=None):
        nonlocal actual
        if key in heldset:return
        heldset.add(key);actual+=len(components[key])
        for pi,ei in components[key]:pair_held_counts[pi]+=1
        allocation.append(dict(component_key=key,stage=stage,pair_order=pair_order,edges=components[key]))
    for pi,p in enumerate(pairs):
        if not p['eligible'] or len(p['obs'])<28:continue
        for key in permuted:
            if pair_held_counts[pi]>=8:break
            if any(i==pi for i,j in components[key]):reserve(key,'pair_minimum8',pi)
    for key in permuted:
        if actual>=target:break
        reserve(key,'global_twenty_percent')
    w(out/'global_component_allocation.json',dict(seed=seed,components=[dict(key=k,edges=components[k]) for k in keys],permutation=permuted,allocation=allocation,final_held_counts_by_pair=pair_held_counts,global_held_edges=actual,target_edges=target,policy_changed_from_v29=True,selected_before_any_pose_fit=True))
    heldcoords={v:[] for v in vm}
    for p in pairs:
        p['global_held']=a.np.array([find(nn[0]) in heldset for nn in p['nodes']],bool)
        for j,v in enumerate(p['pair']):heldcoords[v].extend(p['obs'][p['global_held'],j])
    for v in heldcoords:heldcoords[v]=a.np.asarray(heldcoords[v],float).reshape(-1,2)
    records=[]
    for p in pairs:
        protected=p['global_held'].copy()
        for j,v in enumerate(p['pair']):
            hc=heldcoords[v]
            if len(hc) and len(p['obs']):
                # Bounded blocks avoid full NxM allocations on very dense images.
                for lo in range(0,len(p['obs']),256):protected[lo:lo+256]|=a.np.min(a.np.linalg.norm(p['obs'][lo:lo+256,j,None]-hc[None],axis=2),axis=1)<=2
        p['protected']=protected
    flags=classify_tracks(pairs)
    for p,track_flags in zip(pairs,flags):
        protected=p['protected']
        p['track_flags']=track_flags
        a.np.save(out/f"candidate_{p['order']}_track_flags.npy",track_flags)
        train=tiered_training(c,a.np.flatnonzero(~protected),p['obs'],p['labels'],track_flags);leafcount=len(set(p['labels'][train][p['labels'][train]>0].tolist()));grid=len(set(tuple(q) for q in a.np.clip(a.np.floor(p['obs'][train].reshape(-1,4)/250),0,3).astype(int).tolist()))
        p.update(protected=protected,train=train)
        training_geometry=dict(status='input_ineligible_or_insufficient_training',common_positive_count=0,common_count=0,minimum_geometry_count=0,mean_robust_training_cost=1e300)
        ip=None
        if p['eligible'] and len(train)>=20:
            initial,init_audit=init.build_initializations(p['obs'][train],c,seed)
            ip=out/f"candidate_{p['order']}_initializations.json"
            w(ip,dict(training_xy=p['obs'][train],seed=seed,initializer_source_sha256=a.sha(a.HELPER),initializations=initial,audit=init_audit))
            if all(q['status']=='initialized' for q in initial):
                masks=[a.np.asarray(q['initial_geometry_mask'],bool) for q in initial]
                common=a.np.logical_and.reduce(masks)
                costs=[float(q['selection_rank_key'][1]) for q in initial]
                training_geometry.update(status='three_starts_initialized',common_positive_count=int(((p['labels'][train]>0)&common).sum()),common_count=int(common.sum()),minimum_geometry_count=min(int(m.sum()) for m in masks),mean_robust_training_cost=float(a.np.mean(costs)))
            else:training_geometry['status']='one_or_more_initializations_failed'
        p.update(training_geometry=training_geometry,initialization_path=ip,rank=(training_geometry['common_positive_count'],training_geometry['common_count'],training_geometry['minimum_geometry_count'],-training_geometry['mean_robust_training_cost'],len(train),leafcount,grid,-p['order']))
        records.append(dict(training_geometry=p['training_geometry'],initialization_path=str(p['initialization_path']) if p['initialization_path'] else None,initialization_sha256=a.sha(p['initialization_path']) if p['initialization_path'] else None,input_eligible=p['eligible'],eligible_leaf_ids=p['eligible_leaves'],order=p['order'],actual_view_pair=p['pair'],raw_unique=len(p['obs']),training_count=len(train),training_leaf_count=leafcount,training_grid_bucket_count=grid,rank_key=list(p['rank']),global_heldout_count_diagnostic_not_ranking=int(p['global_held'].sum()),source_pair_npz=p['source_pair_npz'],source_pair_sha256=p['source_pair_sha256']))
    eligible=[p for p in pairs if p['eligible']];chosen=max(eligible,key=lambda p:p['rank']) if eligible else pairs[0];held=c.spatial_select(a.np.flatnonzero(chosen['global_held']),chosen['obs'],32);selected=a.np.r_[chosen['train'],held]
    if not eligible:chosen['train']=a.np.empty(0,int);held=a.np.empty(0,int);selected=a.np.empty(0,int)
    a.np.save(out/'selected_track_flags.npy',chosen['track_flags'])
    a.np.savez_compressed(out/'selected_input.npz',all_native=chosen['native'],leaf_label=chosen['labels'],feature_indices=chosen['features'],all_work=chosen['obs'],selected=selected,train_indices=chosen['train'],heldout_indices=held,heldout_protected=chosen['protected'])
    # Save ALL candidates and split labels for independent leakage/ranking audit.
    for p in pairs:a.np.savez_compressed(out/f"candidate_{p['order']}.npz",all_native=p['native'],all_work=p['obs'],leaf_label=p['labels'],feature_indices=p['features'],global_heldout=p['global_held'],heldout_protected=p['protected'],train_indices=p['train'])
    rec=dict(selected_initialization_path=str(chosen['initialization_path']) if chosen['initialization_path'] else None,selected_initialization_sha256=a.sha(chosen['initialization_path']) if chosen['initialization_path'] else None,any_eligible_pair=bool(eligible),group_id=g['id'],actual_view_pair=chosen['pair'],selected_pair_order=chosen['order'],candidates=records,global_split=dict(connected_components=len(keys),heldout_components=len(heldset),heldout_edges=actual,target_edges=target,seed=seed),selected_input_path=str(out/'selected_input.npz'),selected_input_sha256=a.sha(out/'selected_input.npz'),protocol_sha256=a.sha(ROOT/'development_protocol.json'),source_matching_sha256=a.sha(front/'matching.json'),selection_before_final_BA=True,training_only_initialization_for_pair_selection=True,heldout_used_for_pair_ranking=False)
    w(receipt,rec);print('SELECT',g['id'],chosen['pair'],chosen['rank'],flush=True);return rec
def run_one(g,c,init,gates):
    r=a.read(ROOT/'inputs'/g['id']/'input_selection.json');assert a.sha(r['selected_input_path'])==r['selected_input_sha256'];z=a.np.load(r['selected_input_path'],allow_pickle=False)
    views={v['id']:v for v in g['views']};copy=dict(g,views=[views[v] for v in r['actual_view_pair']]);a.cached=lambda *args:(a.np.empty((0,2,2)),[]);a.rootsift=lambda *args:None;a.new_matches=lambda *args:z['all_native'];a.partition=lambda *args:(z['selected'],z['train_indices'],z['heldout_indices'],z['heldout_protected'])
    cached_init=init
    if r.get('selected_initialization_path'):
        assert a.sha(r['selected_initialization_path'])==r['selected_initialization_sha256']
        prepared=a.read(r['selected_initialization_path'])
        def from_cache(train,core,seed):
            assert seed==prepared['seed'] and a.sha(a.HELPER)==prepared['initializer_source_sha256']
            assert a.np.array_equal(train,a.np.asarray(prepared['training_xy'],float))
            initial=[]
            for item in prepared['initializations']:
                q=dict(item)
                for key in ('K','R','t','xyz','initial_geometry_mask','initial_errors_work_px','initial_ray_angles_deg','initial_positive'):
                    if key in q:q[key]=a.np.asarray(q[key],bool if key in ('initial_geometry_mask','initial_positive') else float)
                initial.append(q)
            return initial,dict(prepared['audit'],precomputed_input_hash_and_seed_verified=True,precomputed_path=r['selected_initialization_path'],precomputed_sha256=r['selected_initialization_sha256'])
        cached_init=a.types.SimpleNamespace(build_initializations=from_cache)
    return a.run(copy,c,cached_init,gates)
def worker(ids):
    a.ROOT=ROOT;verify();gm={g['id']:g for g in a.read(a.MANIFEST)['groups']};c,init,gates=a.core()
    for gid in ids:run_one(gm[gid],c,init,gates)
    return len(ids)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare_dev','run_dev','freeze','prepare_all','run_all']);mode=ap.parse_args().mode;a.ROOT=ROOT
    if mode=='prepare_dev':freeze_development()
    verify();groups=a.read(a.MANIFEST)['groups'];isdev=mode.endswith('dev');groups=[g for g in groups if g['id'] in a.DEV] if isdev else [g for g in groups if g['id'] in a.read(ROOT.parent/'scope/priority39_group_ids_v1.json')['group_ids']]
    if mode in ('prepare_all','run_all'):
        fr=a.read(ROOT/'application_frozen.json')
        for p,h in {**fr['sources'],**fr['development_results']}.items():assert a.sha(p)==h
    if mode.startswith('prepare'):
        c,init,_=a.core()
        for g in groups:prepare(g,c,init)
        return
    if mode=='freeze':
        paths=[ROOT/'results'/gid/'camera_result.json' for gid in a.DEV];assert all(p.exists() for p in paths);assert all(a.read(p)['status']!='runtime_failure' for p in paths)
        w(ROOT/'application_frozen.json',dict(sources=a.read(ROOT/'development_protocol.json')['sources'],development_results={str(p):a.sha(p) for p in paths},rules=RULES,independent_branch_no_best_of=True));return
    ids=[g['id'] for g in groups if not (ROOT/'results'/g['id']/'camera_result.json').exists()]
    if False:worker(ids)
    else:
        w(ROOT/'dispatch_v1.json',dict(partitions=[ids[::2],ids[1::2]],workers=2,threads_each=1,runner_sha256=a.sha(HERE)))
        with ProcessPoolExecutor(max_workers=2) as pool:print([q.result() for q in [pool.submit(worker,part) for part in (ids[::2],ids[1::2])]],flush=True)
if __name__=='__main__':main()
