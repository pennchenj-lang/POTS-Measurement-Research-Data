"""Single frozen native leaf-texture frontend; no geometry-based matching selection."""
import run_identity_camera_v1 as a
import itertools,importlib.util,datetime,argparse,time,traceback
ROOT=a.ROOT;HERE=a.Path(__file__).resolve()
SRC=a.P/'work/learned_camera_20261004_v1/official_source/lightglue/lightglue.py';WEIGHT=a.P/'models/lightglue_sift_20261004/sift_lightglue.pth'
RULES=dict(full_long=2048,full_detect=8192,scene_cap=4096,leaf_total_cap=4096,per_leaf_cap=1024,leaf_crop_margin=16,leaf_crop_long=1536,leaf_contrast=.008,scene_contrast=.015,native_dedup_grid=.5,upsampling=False,
 matcher=dict(filter_threshold=.1,depth_confidence=.95,width_confidence=.99,flash=True,mp=False),
 union='Local same-positive-label matches first then full-array matches, descending score then feature indices; both endpoints must be unused within pair; explicit different positive labels rejected.',
 feature_identity='One deduplicated shared master array per original view; original view/feature index preserved across matching contexts and all view pairs.',
 camera_and_surface_thresholds='unchanged',selection_uses_pose_or_heldout=False,source_camera_used=False)
def freeze():
 files=[HERE,HERE.with_name('run_identity_camera_v1.py'),HERE.with_name('camera_numeric_v1.py'),a.MANIFEST,a.IDENTITIES,a.SCOPE,SRC,WEIGHT,ROOT.parent/'segmentation/protocols/frozen_protocol_v1.json',ROOT.parent/'scope/development8_control4_frozen_v1.json']
 rec=dict(schema='v32-native-leaf-texture-frontend-v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rules=RULES,sources={str(p):a.sha(p) for p in files});p=ROOT/'frontend_protocol_v1.json'
 if p.exists():assert a.read(p)['sources']==rec['sources']
 else:a.write(p,rec)
def verify():
 for p,h in a.read(ROOT/'frontend_protocol_v1.json')['sources'].items():assert a.sha(p)==h,p
def fullscene(g):return (g['source']=='self' and int(g['id'][1:])>=12) or g.get('source_dataset')=='CO3D'
def features(gray,mask,x0,y0,long,cap,contrast):
 h,w=gray.shape;s=min(1.,long/max(h,w));rw,rh=round(w*s),round(h*s)
 img=a.cv2.resize(gray,(rw,rh),interpolation=a.cv2.INTER_AREA) if s<1 else gray
 mm=a.cv2.resize(mask,(rw,rh),interpolation=a.cv2.INTER_NEAREST) if mask is not None and s<1 else mask
 kp,d=a.cv2.SIFT_create(nfeatures=cap,contrastThreshold=contrast).detectAndCompute(img,mm)
 if not kp:return []
 d=a.np.sqrt(a.np.maximum(d/(d.sum(1,keepdims=True)+1e-12),1e-6));d/=a.np.linalg.norm(d,axis=1,keepdims=True)+1e-12
 return [dict(xy=(a.np.asarray(q.pt)+.5)/[rw/w,rh/h]-.5+[x0,y0],desc=z,scale=q.size/((rw/w*rh/h)**.5),ori=a.np.deg2rad(q.angle),response=q.response) for q,z in zip(kp,d)]
def extract(g):
 out=ROOT/'frontend'/g['id'];out.mkdir(parents=True,exist_ok=True);rp=out/'extraction.json'
 if rp.exists():return a.read(rp)
 sp=ROOT.parent/'segmentation/groups'/g['id']/'segmentation.json';seg=a.read(sp);vm={v['view_id']:v for v in seg['views']};records=[]
 for view in g['views']:
  vid=view['id'];ip=view.get('frozen_experiment_image_path',view['path']);expect=view.get('frozen_experiment_image_sha256',view['display_sha256']);assert a.sha(ip)==expect
  with a.Image.open(ip) as im:gray=a.np.asarray(a.ImageOps.exif_transpose(im).convert('L'))
  h,w=gray.shape;assert [w,h]==[view['width'],view['height']]
  lp=vm[vid].get('label_mask_path');lab=a.np.asarray(a.Image.open(lp)) if lp else a.np.zeros((h,w),a.np.uint16)
  base=features(gray,None if fullscene(g) else (lab>0).astype(a.np.uint8)*255,0,0,2048,8192,.015)
  base.sort(key=lambda q:(-q['response'],float(q['xy'][1]),float(q['xy'][0]),q['scale'],q['ori']))
  pools=[];roi={}
  for label in sorted(set(lab.ravel())-{0}):
   yy,xx=a.np.where(lab==label);x0=max(0,int(xx.min())-16);y0=max(0,int(yy.min())-16);x1=min(w,int(xx.max())+17);y1=min(h,int(yy.max())+17)
   local=features(gray[y0:y1,x0:x1],(lab[y0:y1,x0:x1]==label).astype(a.np.uint8)*255,x0,y0,1536,1024,.008)
   local=[q for q in local if a.labels_at(a.np.asarray([q['xy']]),lab)[0]==label]
   local.sort(key=lambda q:(-q['response'],float(q['xy'][1]),float(q['xy'][0]),q['scale'],q['ori']))
   pools.append(local);roi[str(int(label))]=[x0,y0,x1,y1]
  local=[];i=0
  while len(local)<4096 and any(i<len(pool) for pool in pools):
   for pool in pools:
    if i<len(pool) and len(local)<4096:local.append(pool[i])
   i+=1
  chosen=[];seen=set();localcount=0;scenecount=0
  for source,seq,cap in [(1,local,4096),(0,base,4096)]:
   n=0
   for q in seq:
    key=tuple(a.np.rint(q['xy']/.5).astype(int))
    if key in seen:continue
    label=int(a.labels_at(a.np.asarray([q['xy']]),lab)[0])
    if not fullscene(g) and label==0:continue
    seen.add(key);chosen.append(dict(q,label=label,source=source));n+=1
    if n>=cap:break
   if source:localcount=n
   else:scenecount=n
  native=a.np.asarray([q['xy'] for q in chosen],a.np.float64).reshape(-1,2);s=min(1.,2048/max(w,h));rw,rh=round(w*s),round(h*s)
  fp=out/f'view_{vid}_features.npz';a.np.savez_compressed(fp,native_xy=native,keypoints=((native+.5)*[rw/w,rh/h]-.5).astype(a.np.float32),descriptors=a.np.asarray([q['desc'] for q in chosen],a.np.float32).reshape(-1,128),native_scales=a.np.asarray([q['scale'] for q in chosen],a.np.float32),scales=a.np.asarray([q['scale']*((rw/w*rh/h)**.5) for q in chosen],a.np.float32),oris=a.np.asarray([q['ori'] for q in chosen],a.np.float32),image_size=a.np.asarray([rw,rh],a.np.float32),leaf_label=a.np.asarray([q['label'] for q in chosen],int),feature_source=a.np.asarray([q['source'] for q in chosen],int))
  records.append(dict(view_id=vid,status='extracted',count=len(chosen),local_features=localcount,scene_features=scenecount,domain='full_scene' if fullscene(g) else 'positive_leaf_label_union',roi_by_label=roi,feature_path=str(fp),feature_sha256=a.sha(fp),image_path=ip,image_sha256=expect,label_map_path=lp,label_map_sha256=a.sha(lp) if lp else None,native_size=[w,h]))
 result=dict(group_id=g['id'],actual_view_ids=[x['id'] for x in g['views']],views=records,segmentation_path=str(sp),segmentation_sha256=a.sha(sp),protocol_sha256=a.sha(ROOT/'frontend_protocol_v1.json'));a.write(rp,result);print('EXTRACT',g['id'],[(r['view_id'],r['local_features'],r['scene_features']) for r in records],flush=True);return result
def load_matcher():
 import torch
 torch.set_num_threads(1);torch.manual_seed(20261030)
 spec=importlib.util.spec_from_file_location('pinned_lightglue_v32',SRC);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 matcher=m.LightGlue(features=None,input_dim=128,add_scale_ori=True,weights=None,**RULES['matcher']);state=torch.load(WEIGHT,map_location='cpu',weights_only=True)
 for i in range(matcher.conf.n_layers):
  for part in ('self_attn','cross_attn'):state={k.replace(f'{part}.{i}',f'transformers.{i}.{part}'):v for k,v in state.items()}
 assert set(matcher.state_dict())-set(state)=={'confidence_thresholds'};state['confidence_thresholds']=matcher.confidence_thresholds.detach().clone();matcher.load_state_dict(state,strict=True)
 return matcher.eval().cuda(),torch
def tensors(z,indices,torch,roi=None):
 d={k:z[k][indices].copy() for k in ('keypoints','descriptors','scales','oris')};d['image_size']=z['image_size'].copy()
 if roi is not None:
  x0,y0,x1,y1=roi;cw,ch=x1-x0,y1-y0;s=min(1.,1536/max(cw,ch));rw,rh=round(cw*s),round(ch*s)
  d['keypoints']=((z['native_xy'][indices]-[x0,y0]+.5)*[rw/cw,rh/ch]-.5).astype(a.np.float32);d['scales']=(z['native_scales'][indices]*((rw/cw*rh/ch)**.5)).astype(a.np.float32);d['image_size']=a.np.asarray([rw,rh],a.np.float32)
 return {k:torch.from_numpy(v).unsqueeze(0).cuda() for k,v in d.items()}
def match(g,matcher,torch):
 out=ROOT/'frontend'/g['id'];rp=out/'matching.json'
 if rp.exists():return a.read(rp)
 ex=a.read(out/'extraction.json');vm={r['view_id']:r for r in ex['views']};pairs=[]
 for pair in itertools.combinations(ex['actual_view_ids'],2):
  tic=time.perf_counter();zs=[]
  for v in pair:assert a.sha(vm[v]['feature_path'])==vm[v]['feature_sha256'];zs.append(a.np.load(vm[v]['feature_path']))
  contexts=[None]+sorted((set(zs[0]['leaf_label'])&set(zs[1]['leaf_label']))-{0});edges=[];counts=[]
  for label in contexts:
   ix=[a.np.arange(len(z['native_xy'])) if label is None else a.np.flatnonzero(z['leaf_label']==label) for z in zs]
   if min(map(len,ix))<2:continue
   fs=[tensors(z,idx,torch,vm[v]['roi_by_label'].get(str(label)) if label is not None else None) for z,idx,v in zip(zs,ix,pair)]
   with torch.inference_mode():r=matcher(dict(image0=fs[0],image1=fs[1]))
   matched=r['matches'][0].cpu().numpy();score=r['scores'][0].cpu().numpy();del r,fs;torch.cuda.empty_cache()
   rejected=0
   for (i,j),ss in zip(matched,score):
    aa,bb=int(ix[0][i]),int(ix[1][j]);la,lb=int(zs[0]['leaf_label'][aa]),int(zs[1]['leaf_label'][bb])
    if la>0 and lb>0 and la!=lb:rejected+=1;continue
    edges.append((0 if label is not None else 1,-float(ss),aa,bb,la if la>0 and la==lb else 0))
   counts.append(dict(context='full' if label is None else int(label),raw_matches=len(matched),different_leaf_rejections=rejected))
  edges.sort();used0=set();used1=set();selected=[]
  for context,ns,i,j,label in edges:
   if i in used0 or j in used1:continue
   used0.add(i);used1.add(j);selected.append((i,j,label,-ns,context))
  ids=a.np.asarray([[q[0],q[1]] for q in selected],int).reshape(-1,2);labels=a.np.asarray([q[2] for q in selected],int)
  native=a.np.stack([z['native_xy'][ids[:,j]] for j,z in enumerate(zs)],axis=1);fp=out/f'pair_{pair[0]}_{pair[1]}.npz';a.np.savez_compressed(fp,all_native=native,leaf_label=labels,feature_indices=ids,scores=a.np.asarray([q[3] for q in selected]),matching_context=a.np.asarray([q[4] for q in selected],int))
  pairs.append(dict(actual_view_pair=list(pair),status='matched',contexts=counts,pair_npz=str(fp),pair_sha256=a.sha(fp),retained_matches=len(selected),positive_same_leaf_matches=int((labels>0).sum()),local_priority_matches=sum(q[4]==0 for q in selected),elapsed_seconds=time.perf_counter()-tic));print('MATCH',g['id'],pair,len(selected),'leaf',int((labels>0).sum()),flush=True)
 result=dict(group_id=g['id'],pairs=pairs,extraction_sha256=a.sha(out/'extraction.json'),protocol_sha256=a.sha(ROOT/'frontend_protocol_v1.json'));a.write(rp,result);return result
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['dev','all']);args=ap.parse_args()
 if args.stage=='dev':freeze()
 else:assert (ROOT/'application_frozen.json').exists()
 verify();ids=a.DEV if args.stage=='dev' else a.read(ROOT.parent/'scope/priority53_group_ids_v1.json')['group_ids'];groups=[g for g in a.read(a.MANIFEST)['groups'] if g['id'] in ids]
 for g in groups:extract(g)
 matcher,torch=load_matcher()
 for g in groups:match(g,matcher,torch)
 a.write(ROOT/f'frontend_{args.stage}_complete.json',dict(groups=ids,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
if __name__=='__main__':main()
