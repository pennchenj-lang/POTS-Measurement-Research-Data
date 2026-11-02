"""Development-only given-sequence-camera candidate-W surface experiments.

This adapter does not turn segmentation candidates into reviewed observations.
The reused POTS solver is unchanged; its observation compiler is explicitly
replaced by the candidate compiler at runtime. No camera fitting, GS, metric
scale or independent hidden-surface accuracy is claimed.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
for key in ('TEMP', 'TMP', 'TMPDIR', 'PIP_CACHE_DIR', 'TORCH_HOME', 'HF_HOME', 'MPLCONFIGDIR', 'XDG_CACHE_HOME'):
    value = ROOT / 'cache' / 'given_surface' / key.lower()
    value.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(value)
sys.path.insert(0, str(PROJECT / 'src'))

import numpy as np
import scipy
from PIL import Image, ImageDraw, __version__ as PIL_VERSION
from pots_gs.triangulation import Camera
from pots_gs.geometry import LEAF_FACES, Provenance, create_leaf_mesh, mesh_area
from pots_gs.subdivision import subdivision_levels

OLD = PROJECT / 'work/marker_guided_recovery_20261025_v9/code'
SOLVER = OLD / 'frozen_normal4_scaffold_v1.py'
GEOMETRY = PROJECT.parent / 'measurement_revision_v039/code/frozen_replay/snapshot/geometry_diagnostics.py'
DEV_PROTOCOL = ROOT / 'segmentation/protocols/sam_target_development_before_v1.json'
BOUNDARY = np.array([0, 1, 4, 7, 10, 9, 6, 3, 0])
EDGES = sorted({tuple(sorted((a, b))) for f in LEAF_FACES for a, b in zip(f, (*f[1:], f[0]))})
RULES = {
    'schema': 'given-camera-candidate-W-surface-development-v1',
    'source_camera': 'given official full-sequence SfM, not two-image self-calibration or independent 3D truth',
    'development_only': True,
    'source_groups': ['P090', 'P008', 'P091', 'P117', 'P072', 'P014'],
    'surface': 'POTS 11 controls / exact taskbook 12 triangles; all controls estimated',
    'solver': {'regularization': .2, 'max_nfev': 250, 'sigma_px': 2.,
               'differential_weight': 1., 'max_displacement': 1.},
    'adjacent_normal_weight': 4.,
    'candidate_visible_W_RMS_work_px': 8.,
    'geometry_threshold_source': 'unmodified V9 geometry.audit / EPS=1e-8 AREA_EPS=1e-12',
    'initializer': 'retained candidate W only: closest centroid rays give a depth PRIOR; first-view PCA extent lifted to first-view-facing plane',
    'initial_ray_min_angle_deg': 1.,
    'initial_station_u': [.25, .5, .75],
    'initial_width_profile': [.68, 1., .68],
    'centroids_are_material_correspondences': False,
    'PCA_endpoints_are_B_T_observations': False,
    'B_T_Q_observations': 0,
    'masks_used_as_complete_leaf_edges': False,
    'full_masks_used_for_initialization_or_fitting': False,
    'controlled_missing_fractions': [.2, .4, .6],
    'withheld_view': '01',
    'withheld_arc_selection': 'longest eligible open W path by polyline length, first path breaks ties; fixed before fitting',
    'withheld_interval': 'central contiguous ceil(fraction * segment_count) segments; index-based, actual length fraction reported',
    'evaluation_samples': 'interior .25/.5/.75 of withheld segments; no segment endpoints in evaluation',
    'control': 'same retained-two-view initialization, remove view02 W residual only; not a complete single-view pipeline',
    'outputs_are_candidate_diagnostics': True,
    'natural_hidden_surface_reference_available': False,
    'counted_verified_models_or_occlusion_successes': 0,
    'distortion': 'official JPEG pinhole assumption; no independently verified distortion coefficients',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with path.open('x', encoding='utf-8') as handle:
        handle.write(payload)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def transform(xy, matrix):
    xy = np.asarray(xy, float).reshape(-1, 2)
    h = np.column_stack((xy, np.ones(len(xy)))) @ np.asarray(matrix).T
    return h[:, :2] / h[:, 2, None]


def case_id(case):
    return case.get('group_id', case.get('cohort_id', case.get('case_id')))


def initialization(cameras, prepared):
    """No mask, full case, material endpoints or held-out samples are arguments."""
    by_view = {p['view_id']: np.asarray(p['W'], float) for p in prepared}
    if set(by_view) != {'01', '02'} or any(len(p) < 3 for p in by_view.values()):
        raise ValueError('Initialization needs retained candidate W in both selected views')
    # Exact repeated interpolation endpoints must not bias PCA/centroids.
    by_view = {v: np.unique(p, axis=0) for v, p in by_view.items()}
    centers2d = {v: p.mean(axis=0) for v, p in by_view.items()}
    rays = {}
    for view, c in cameras.items():
        d = c.R.T @ np.linalg.solve(c.K, np.r_[centers2d[view], 1.])
        rays[view] = d / np.linalg.norm(d)
    d1, d2 = rays['01'], rays['02']
    angle = float(np.degrees(np.arccos(np.clip(abs(d1 @ d2), -1, 1))))
    if angle < RULES['initial_ray_min_angle_deg']:
        raise ValueError(f'Centroid-ray depth prior ill conditioned: acute angle={angle:.6g} deg')
    c1, c2 = cameras['01'], cameras['02']
    distances = np.linalg.lstsq(np.column_stack((d1, -d2)), c2.center-c1.center, rcond=None)[0]
    if np.any(distances <= 0):
        raise ValueError('Centroid-ray closest point prior lies behind a camera')
    ray_points = np.array([c1.center + distances[0]*d1, c2.center + distances[1]*d2])
    origin = ray_points.mean(axis=0)
    normal = c1.R[2].copy()
    points = by_view['01']
    center = centers2d['01']
    _, _, vh = np.linalg.svd(points-center, full_matrices=False)
    axis = vh[0]
    if axis[1] < 0 or (axis[1] == 0 and axis[0] < 0):
        axis = -axis
    side = np.array([-axis[1], axis[0]])
    longitudinal = (points-center) @ axis
    lateral = (points-center) @ side
    lo, hi = float(longitudinal.min()), float(longitudinal.max())
    left, right = float(lateral.min()), float(lateral.max())
    if min(hi-lo, right-left) <= 1e-6:
        raise ValueError('Retained W does not support a nondegenerate PCA extent')
    # Canonical topology endpoint slots are geometric hypotheses, not B/T observations.
    uv = np.empty((11, 2))
    uv[0], uv[10] = center + lo*axis, center + hi*axis
    for j, (u, width) in enumerate(zip(RULES['initial_station_u'], RULES['initial_width_profile'])):
        middle = center + (lo + u*(hi-lo))*axis
        uv[1+3*j] = middle + width*left*side
        uv[2+3*j] = middle
        uv[3+3*j] = middle + width*right*side
    xyz = []
    for pixel in uv:
        ray = c1.R.T @ np.linalg.solve(c1.K, np.r_[pixel, 1.])
        distance = float((origin-c1.center)@normal / (ray@normal))
        if distance <= 0:
            raise ValueError('Initialization plane reached behind first camera')
        xyz.append(c1.center + distance*ray)
    xyz = np.asarray(xyz)
    if any(np.any((xyz@c.R.T+c.t)[:, 2] <= 0) for c in cameras.values()):
        raise ValueError('Initialized controls have nonpositive depth')
    return xyz, {
        'mode': RULES['initializer'], 'depth_is_unverified_prior': True,
        'retained_W_unique_counts': {v: len(p) for v, p in by_view.items()},
        'retained_W_sha256': {v: hashlib.sha256(p.astype('<f8').tobytes()).hexdigest() for v,p in by_view.items()},
        'centroids_work': centers2d, 'centroid_ray_acute_angle_deg': angle,
        'closest_ray_gap_source_units': float(np.linalg.norm(ray_points[0]-ray_points[1])),
        'closest_ray_distances_source_units': distances, 'plane_origin': origin, 'plane_normal': normal,
        'reference_view': '01', 'initial_controls_work_01': uv,
        'PCA_endpoint_slots_are_observed_anatomical_B_T': False,
        'full_mask_used': False, 'heldout_points_used': False,
        'correspondences_triangulated_as_observations': 0,
    }


def geometry_and_projection(solver, geometry, xyz, initial, cameras, prepared):
    ga = geometry.audit(xyz, np.asarray(LEAF_FACES), initial)
    records = []
    for item in prepared:
        view = item['view_id']
        uv, z = solver.project_xyz(cameras[view], xyz)
        residual, _ = solver.polyline_residual(item['W'], uv[BOUNDARY], ordered=False)
        rms = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
        records.append({'view_id': view, 'retained_candidate_W_rms_work_px': rms,
                        'retained_candidate_W_within_original_8_work_px': rms <= 8.,
                        'all_depth_positive': bool(np.all(z > 0))})
    geometry_ok = (not ga['triangle_intersection_pairs'] and
                   not ga['degenerate_faces_at_normalized_area_1e_minus12'] and not ga['nonmanifold_edges'] and
                   all(r['all_depth_positive'] for r in records))
    return {'geometry': ga, 'views': records,
            'candidate_internal_consistency_passed': bool(geometry_ok and all(r['retained_candidate_W_within_original_8_work_px'] for r in records)),
            'surface_geometry_consistent': bool(geometry_ok), 'reviewed_observation_consistency_passed': None,
            'not_independent_accuracy': True, 'natural_hidden_surface_accuracy': None}


def overlay(out, view, meta, projected_native, candidate_points_native, heldout_native):
    image = Image.open(meta['image_path']).convert('RGB')
    points = np.vstack([projected_native, candidate_points_native,
                        np.asarray(heldout_native).reshape(-1, 2)])
    margin = max(20., .25*float(np.max(np.ptp(points, axis=0))))
    x0, y0 = np.maximum(np.floor(points.min(0)-margin), [0, 0]).astype(int)
    x1, y1 = np.minimum(np.ceil(points.max(0)+margin), image.size).astype(int)
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Projected overlay crop does not intersect image')
    draw = ImageDraw.Draw(image)
    for a,b in EDGES:
        draw.line([tuple(projected_native[a]), tuple(projected_native[b])], fill=(255,70,45), width=2)
    for xy in candidate_points_native:
        draw.ellipse((xy[0]-1.5,xy[1]-1.5,xy[0]+1.5,xy[1]+1.5), fill=(50,240,60))
    for xy in heldout_native:
        draw.ellipse((xy[0]-2,xy[1]-2,xy[0]+2,xy[1]+2), fill=(30,180,255))
    image.save(out/f'projection_view{view}_native.jpg', quality=93)
    crop = image.crop((int(x0),int(y0),int(x1),int(y1)))
    scale = min(3., 850/max(crop.size))
    crop = crop.resize((max(1,round(crop.width*scale)),max(1,round(crop.height*scale))))
    board = Image.new('RGB',(max(650,crop.width),crop.height+55),'white')
    board.paste(crop,(0,55))
    ImageDraw.Draw(board).text((8,8),f'{view}: red estimated mesh | green retained candidate W | cyan held out',fill='black')
    ImageDraw.Draw(board).text((8,28),'Given sequence SfM / unknown scale / candidates are not truth',fill='black')
    board.save(out/f'projection_view{view}_crop.jpg',quality=93)
    return {'native_image':str(out/f'projection_view{view}_native.jpg'),
            'crop_image':str(out/f'projection_view{view}_crop.jpg'), 'crop_native_xyxy':[int(x0),int(y0),int(x1),int(y1)]}


def retain_only(case):
    result = copy.deepcopy(case)
    for v in result['views'].values():
        for key in list(v):
            if key.startswith('initialization_support') or 'axis' in key or 'endpoint_hypothes' in key:
                del v[key]
    result['initialization_policy'] = 'recomputed from currently retained prepared W; no full mask or cached PCA'
    return result


def controlled_split(case, fraction):
    result = retain_only(case)
    paths = result['views']['01']['W']
    eligible = [(i,p) for i,p in enumerate(paths) if len(p['xy_native']) >= 5 and
                all(s == 'candidate_natural_margin' for s in p['segment_evidence'])]
    if not eligible:
        raise ValueError('No sufficiently long visually qualified candidate natural-margin arc')
    index, path = max(eligible, key=lambda ip: (np.linalg.norm(np.diff(ip[1]['xy_native'],axis=0),axis=1).sum(), -ip[0]))
    xy = np.asarray(path['xy_native'], float)
    n = len(xy)-1
    count = int(np.ceil(fraction*n))
    start = (n-count)//2
    stop = start+count  # held-out edges [start, stop)
    if start < 1 or stop > n-1:
        raise ValueError('Controlled interval must leave retained portions on both sides')
    held = np.vstack([xy[i]+t*(xy[i+1]-xy[i]) for i in range(start,stop) for t in (.25,.5,.75)])
    parts = []
    for tag, lo, hi in [('before',0,start+1), ('after',stop,len(xy))]:
        p = copy.deepcopy(path)
        p['path_id'] = path['path_id'] + '_retained_' + tag
        p['xy_native'] = p['xy_native'][lo:hi]
        p['point_visibility'] = p['point_visibility'][lo:hi]
        p['segment_evidence'] = p['segment_evidence'][lo:hi-1]
        if 'source_contour_indices' in p:
            p['source_contour_indices'] = p['source_contour_indices'][lo:hi]
        assert len(p['xy_native']) >= 2
        parts.append(p)
    result['views']['01']['W'] = paths[:index] + parts + paths[index+1:]
    lengths = np.linalg.norm(np.diff(xy,axis=0),axis=1)
    split = {'type':'controlled_missing_image_candidate_edge', 'reference_is_independent_truth':False,
             'withheld_view':'01', 'source_path_id':path['path_id'], 'source_path_index':index,
             'requested_segment_fraction':fraction, 'source_segment_count':n,
             'withheld_segment_interval_half_open':[start,stop], 'withheld_segment_count':count,
             'actual_segment_fraction':count/n,
             'actual_arc_length_fraction':float(lengths[start:stop].sum()/lengths.sum()),
             'evaluation_points_native':held, 'evaluation_count':len(held),
             'selection_used_fit':False, 'samples_are_candidate_edge_interpolants':True,
             'evaluated_segment_endpoints':False,
             'same_initialization_for_with_and_without_second_view_residual':True}
    result['controlled_missing'] = {k:v for k,v in split.items() if k != 'evaluation_points_native'}
    return result, split


def camera_metas(camera):
    return {v['view_id']:{**v,'image_sha256':v['source_sha256']} for v in camera['views']}


def freeze(compiler_path, case_paths, out):
    out = Path(out)
    if out.exists():
        raise FileExistsError('Frozen protocol output already exists: '+str(out))
    dev = read(DEV_PROTOCOL)
    assert [r['cohort_id'] for r in dev['development']] == RULES['source_groups']
    sources = [Path(__file__), Path(compiler_path), SOLVER, GEOMETRY, DEV_PROTOCOL,
               PROJECT/'src/pots_gs/geometry.py', PROJECT/'src/pots_gs/triangulation.py',
               PROJECT/'src/pots_gs/subdivision.py', OLD/'recovery_driver_v1.py']
    cases = {case_id(read(p)):Path(p) for p in case_paths}
    if set(cases) != set(RULES['source_groups']):
        raise ValueError('Exactly all six fixed development cases required')
    compiler = load('candidate_compiler_freeze',Path(compiler_path))
    # All candidate paths/splits and input hashes are saved before any fitting.
    out.mkdir(parents=True,exist_ok=False)
    protocol = {'rules':RULES, 'source_files_sha256':{str(p):sha(p) for p in sources},
                'compiler_path':str(Path(compiler_path)),
                'runtime':{'python':sys.version,'executable':sys.executable,'numpy':np.__version__,
                           'scipy':scipy.__version__,'PIL':PIL_VERSION,'CPU_threads':1,'GPU_used':False},
                'development_groups':[], 'created_before_fitting':True,
                'expected_trials_with_camera':28, 'independent_surface_truth_available':False}
    for cid in RULES['source_groups']:
        cp=cases[cid];case=read(cp)
        campath=ROOT/'metadata/cameras'/cid/'official_pair_camera_v1.json'
        row={'group_id':cid,'original_case_path':str(cp),'original_case_sha256':sha(cp),
             'given_camera_available':campath.exists(),'conditions':[]}
        if not campath.exists():
            row['status']='source_provided_camera_unavailable_not_fitted'
            protocol['development_groups'].append(row)
            continue
        camera=read(campath);metas=camera_metas(camera)
        if camera.get('given_camera_branch') is not True:
            raise ValueError('Requires explicit given source camera branch')
        row.update(camera_path=str(campath),camera_sha256=sha(campath))
        for name,fraction in [('full',None)]+[(f'missing{int(f*100):02}',f) for f in RULES['controlled_missing_fractions']]:
            frozen, evaluation = (retain_only(case),None) if fraction is None else controlled_split(case,fraction)
            prepared,audit,reasons=compiler.compile_candidate_case(frozen,metas,allow_model_candidate_constraints=True)
            assert all(not p['material'] and not len(p['Q']) for p in prepared)
            if evaluation:
                held=transform(evaluation['evaluation_points_native'],metas['01']['native_to_work'])
                train=next(p['W'] for p in prepared if p['view_id']=='01')
                distances=np.linalg.norm(held[:,None,:]-train[None,:,:],axis=2).min(axis=1)
                if float(distances.min()) <= 1e-8:
                    raise ValueError('Exact retained/held-out sample overlap')
                evaluation['min_heldout_to_retained_sample_distance_work_px']=float(distances.min())
                evaluation['heldout_is_spatially_correlated_same_source_not_independent_sample']=True
            frozen_path=out/'cases'/cid/(name+'_candidate_constraints.json')
            dump(frozen_path,frozen)
            cond={'condition':name,'fraction':fraction,'case_path':str(frozen_path),'case_sha256':sha(frozen_path),
                  'prepared_counts':{p['view_id']:len(p['W']) for p in prepared},'candidate_audit':audit,
                  'reasons':reasons,'branches':['two_view_W'] if fraction is None else ['two_view_W','without_view02_W_residual']}
            if evaluation:
                ep=out/'cases'/cid/(name+'_evaluation_only.json');dump(ep,evaluation)
                cond.update(evaluation_path=str(ep),evaluation_sha256=sha(ep))
            row['conditions'].append(cond)
        protocol['development_groups'].append(row)
    path=out/'protocol_before_fit_v1.json';dump(path,protocol)
    print(json.dumps({'status':'frozen_before_fit','protocol':str(path),'sha256':sha(path),'actual_fits_run':0}),flush=True)
    return path


def run_trial(protocol_path, protocol, row, condition, branch, out):
    out=Path(out)
    if out.exists():
        raise FileExistsError('Preserve previous trial: '+str(out))
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    receipt={'group_id':row['group_id'],'condition':condition['condition'],'branch':branch,
             'data_type':'real','evidence_class':'model_candidate_visible_natural_margin',
             'author_reviewed':False,'camera_mode':'source_provided_sequence_SfM',
             'two_image_camera_recovery_claim':False,'physical_scale_available':False,
             'independent_3D_reference_available':False,'natural_hidden_surface_reference_available':False,
             'protocol_path':str(protocol_path),'protocol_sha256':sha(protocol_path),
             'camera_path':row['camera_path'],'camera_sha256':row['camera_sha256'],
             'case_path':condition['case_path'],'case_sha256':condition['case_sha256'],
             'verified_model_or_occlusion_success_claim':False,
             'all_failures_retained':True,'heldout_used_for_initialization_optimization_selection':False}
    try:
        assert sha(row['camera_path'])==row['camera_sha256']
        assert sha(condition['case_path'])==condition['case_sha256']
        camera=read(row['camera_path']);metas=camera_metas(camera);case=read(condition['case_path'])
        cams={v:Camera(v,c['K'],c['R'],c['t'],unit=c['unit']) for v,c in camera['cameras_work'].items()}
        compiler=load('candidate_compiler_trial',protocol['compiler_path'])
        prepared,audit,reasons=compiler.compile_candidate_case(case,metas,allow_model_candidate_constraints=True)
        initial,initial_audit=initialization(cams,prepared)
        dump(out/'initialization.json',{'xyz':initial,'faces':LEAF_FACES,**initial_audit})
        dump(out/'candidate_constraints_passed_to_solver.json',{'prepared':prepared,'audit':audit,'reasons':reasons,
             'training_views':['01','02'] if branch=='two_view_W' else ['01'],
             'candidate_to_true_margin_relabeling':False,'B_T_Q_observations':0})
        solver=load('v12_unchanged_normal4_solver',SOLVER)
        geometry=load('v12_unchanged_geometry_audit',GEOMETRY)
        assert solver.NORMAL_WEIGHT==RULES['adjacent_normal_weight']
        used_views=['01','02'] if branch=='two_view_W' else ['01']
        fit_cams={v:cams[v] for v in used_views}
        fitted_prepared=[p for p in prepared if p['view_id'] in used_views]
        fitted_audit=[p for p in audit if p['view_id'] in used_views]
        def candidate_prepared_adapter(cameras,observations,**kwargs):
            if set(cameras)!=set(used_views) or kwargs.get('organ_id')!=case['organ_id']:
                raise ValueError('Runtime prepared-candidate scope mismatch')
            return fitted_prepared,fitted_audit,reasons+['Explicit candidate compiler replaces reviewed-margin compiler; no candidate relabeling']
        solver.prepare_observations=candidate_prepared_adapter
        fitted=solver.fit_scaffold(initial,fit_cams,[],organ_id=case['organ_id'],data_type='real',
                                   allow_unreviewed_diagnostic=True,allow_single_view_diagnostic=(len(used_views)==1),**RULES['solver'])
        fitted.mesh.coordinate_system='Original CO3D sequence SfM world coordinates; unknown physical scale'
        fitted.mesh.metadata.update({'source_camera_mode':'given_full_sequence_SfM','evidence_class':'model_candidate_visible_natural_margin',
            'author_reviewed':False,'verified_observations':False,'canonical_B_T_labels_are_observed_anatomical_endpoints':False,
            'all_controls_estimated':True,'W_effective_semantics':'one-way predicted candidate margin samples, not verified true margins',
            'hidden_shape_is_prior_regularized_estimate':True,'actual_training_views':used_views,
            'full_masks_used_for_initialization_or_fitting':False,'heldout_points_used_for_initialization_or_fitting':False})
        fitted.mesh.vertices=[replace(vertex,provenance=Provenance('estimated',tuple(used_views),
             tuple(f'{v}:model_candidate_W' for v in used_views),
             'Given sequence SfM; regularized candidate W fit; canonical B/T slots are not observed anatomical endpoints'))
             for vertex in fitted.mesh.vertices]
        levels=subdivision_levels(fitted.mesh,2)
        for mesh in levels:
            mesh.write_json(out/f'model_L{mesh.level}.json');mesh.write_obj(out/f'model_L{mesh.level}.obj')
        areas=[mesh_area(m) for m in levels]
        assert np.allclose(areas,areas[0],rtol=1e-10,atol=1e-12)
        diagnostics={**fitted.diagnostics,'effective_evidence_class':'model_candidate_visible_natural_margin',
                     'original_solver_W_wording_is_not_evidence_reclassification':True,
                     'observations_passed_by_explicit_candidate_compiler':True,'candidate_audit':audit,
                     'initializer_uses_retained_two_view_W':True,
                     'without_second_view_branch_is_residual_ablation_not_single_view_reconstruction':branch!='two_view_W'}
        dump(out/'solver_diagnostics.json',diagnostics)
        consistency=geometry_and_projection(solver,geometry,fitted.xyz,initial,cams,prepared)
        dump(out/'candidate_internal_consistency.json',consistency)
        # Evaluation file is first opened only after fitting and mesh export.
        evaluation=None
        if condition.get('evaluation_path'):
            assert sha(condition['evaluation_path'])==condition['evaluation_sha256']
            evaluation=read(condition['evaluation_path'])
        projections={};evaluation_result=None
        for item in prepared:
            v=item['view_id'];uv,z=solver.project_xyz(cams[v],fitted.xyz)
            native=transform(uv,metas[v]['work_to_native'])
            held=np.empty((0,2)) if evaluation is None or v!='01' else np.asarray(evaluation['evaluation_points_native'])
            points_native=transform(item['W'],metas[v]['work_to_native'])
            projections[v]={'vertices_work':uv,'vertices_native':native,'depth':z,
                            'overlay':overlay(out,v,metas[v],native,points_native,held)}
            if len(held):
                held_work=transform(held,metas[v]['native_to_work'])
                residual,_=solver.polyline_residual(held_work,uv[BOUNDARY],ordered=False)
                err=np.linalg.norm(residual,axis=1)
                residual_native,_=solver.polyline_residual(held,native[BOUNDARY],ordered=False)
                en=np.linalg.norm(residual_native,axis=1)
                evaluation_result={'schema':'controlled-missing-image-candidate-edge-evaluation-v1',
                    'evaluation_input_path':condition['evaluation_path'],'evaluation_sha256':condition['evaluation_sha256'],
                    'view_id':v,'sample_count':len(err),'rms_work_px':float(np.sqrt(np.mean(err**2))),
                    'median_work_px':float(np.median(err)),'p95_work_px':float(np.percentile(err,95)),
                    'coverage_within_original_8_work_px':float(np.mean(err<=8.)),
                    'rms_native_px':float(np.sqrt(np.mean(en**2))),'distances_work_px':err,
                    'used_for_fitting_initialization_or_selection':False,'is_natural_hidden_surface_truth':False,
                    'prediction_role':'estimated mesh boundary projection','reference_role':'withheld SAM candidate visible edge samples',
                    'requested_segment_fraction':evaluation['requested_segment_fraction'],
                    'actual_arc_length_fraction':evaluation['actual_arc_length_fraction'],
                    'min_heldout_to_retained_sample_distance_work_px':evaluation['min_heldout_to_retained_sample_distance_work_px']}
                dump(out/'controlled_missing_evaluation.json',evaluation_result)
        dump(out/'projections.json',projections)
        receipt.update(status='candidate_surface_computed',surface_computed=True,
                       optimizer_success=fitted.diagnostics['optimizer_success'],
                       candidate_internal_consistency_passed=consistency['candidate_internal_consistency_passed'],
                       final_candidate_residuals=consistency['views'],mesh_levels=[{'level':m.level,'vertices':len(m.vertices),'faces':len(m.faces),'area_source_units_squared':a} for m,a in zip(levels,areas)],
                       controlled_missing_evaluation=evaluation_result)
    except Exception as exc:
        receipt.update(status='attempt_failed_preserved',surface_computed=(out/'model_L0.json').exists(),
                       error=str(exc),traceback=traceback.format_exc())
    receipt['wall_seconds']=time.perf_counter()-started
    receipt['output_file_sha256']={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    dump(out/'execution_receipt.json',receipt)
    print(json.dumps({k:receipt.get(k) for k in ('group_id','condition','branch','status','error','wall_seconds','candidate_internal_consistency_passed')},ensure_ascii=False),flush=True)
    return receipt


def run(protocol_path, out, groups=None, conditions=None):
    protocol=read(protocol_path)
    if protocol['rules']!=RULES:
        raise ValueError('Rules differ from frozen protocol')
    for path,expected in protocol['source_files_sha256'].items():
        if sha(path)!=expected:
            raise ValueError('Frozen code/source changed: '+path)
    if protocol['runtime']['python']!=sys.version or protocol['runtime']['numpy']!=np.__version__ or protocol['runtime']['scipy']!=scipy.__version__:
        raise ValueError('Frozen numerical runtime differs')
    selected=RULES['source_groups'] if groups is None else groups
    if not set(selected)<=set(RULES['source_groups']):
        raise ValueError('No nondevelopment group allowed by this adapter')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    results=[]
    for row in protocol['development_groups']:
        cid=row['group_id']
        if cid not in selected:
            continue
        if not row['given_camera_available']:
            p=out/cid/'source_camera_unavailable.json'
            record={'group_id':cid,'status':'source_provided_camera_unavailable_not_fitted','surface_computed':False,
                    'counted_as_failure_of_geometry_solver':False,'reason':'PAA is outside the CO3D source-camera interface',
                    'protocol_sha256':sha(protocol_path)}
            if not p.exists():dump(p,record)
            results.append(record);continue
        for condition in row['conditions']:
            if conditions and condition['condition'] not in conditions:
                continue
            for branch in condition['branches']:
                dest=out/cid/condition['condition']/branch
                results.append(run_trial(protocol_path,protocol,row,condition,branch,dest))
    print(json.dumps({'invocation_completed':True,'receipt_count':len(results),
                      'candidate_surfaces_computed':sum(x.get('surface_computed',False) for x in results),
                      'verified_model_successes_claimed':0},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser()
    commands=parser.add_subparsers(dest='command',required=True)
    f=commands.add_parser('freeze');f.add_argument('--compiler',required=True);f.add_argument('--cases',nargs='+',required=True);f.add_argument('--out',required=True)
    r=commands.add_parser('run');r.add_argument('--protocol',required=True);r.add_argument('--out',required=True);r.add_argument('--groups',nargs='+');r.add_argument('--conditions',nargs='+',choices=['full','missing20','missing40','missing60'])
    args=parser.parse_args()
    if args.command=='freeze':freeze(args.compiler,args.cases,args.out)
    else:run(args.protocol,args.out,args.groups,args.conditions)


if __name__=='__main__':
    main()

