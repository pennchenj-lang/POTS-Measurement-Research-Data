"""Same-organ calibrated path/boundary fitting for the POTS geometry core.

This module estimates ONE fixed-topology leaf from training observations. It
does not implement Gaussian splatting, automatic annotation, amodal masks, or
recover unique shape from a single view. Camera pixels must be undistorted.
Q labels are per-view ordered path samples; W labels are per-view margin
samples. Neither is ever triangulated by matching its index across views.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import numpy as np

from .geometry import LEAF_FACES, LEAF_LABELS, Provenance, create_leaf_mesh
from .triangulation import Camera

MIDRIB = np.array([0, 2, 5, 8, 10])
BOUNDARY = np.array([0, 1, 4, 7, 10, 9, 6, 3, 0])
EDGES = np.array(sorted({tuple(sorted((a, b))) for f in LEAF_FACES
                        for a, b in zip(f, (*f[1:], f[0]))}), dtype=int)


@dataclass
class ScaffoldFit:
    xyz: np.ndarray
    mesh: object
    diagnostics: dict


def project_xyz(camera: Camera, xyz):
    """Project world points; clipping is only a numerical guard, not acceptance."""
    xyz = np.asarray(xyz, float)
    c = xyz @ camera.R.T + camera.t
    depth = c[:, 2]
    safe = np.where(np.abs(depth) < 1e-9, np.where(depth < 0, -1e-9, 1e-9), depth)
    homogeneous = c @ camera.K.T
    return homogeneous[:, :2] / safe[:, None], depth


def _at_parameter(polyline, parameter):
    nseg = len(polyline) - 1
    segment = np.minimum(np.floor(parameter).astype(int), nseg - 1)
    fraction = parameter - segment
    return polyline[segment] + fraction[:, None] * (polyline[segment + 1] - polyline[segment])


def polyline_residual(points, polyline, *, ordered=False):
    """Return point-minus-curve residuals and within-view segment parameters.

    W uses exact point-to-segment distance. Q uses nondecreasing dynamic
    programming over a lattice containing all point-to-segment projections
    and 65 uniformly spaced curve parameters. It is a finite-candidate ordered
    approximation, not a proof of the continuous global curve-matching optimum.
    Equal parameters are allowed; there is no artificial equal-spacing prior.
    """
    points, curve = np.asarray(points, float), np.asarray(polyline, float)
    if points.ndim != 2 or points.shape[1] != 2 or curve.ndim != 2 or curve.shape[1] != 2 or len(curve) < 2:
        raise ValueError("points and polyline must be N x 2 and M x 2")
    if not len(points):
        return np.empty((0, 2)), np.empty(0)
    a, v = curve[:-1], np.diff(curve, axis=0)
    fraction = np.clip(np.einsum('nsi,si->ns', points[:, None] - a, v) /
                       np.maximum(np.sum(v * v, axis=1), 1e-20), 0, 1)
    parameter = np.arange(len(v))[None] + fraction
    if not ordered:
        candidates = a[None] + fraction[..., None] * v[None]
        cost = np.sum((points[:, None] - candidates) ** 2, axis=2)
        selected = np.argmin(cost, axis=1)
        return points - candidates[np.arange(len(points)), selected], parameter[np.arange(len(points)), selected]
    lattice = np.unique(np.r_[np.linspace(0, len(v), 65), parameter.ravel()])
    candidates = _at_parameter(curve, lattice)
    cost = np.sum((points[:, None] - candidates[None]) ** 2, axis=2)
    previous = cost[0].copy()
    back = []
    for row in cost[1:]:
        minimum = np.minimum.accumulate(previous)
        # First index attaining each prefix minimum gives deterministic ties.
        idx = np.arange(len(lattice))
        winners = np.maximum.accumulate(np.where(previous <= minimum, idx, 0))
        back.append(winners)
        previous = row + minimum
    chosen = [int(np.argmin(previous))]
    for parents in reversed(back):
        chosen.append(int(parents[chosen[-1]]))
    chosen = np.array(chosen[::-1])
    return points - candidates[chosen], lattice[chosen]


def _path_points(path, role, *, allow_unreviewed, real, diagnostic_reasons):
    xy = np.asarray(path.get('xy', []), float)
    if xy.size == 0:
        return np.empty((0, 2)), {'points': 0, 'visible_segments': 0, 'ignored_segments': 0}
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError(f'{role} xy must be finite N x 2')
    if not path.get('reviewed', False):
        if real and not allow_unreviewed:
            raise ValueError(f'{role} annotations require review')
        diagnostic_reasons.append(f'{role} path not reviewed')
    states = path.get('point_status', ['visible'] * len(xy))
    if len(states) != len(xy) or any(s not in {'visible', 'occluded', 'unknown'} for s in states):
        raise ValueError(f'{role} point_status is invalid')
    if path.get('closed', False):
        raise ValueError('Open observation paths must not be automatically closed; supply an explicit last edge')
    key = 'segment_visibility' if role == 'Q' else 'segment_kind'
    edges = path.get(key, ['unknown'] * max(0, len(xy) - 1))
    allowed = {'visible', 'occluded', 'unknown'} if role == 'Q' else {'true_margin', 'occluder_cut', 'unknown'}
    if len(edges) != max(0, len(xy) - 1) or any(s not in allowed for s in edges):
        raise ValueError(f'{role} {key} is invalid')
    if role == 'Q' and path.get('order') != 'B_to_T':
        raise ValueError('Q order must explicitly be B_to_T')
    visible_key = 'visible' if role == 'Q' else 'true_margin'
    out, used = [], 0
    if role == 'Q':
        # Each visible Q click is evidence even when the adjoining curve is unknown.
        for i, point in enumerate(xy):
            if states[i] == 'visible':
                out.append(point)
            if i < len(edges) and edges[i] == visible_key:
                if states[i] != 'visible' or states[i + 1] != 'visible':
                    raise ValueError('An observed path segment requires two visible endpoints')
                out.extend(point + t * (xy[i + 1] - point) for t in (.25, .5, .75))
                used += 1
    else:
        # No inference from a W index or an untyped point: only declared true
        # margin segments contribute. Occluder cuts never constrain leaf shape.
        for i, kind in enumerate(edges):
            if kind == visible_key:
                if states[i] != 'visible' or states[i + 1] != 'visible':
                    raise ValueError('A true observed margin requires two visible endpoints')
                out.extend(xy[i] + t * (xy[i + 1] - xy[i]) for t in (0., .25, .5, .75, 1.))
                used += 1
    points = np.asarray(out, float).reshape(-1, 2)
    return points, {'points': len(points), 'visible_segments': used, 'ignored_segments': len(edges) - used}


def prepare_observations(cameras, observations, *, organ_id, data_type,
                         allow_unreviewed_diagnostic=False, allow_single_view_diagnostic=False):
    """Validate and compile training-only evidence, preserving original records."""
    if data_type not in {'synthetic', 'real'} or not organ_id:
        raise ValueError('Explicit real/synthetic data_type and organ_id are required')
    cameras = dict(cameras)
    if any(not isinstance(c, Camera) or c.view_id != k for k, c in cameras.items()):
        raise ValueError('cameras must map view IDs to matching Camera objects')
    if len({c.unit for c in cameras.values()}) != 1:
        raise ValueError('camera units must agree')
    prepared, reasons, audit = [], [], []
    seen, material_ids = set(), {}
    for record in observations:
        view = record['view_id']
        if view in seen or view not in cameras:
            raise ValueError('Each training view must appear exactly once and have a camera')
        seen.add(view)
        if record.get('split') != 'train':
            raise ValueError('Evaluation/held-out observations must not enter fitting')
        if record.get('organ_id') != organ_id:
            raise ValueError('All observations must refer to the same organ_id')
        if record.get('coordinate_domain') != 'undistorted_pixels':
            raise ValueError('Camera observations must be in undistorted_pixels domain')
        if data_type == 'real' and (not record.get('reviewed', False) or not record.get('same_leaf_confirmed', False)):
            if not allow_unreviewed_diagnostic:
                raise ValueError('Real fitting requires same-leaf confirmation and reviewed annotations')
            reasons.append(f'{view}: missing real human review')
        materials = []
        for point in record.get('material', []):
            if point.get('status') != 'visible':
                continue
            label, identity = point.get('label'), point.get('correspondence_id')
            if label not in {'B', 'T'} or not identity:
                raise ValueError('Only explicit B/T physical correspondence IDs are accepted as material anchors')
            if label in material_ids and material_ids[label] != identity:
                raise ValueError('Material point correspondence IDs disagree across views')
            material_ids[label] = identity
            if not point.get('reviewed', False):
                if data_type == 'real' and not allow_unreviewed_diagnostic:
                    raise ValueError('Real material correspondence requires review')
                reasons.append(f'{view}: {label} not reviewed')
            xy = np.asarray(point['xy'], float)
            if xy.shape != (2,) or not np.isfinite(xy).all():
                raise ValueError('Material xy must be a finite pixel pair')
            materials.append((0 if label == 'B' else 10, xy, identity))
        q, qaudit = _path_points(record.get('Q', {}), 'Q', allow_unreviewed=allow_unreviewed_diagnostic,
                                real=data_type == 'real', diagnostic_reasons=reasons)
        wpoints, waudit = [], []
        for path in record.get('W', []):
            points, item = _path_points(path, 'W', allow_unreviewed=allow_unreviewed_diagnostic,
                                       real=data_type == 'real', diagnostic_reasons=reasons)
            if len(points):
                wpoints.extend(points)
            waudit.append(item)
        w = np.asarray(wpoints, float).reshape(-1, 2)
        if not materials and not len(q) and not len(w):
            raise ValueError(f'{view}: no usable observation')
        prepared.append({'view_id': view, 'material': materials, 'Q': q, 'W': w})
        audit.append({'view_id': view, 'material_points': len(materials), 'Q': qaudit, 'W': waudit})
    if len(prepared) < 2:
        if not allow_single_view_diagnostic or not prepared:
            raise ValueError('At least two training views are required')
        reasons.append('single-view diagnostic; depth not determined')
    if set(cameras) != seen:
        raise ValueError('Supply only training cameras; unused or held-out cameras are not accepted')
    return prepared, audit, sorted(set(reasons))


def fit_scaffold(initial_xyz, cameras, observations, *, organ_id, data_type,
                 regularization=.2, max_nfev=250, sigma_px=2.,
                 differential_weight=1., max_displacement=1.,
                 allow_unreviewed_diagnostic=False, allow_single_view_diagnostic=False):
    """Fit 11 control XYZs using material, ordered Q, true W and frozen priors.

    The initial shape/position prior and differential edge prior are normalized
    by the initial bounding-box diagonal. A componentwise displacement box is
    also relative to this length. The prior is part of the inference, not data.
    Per-view/role residual normalization avoids equating dense W annotation
    with independent extra views. No image or hidden reference is accepted.
    The result is a regularized estimate, with no uniqueness/accuracy guarantee.
    """
    from scipy.optimize import least_squares
    xyz0 = np.asarray(initial_xyz, float)
    if xyz0.shape != (11, 3) or not np.isfinite(xyz0).all():
        raise ValueError('initial_xyz must be finite 11 x 3')
    if not (regularization >= 0 and differential_weight >= 0 and sigma_px > 0 and max_displacement > 0 and max_nfev > 0):
        raise ValueError('Invalid solver parameters')
    prepared, audit, reasons = prepare_observations(cameras, observations, organ_id=organ_id, data_type=data_type,
        allow_unreviewed_diagnostic=allow_unreviewed_diagnostic, allow_single_view_diagnostic=allow_single_view_diagnostic)
    scale = float(np.linalg.norm(np.ptp(xyz0, axis=0)))
    if scale <= 1e-12:
        raise ValueError('Initial leaf is degenerate')
    # Validate the actual taskbook mesh before optimizer evaluation.
    source_views = tuple(sorted(cameras))
    initial_origin = [Provenance('estimated', source_views, (), 'input initialization; not an observed XYZ')] * 11
    create_leaf_mesh(xyz0, organ_id, data_type=data_type, provenance=initial_origin, confidence=[0.] * 11)
    if any(np.any(project_xyz(c, xyz0)[1] <= scale * 1e-6) for c in cameras.values()):
        raise ValueError('Initial mesh must have positive camera depth')
    edge0 = xyz0[EDGES[:, 1]] - xyz0[EDGES[:, 0]]

    def evidence(xyz):
        blocks, metrics = [], []
        for item in prepared:
            view = item['view_id']
            projected, depths = project_xyz(cameras[view], xyz)
            roles = {}
            if item['material']:
                r = np.array([projected[i] - xy for i, xy, _ in item['material']])
                blocks.append(r.ravel() / sigma_px / np.sqrt(len(r)))
                roles['material_rms_px'] = float(np.sqrt(np.mean(np.sum(r * r, axis=1))))
            if len(item['Q']):
                r, assignment = polyline_residual(item['Q'], projected[MIDRIB], ordered=True)
                blocks.append(r.ravel() / sigma_px / np.sqrt(len(r)))
                roles['Q_rms_px'] = float(np.sqrt(np.mean(np.sum(r * r, axis=1))))
                roles['Q_within_view_parameters'] = assignment.tolist()
            if len(item['W']):
                r, _ = polyline_residual(item['W'], projected[BOUNDARY], ordered=False)
                blocks.append(r.ravel() / sigma_px / np.sqrt(len(r)))
                roles['W_rms_px'] = float(np.sqrt(np.mean(np.sum(r * r, axis=1))))
            blocks.append(10 * np.minimum(depths / scale - 1e-6, 0))
            metrics.append({'view_id': view, **roles, 'min_camera_depth': float(depths.min())})
        return blocks, metrics

    def residual(delta):
        xyz = xyz0 + delta.reshape(11, 3) * scale
        blocks, _ = evidence(xyz)
        blocks.append(np.sqrt(regularization) * delta / np.sqrt(11))
        edges = xyz[EDGES[:, 1]] - xyz[EDGES[:, 0]]
        blocks.append(np.sqrt(regularization * differential_weight) * ((edges - edge0) / scale).ravel() / np.sqrt(len(EDGES)))
        return np.concatenate(blocks)

    fit = least_squares(residual, np.zeros(33), bounds=(-max_displacement, max_displacement),
                        max_nfev=int(max_nfev), method='trf', ftol=1e-8, xtol=1e-8, gtol=1e-8)
    xyz = xyz0 + fit.x.reshape(11, 3) * scale
    _, before = evidence(xyz0)
    _, after = evidence(xyz)
    origin = [Provenance('estimated', source_views, tuple(f'{v}:training_annotation' for v in source_views),
              'Regularized multi-view estimate; Q/W are view-dependent path/margin samples, not triangulated same-index points') for _ in LEAF_LABELS]
    mesh = create_leaf_mesh(xyz, organ_id, data_type=data_type, provenance=origin, confidence=[0.] * 11)
    mesh.coordinate_system = f'world coordinates of supplied calibrated cameras; unit={next(iter(cameras.values())).unit}'
    a = xyz[np.asarray(LEAF_FACES)]
    areas = np.linalg.norm(np.cross(a[:, 1] - a[:, 0], a[:, 2] - a[:, 0]), axis=1) * .5
    diagnostics = {
        'schema': 'pots-gs.multiview-scaffold-fit.v1', 'data_type': data_type, 'organ_id': organ_id,
        'status': 'unreviewed_diagnostic' if reasons else 'reviewed_input_fit',
        'review_warnings': reasons, 'optimizer_success': bool(fit.success), 'optimizer_status': int(fit.status),
        'optimizer_message': str(fit.message), 'nfev': int(fit.nfev), 'training_views': list(source_views),
        'evaluation_used': False, 'gs_optimized': False, 'parameters': 33,
        'regularization': float(regularization), 'differential_weight': float(differential_weight),
        'initial_scale': scale, 'sigma_px': float(sigma_px), 'max_displacement_in_initial_scale': float(max_displacement),
        'initial_residual_norm': float(np.linalg.norm(residual(np.zeros(33)))),
        'final_residual_norm': float(np.linalg.norm(fit.fun)),
        'initial': before, 'final': after, 'observation_audit': audit,
        'max_vertex_displacement': float(np.linalg.norm(xyz - xyz0, axis=1).max()),
        'min_triangle_area': float(areas.min()),
        'all_positive_depth': all(m['min_camera_depth'] > 0 for m in after),
        'box_boundary_hits': int(np.count_nonzero(np.isclose(np.abs(fit.x), max_displacement, atol=1e-5))),
        'confidence_semantics': 'all vertex confidence=0: uncertainty not calibrated',
        'Q_assignment': 'within-view monotone finite-candidate curve match, no cross-view index correspondence',
        'W_assignment': 'one-way true-margin sample to projected organ boundary; no occluder cut or auto-closing',
        'limitations': ['11 controls approximate the leaf; regularization contributes shape',
                       'visible path/boundary fit does not prove unique 3D depth or unseen curvature',
                       'no whole-plant occlusion renderer; input visibility is supplied',
                       'positive depth and nondegenerate triangles do not prove no self-intersection',
                       'optimizer convergence and box non-contact do not prove reconstruction accuracy']}
    mesh.metadata.update({'algorithm': 'POTS multi-view scaffold fitting', 'geometry_optimized': True,
                          'fit_diagnostics': diagnostics, 'training_observations': copy.deepcopy(observations),
                          'Q_W_indices_are_material_correspondences': False,
                          'observation_source_views_are_dependencies': True, 'gs_optimized': False})
    return ScaffoldFit(xyz, mesh, diagnostics)
