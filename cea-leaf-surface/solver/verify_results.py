"""Check source/input integrity and compare completed fits to frozen outputs."""
from pathlib import Path
import argparse, hashlib, json, sys
sys.dont_write_bytecode = True
import numpy as np
ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def verify_manifest():
    manifest = read(ROOT / 'MANIFEST_SHA256.json')
    for relative, expected in manifest['files_sha256'].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Manifest path escapes package')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Changed packaged source or input: ' + relative)
    return manifest

def verify_results(output):
    verify_manifest()
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Results must be inside the package')
    expectations = read(ROOT / 'data/expected_metrics.json')
    comparisons = []
    # This file holds previous solver outputs, not target geometry.
    with np.load(ROOT / 'data/expected_controls.npz', allow_pickle=False) as expected:
        for method in ('POTS33', 'symmetric24'):
            geometry = read(output / method / 'L0.json')
            xyz = np.asarray([v['xyz'] for v in geometry['vertices']], float)
            delta = xyz - expected[method]
            controls_pass = bool(np.allclose(xyz, expected[method],
                atol=expectations['control_atol'], rtol=expectations['control_rtol']))
            actual = read(output / method / 'metrics.json')
            frozen = expectations['methods'][method]
            field_checks = {}
            for field in ('parameters', 'optimizer_success', 'all_positive_depth', 'level_vertex_face_counts'):
                field_checks[field] = actual[field] == frozen[field]
            for field in ('final_residual_norm', 'areas_L0_L1_L2'):
                field_checks[field] = bool(np.allclose(actual[field], frozen[field],
                    atol=expectations['metric_atol'], rtol=expectations['metric_rtol']))
            for a, b in zip(actual['training_role_RMS_px'], frozen['training_role_RMS_px']):
                if a['view_id'] != b['view_id']:
                    raise ValueError('Training-view order changed')
                for role in ('material_rms_px', 'Q_rms_px', 'W_rms_px'):
                    field_checks[a['view_id'] + ':' + role] = bool(np.isclose(a[role], b[role],
                        atol=expectations['metric_atol'], rtol=expectations['metric_rtol']))
            comparisons.append({'method': method, 'controls_match': controls_pass,
                'max_abs_control_difference': float(np.max(np.abs(delta))),
                'control_coordinate_RMS_difference': float(np.sqrt(np.mean(delta * delta))),
                'metric_checks': field_checks, 'passed': controls_pass and all(field_checks.values())})
    return {'schema': 'pots.algorithm-example-verification.v1',
        'passed': all(row['passed'] for row in comparisons), 'comparisons': comparisons,
        'scope': 'Frozen algorithm-output regression only; not independent reconstruction accuracy.',
        'control_atol': expectations['control_atol'], 'control_rtol': expectations['control_rtol'],
        'metric_atol': expectations['metric_atol'], 'metric_rtol': expectations['metric_rtol']}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='results')
    args = parser.parse_args()
    report = verify_results(ROOT / args.results)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['passed'] else 1)
