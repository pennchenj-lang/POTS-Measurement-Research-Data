"""One offline command: fit the two frozen representations and verify outputs."""
from pathlib import Path
import argparse, copy, importlib, json, os, sys, time
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
cache = ROOT / '.cache'
cache.mkdir(exist_ok=True)
for key in ('TEMP', 'TMP', 'TMPDIR'):
    os.environ[key] = str(cache)

import numpy as np
import scipy
from pots_gs.triangulation import Camera
from pots_gs.geometry import mesh_area
from pots_gs.subdivision import subdivision_levels
from pots_gs import multiview_scaffold, symmetric_ribbon
from verify_results import verify_manifest, verify_results

def write_new(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)

def run(output):
    verify_manifest()
    output = (ROOT / output).resolve()
    if not output.is_relative_to(ROOT) or output == ROOT:
        raise ValueError('Output must be a new subdirectory inside this package')
    if output.exists():
        raise FileExistsError('Preserve previous outputs; choose a new --output directory')
    data = json.loads((ROOT / 'data/curved_common_gap.json').read_text(encoding='utf-8'))
    if data['data_type'] != 'synthetic':
        raise ValueError('This small example accepts only its bundled synthetic input')
    cameras = {name: Camera(view_id=name, **camera) for name, camera in data['cameras'].items()}
    initial = np.asarray(data['initial_xyz'], float)
    loaded = {}
    for name in ('pots_gs', 'pots_gs.geometry', 'pots_gs.subdivision', 'pots_gs.triangulation',
                 'pots_gs.multiview_scaffold', 'pots_gs.symmetric_ribbon'):
        location = Path(importlib.import_module(name).__file__).resolve()
        if not location.is_relative_to(ROOT):
            raise RuntimeError('An algorithm module was loaded from outside this package: ' + name)
        loaded[name] = str(location.relative_to(ROOT))
    output.mkdir()
    records = []
    # Expected coordinates are deliberately not loaded by the fitting loop.
    # No reference surface exists in the input or in the fitter API.
    for name, module in (('POTS33', multiview_scaffold), ('symmetric24', symmetric_ribbon)):
        destination = output / name
        destination.mkdir()
        started = time.perf_counter()
        fit = module.fit_scaffold(initial.copy(), cameras, copy.deepcopy(data['observations']),
                                  organ_id='leaf01', data_type='synthetic', **data['solver'])
        elapsed = time.perf_counter() - started
        levels = subdivision_levels(fit.mesh, 2)
        for mesh in levels:
            mesh.write_json(destination / f'L{mesh.level}.json')
            mesh.write_obj(destination / f'L{mesh.level}.obj')
        selected = {k: fit.diagnostics[k] for k in ('parameters', 'optimizer_success', 'optimizer_status',
            'nfev', 'final_residual_norm', 'all_positive_depth')}
        selected['training_role_RMS_px'] = [{key: row[key] for key in
            ('view_id', 'material_rms_px', 'Q_rms_px', 'W_rms_px')} for row in fit.diagnostics['final']]
        selected['areas_L0_L1_L2'] = [mesh_area(mesh) for mesh in levels]
        selected['level_vertex_face_counts'] = [[len(mesh.vertices), len(mesh.faces)] for mesh in levels]
        selected['elapsed_s'] = elapsed
        write_new(destination / 'metrics.json', selected)
        records.append({'method': name, **selected})
        print(name, 'generated', selected['level_vertex_face_counts'], flush=True)
    write_new(output / 'execution.json', {'data_type': 'synthetic', 'case': data['case_id'],
        'python_version': sys.version, 'numpy_version': np.__version__, 'scipy_version': scipy.__version__,
        'package_module_paths': loaded, 'references_passed_to_fit': False, 'GS_used': False,
        'methods': records})
    report = verify_results(output)
    write_new(output / 'verification.json', report)
    print('Verification:', 'PASS' if report['passed'] else 'FAIL', flush=True)
    if not report['passed']:
        raise RuntimeError('Frozen output comparison failed; inspect verification.json')
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results', help='New relative directory inside the package')
    run(parser.parse_args().output)
