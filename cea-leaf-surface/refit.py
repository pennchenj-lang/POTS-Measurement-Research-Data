"""Refit one leaf with the saved cameras, initial controls and boundary observations."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'solver'))
from pots_gs.triangulation import Camera
import photo_surface_solver as solver

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',type=Path)
    args=parser.parse_args()
    case=json.loads(args.case.read_text(encoding='utf8'))
    cameras={v:Camera(v,c['K'],c['R'],c['t'],unit='relative') for v,c in case['cameras'].items()}
    obs=case['prepared']
    for o in obs:
        for field in ('Q','W'):o[field]=np.asarray(o[field],float).reshape(-1,2)
    active={o['view_id']:cameras[o['view_id']] for o in obs}
    solver.prepare_observations=lambda *args,**kw:(obs,[],['Saved candidate-frontier observations.'])
    result=solver.fit_scaffold(case['initial_xyz'],active,[],organ_id='leaf',data_type='real',
        allow_unreviewed_diagnostic=True, allow_single_view_diagnostic=len(active)==1,
        regularization=.2,max_nfev=250,sigma_px=2.,differential_weight=1.,max_displacement=1.)
    delta=float(np.max(np.abs(np.asarray(case['fitted_xyz'])-result.xyz)))
    print('Maximum absolute coordinate difference from the archived fit:',delta)
    if delta>1e-4:raise SystemExit('The fit differs from the archived solution.')
if __name__=='__main__':main()

