# Same-leaf surface estimation with complementary views

Reviewer materials for *Estimating three-dimensional leaf surfaces under occlusion using same-leaf annotations and complementary views*.

## Reproduce the reported numbers

With Python 3, run `python verify.py` from this directory. It checks the 248-group cumulative ledger, the counts of views and leaf identities, and all 36 boundary-withholding fits. The boundary errors are recalculated from saved projected vertices and held-out pixel coordinates. The six paired leaves are a selected subset of the development collection.

For numerical fitting, install the versions in `solver/requirements.txt`. To refit one photographed leaf from its saved working-camera matrices, starting controls and retained candidate-boundary samples, run:

    python refit.py paired_inputs/G51/L1/withhold_40/two_view.json

To run the calibrated synthetic comparison, change to `solver/` and run:

    python run_example.py --output example_results

Choose a new output directory for another run. The synthetic example compares two representations using identical observations. Its archived control points are previous fitted outputs. In the photographic cases the saved cameras define a relative coordinate system.

## Directory contents

- `data/views.csv` identifies the 742 selected views. Public objects are linked to source records; image hashes distinguish native files and displayed photographs.
- `data/leaf_locators.csv` contains per-view visibility, organ IDs and locator coordinates. There are 769 within-group leaf IDs and 2,268 nonempty location clicks. Each locator identifies a leaf and can lie on a different physical point in another view.
- `data/group_outcomes.csv` contains all 248 groups and preserves failures, controls and prior outcomes. The 214 standard and 131 strict passes are cumulative across development runs. Four legacy group-to-leaf associations remain unresolved (G26, P071, P109 and P145).
- `data/last_run_groups.csv` records the last 39 candidates and four controls. `data/paired_fits.csv` records the 36 surface fits on six leaves. Nonpassing numerical outputs are retained.
- `data/calibrated_*.csv`, `data/field44_controls.csv` and the geometry tables contain the marker ablations, projection and shape comparisons. Distances use the units in each table.
- `cameras/` contains last-round camera results and, where available, frozen selected matches, train/holdout assignments and track diagnostics.
- `paired_inputs/` contains the 36 fitted control arrays, connectivity, projections, initialization, cameras, retained observations and held-out evaluation coordinates. `full_fit/` contains the available full-input last-round leaf models. These arrays distinguish estimated controls from image observations.
- `solver/` and `photo_surface_solver.py` contain fitting code. `implementation/` preserves study-run adapters for inspection. These adapters refer to the original workspace, image archives, feature-model weights and earlier runs. The runnable entry points above start from the released numerical inputs; this package does not independently reproduce all historical rounds from raw photographs.
- `checksums.sha256` lists the included files and their SHA-256 digests.

## Sources and access

The author collection includes 50 groups (153 photos in the collection, 150 selected for the extension). The other 198 sequences come from [CO3D](https://github.com/facebookresearch/co3d) (173), [PAA](https://huggingface.co/datasets/JackTrum938/PAA-Plant-Dataset/tree/19e7f2f2307462aba37b8571acc7a7c0f72a8608) (18), and [GroMo25](https://huggingface.co/datasets/MrigLabIITRopar/GroMo25/tree/5a7c4986fb3951cc54669781554b7755aac49263) (7). The calibrated leaf study used [KOMATSUNA](https://limu.ait.kyushu-u.ac.jp/~agri/komatsuna/). The public-image archives and weights remain with their providers and their access conditions apply. This directory distributes coordinates, outputs, and code, not the original photograph collection. The [earlier release](https://github.com/pennchenj-lang/POTS-Research-Data/tree/b7d10e5c54595eef08609644b94b4e7e1c5ed88c) contains additional calibrated and synthetic inputs, references and model files.

Author: JianPeng Chen.

