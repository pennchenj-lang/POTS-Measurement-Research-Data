# POTS synthetic algorithm example

This small offline example fits the same calibrated, synthetic leaf input with
two frozen representations: POTS33 (11 free XYZ controls) and symmetric24
(24 independent displacements with each middle station tied to its two margins).
Both use the same B/T anchors, view-local Q paths, true W margins, initialization,
loss normalization, priors, and solver settings. No Gaussian splatting is run.

Tested runtime: Python 3.12.2, NumPy 2.5.3, SciPy 1.18.1 on Windows x64.
The exact dependency pins are in `requirements.txt`. If these dependencies are
already installed, the example runs offline and does not download anything.

From this directory:

```text
python run_example.py --output results
python verify_results.py --results results
```

The first command fits both methods, exports JSON/OBJ meshes at L0/L1/L2, reports
training residuals, and verifies the completed outputs. Use a new output name
for another run; existing results are never overwritten. The verifier checks
the package manifest before running. Minor cross-platform numerical differences
are allowed only within the fixed tolerances in `data/expected_metrics.json`.
A failing comparison is reported; the program does not retune or silently retry.

The single bundled case is `curved_common_gap`. Its two calibrated synthetic
views have complementary side occlusion plus a shared image band from y=190 to
280 pixels. B/T remain visible. Q samples preserve within-view order; all Q
intervals are unknown and are not interpolated. Only explicitly confirmed W
margin segments constrain the boundary. Q/W indices are not triangulated as
cross-view physical correspondences. Initial widths and orientation are priors.

L0/L1/L2 contain (11,12), (33,48), and (113,192) vertices/faces. Shared-edge
midpoints preserve the same piecewise-planar surface; subdivision alone does not
add observations or improve geometry. Organ identity and lineage remain in JSON.

`data/expected_controls.npz` contains previously saved solver outputs, not a
reference plant surface. It is loaded only after both fits for regression checks.
The package intentionally contains no target XYZ, held-out camera, real image,
manuscript, research log, weights, or private credentials. Agreement with frozen
outputs establishes computational reproducibility of this example, not physical
accuracy, robust occlusion completion, or automatic annotation capability.

The 24-variable model is a controlled local comparison, not a reproduction of an
external published method. Source snapshots remain byte-identical to their
frozen versions. This local package does not choose an open-source license and
does not publish or upload itself.
