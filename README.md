# POTS research data — public reviewer subset

Creator: JianPeng Chen. Version: v041-public-reviewer.

This release provides 167 selected evidence files supporting the v041 manuscript and Supplementary Information S1–S10. It contains numerical estimates, failures, descriptive statistics, 36 frozen L0/L2 surfaces, synthetic inputs and evaluation references, and the researcher's scientific annotation/calibration records. Publication was authorized by the author. It contains no manuscript, title page, correspondence or raw third-party raster asset.

## Reviewer scope

The seven source_data CSV tables preserve every original numeric cell. The saved comparison uses four synthetic targets, six synthetic conditions and three real leaves from three plants, with POTS33 and the symmetric24 comparator. Results are frozen; no new fitting or physical experiment was performed for this release. Synthetic references are numerical constructions, not independently measured physical leaves. Failures, flagged geometry, input rejection, unknown scale and unavailable-reference states remain explicit.

The real data have unbound physical scale (unknown_checkerboard_square), no eligible independent same-leaf same-state physical area or midrib reference, and no independent repeated acquisition/annotation records. Model lengths and areas must not be relabelled millimetres or square millimetres. This package does not establish physical accuracy. The nine planned_records CSV files are EMPTY prospective forms, not completed observations.

## Files and navigation

- DATA_INDEX.csv lists only included evidence files, their paper uses and source/local/public hashes.
- DATA_DICTIONARY.md and CSV_SCHEMAS.json explain quantities, units, table columns, missingness and sample structure.
- FILE_MANIFEST.csv verifies every included file except itself.
- SOURCE_RETRIEVAL.csv identifies the exact 18 omitted RGB/instance-label source files by official archive URL, member name, acquisition IDs and SHA256. SOURCE_RETRIEVAL.md explains acquisition and calibration mapping.
- EXCLUDED_FILES.csv records every excluded item. No excluded item is claimed to exist in this release.
- EXTERNAL_REFERENCES.csv describes unbundled historical records; source_archive/ identifiers do not resolve within this package.

The inputs/real/annotation and camera records contain researcher-created scientific coordinates, fitted calibration and factual source/crop metadata. They are separate from the provider's instance-label raster images. Existing human/automatic/interpolated/review states remain unchanged and are not certification of new manual work. The combined review export retains p03 rejection records; p03 is not a fitted leaf. Evaluation-only records must not be used for fitting.

## Excluded materials and paths

All 28 local PNG files, provider PDF/HTML documentation, the local source-verification note and broad historical manifests are omitted. This includes optional projected raster previews as well as restricted source/label rasters. Third-party images must be obtained from the original provider under its access terms; their redistribution licence was not confirmed.

Included file references use package-root-relative paths. References to omitted local items are explicitly prefixed excluded_materials/; no such directory is included. References to other original-archive records use source_archive/. Bare filenames and original archive-member identifiers retain their original context, rather than being advertised as package files. Embedded source hashes refer to original bytes; DATA_INDEX.csv separately records the public exported hashes.

The release supports examination of the supplied numerical results. Image-level reruns require obtaining the listed source images. Historical executable paths are provenance, not a complete software distribution. No comprehensive reproducibility claim is implied.

## Source attribution

KOMATSUNA multiview dataset: https://limu.ait.kyushu-u.ac.jp/~agri/komatsuna/
Uchiyama et al., An Easy-to-Setup 3D Phenotyping Platform for KOMATSUNA Dataset, ICCVW (2017), 2038–2045. https://doi.org/10.1109/ICCVW.2017.239
The RGB-D subset depicts different plants and must not be substituted as matched physical reference data. See LICENCE_AND_ACCESS.md for the release boundary.
