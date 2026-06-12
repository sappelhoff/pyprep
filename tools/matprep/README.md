# MATLAB PREP artifacts

Scripts that run the full **MATLAB PREP** pipeline on a test EEG recording and
save a `.set`/`.mat` artifact at each stage. PyPREP's test suite compares its own
output against these artifacts to verify that the two implementations are
numerically equivalent when `matlab_strict=True`
(see [`tests/test_matprep_compare.py`](../../tests/test_matprep_compare.py)).

These scripts originated in the standalone
[`a-hurst/matprep_artifacts`](https://github.com/a-hurst/matprep_artifacts)
repository (by Austin Hurst) and were vendored here in
[#160](https://github.com/sappelhoff/pyprep/issues/160) so PyPREP owns the full
artifact-generation pipeline.

## How it is used

The [`matprep_artifacts`](../../.github/workflows/matprep_artifacts.yml) GitHub
Actions workflow runs these scripts on a MATLAB-equipped runner and publishes the
resulting artifacts to a rolling pre-release tagged **`matprep-artifacts`**. The
test suite downloads the artifacts from that release, so the workflow is the
single source of truth — nothing is committed to the repository.

To regenerate the artifacts (e.g. after bumping a pinned version), trigger the
workflow manually (`workflow_dispatch`); a push that touches `tools/matprep/**`
also triggers it.

## Reproducibility / pinned versions

The artifacts are PyPREP's numeric ground truth, so the algorithm-defining
dependencies are pinned in [`config/pipeline_setup.py`](config/pipeline_setup.py):

- **EEGLAB** — pinned to a specific daily build.
- **MATLAB PREP** (EEG-Clean-Tools) — pinned to a specific commit, not a branch.

The MATLAB runtime itself is left at `latest` in the workflow: the numeric
behaviour is governed by the pinned EEGLAB + PREP code, and pinning the runtime to
a release that may not be provisioned on the runner would add fragility for little
gain. Bumping any pin is a deliberate change — do it in its own PR and regenerate
the artifacts so any shift in the reference output is reviewed.

## Artifacts produced

| Name | Description |
| --- | --- |
| `1_matprep_raw.set` | The montage-fitted test data prior to running PREP. |
| `2_matprep_removetrend.set` | The test data following pre-CleanLine trend removal. |
| `3_matprep_cleanline.set` | The test data following adaptive line-noise removal. |
| `4_matprep_pre_reference.set` | The test data following pre-reference trend removal. |
| `5_matprep_post_reference.set` | The test data following robust re-referencing & interpolation. |
| `matprep_info.mat` | Detailed info about noisy channels during re-referencing. |

## Files

- [`generate_artifacts.m`](generate_artifacts.m) — the pipeline driver run by MATLAB.
- [`config/settings.m`](config/settings.m) — paths, PREP parameters, and which
  channels are made bad-by-NaN/flat/dropout.
- [`config/pipeline_setup.py`](config/pipeline_setup.py) — downloads the test data,
  EEGLAB, and MATLAB PREP before the MATLAB step runs.
