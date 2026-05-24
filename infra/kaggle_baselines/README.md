# Kaggle Dataset payload — Attendence V3 inference baselines

Tiny private Kaggle Dataset holding the committed baseline JSON that the
nightly inference kernel compares against. The actual `inference_baseline.json`
is sourced from `tests/inference_baseline.json` at workflow runtime.

## First-time upload

After replacing `KAGGLE_USERNAME` in `dataset-metadata.json` with your
actual handle, and after the first kernel run has produced a baseline:

```powershell
# Copy the freshly-committed baseline into this dir, then:
Copy-Item .\tests\inference_baseline.json .\infra\kaggle_baselines\

kaggle datasets create -p .\infra\kaggle_baselines
```

## Subsequent updates

The nightly workflow (`.github/workflows/nightly-inference.yml`) syncs the
in-repo baseline into this dataset on every run via `kaggle datasets version`.
You should not need to upload manually after the first time.

If you want to manually accept a new baseline (e.g. after a model update),
copy the relevant JSON, commit it into the repo at
`tests/inference_baseline.json`, and the next nightly will sync.
