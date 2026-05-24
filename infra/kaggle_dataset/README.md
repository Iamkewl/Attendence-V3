# Kaggle Dataset payload — Attendence V3 models

This directory is the upload payload for the **private** Kaggle Dataset
that ships the production ONNX models to the nightly inference kernel.

## What goes here

The actual `.onnx` files are gitignored (they're too large for GitHub).
Before uploading, copy them in:

```powershell
# From repo root
Copy-Item .\infra\models\yolo12n.onnx                                            .\infra\kaggle_dataset\
Copy-Item .\infra\triton\model_repository\lvface\1\model.onnx                    .\infra\kaggle_dataset\lvface_model.onnx
```

Result:
```
infra/kaggle_dataset/
|- dataset-metadata.json
|- README.md  (this file)
|- yolo12n.onnx          (~10 MB)
`- lvface_model.onnx     (~435 MB)
```

## Before the first upload

Edit `dataset-metadata.json` and replace `KAGGLE_USERNAME` with your actual
Kaggle username. Example: if your handle is `iamkewl` the `id` becomes
`iamkewl/attendence-v3-models`.

## First-time upload

```powershell
kaggle datasets create -p .\infra\kaggle_dataset --dir-mode tar
```

`--dir-mode tar` archives the directory into a single tarball before
upload, which is faster than uploading hundreds of small files. For our
case (two big files) it makes negligible difference but Kaggle recommends
it.

The upload takes 5-15 minutes depending on your connection. The
`lvface_model.onnx` (~435 MB) is the slow part.

## Updating the dataset

When you change a model file (e.g. re-export with newer ONNX opset), bump
the dataset version:

```powershell
# Replace the new file(s) in this directory, then:
kaggle datasets version -p .\infra\kaggle_dataset -m "Bump LVFace opset to 17"
```

The nightly kernel always pulls the latest version, so subsequent runs
will see the new file. The kernel will FAIL on the next run because the
model sha256 differs from the committed baseline -- this is intentional.
Inspect the run, decide whether the drift is acceptable, and if so update
`tests/inference_baseline.json` accordingly.

## Security notes

- The dataset is **private** (`isPrivate: true`). Only your Kaggle
  account and any kernels you own can read it.
- Do **not** commit any `.onnx` files into git. The repo's `.gitignore`
  already blocks `*.onnx` and `*.pt`.
- The Kaggle API token (`~/.kaggle/kaggle.json`) controls dataset
  visibility. Treat it like a password.
