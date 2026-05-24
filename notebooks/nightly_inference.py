"""Nightly inference regression check for Attendence-V3.

Runs on Kaggle (GPU notebook). Loads YOLOv12 + LVFace ONNX models from a
Kaggle Dataset, runs them on deterministic synthetic inputs, hashes the
output tensors, and compares to a committed baseline. Emits a JSON report
to /kaggle/working/inference_current.json with overall PASS / FAIL status.

The compare path is deliberately layered:

    1. Model file sha256: any change to the .onnx bytes is loud-failure.
    2. Output tensor sha256: bit-identical match  -> PASS (clean).
    3. Output stats drift: min/max/mean/std within MAX_ABS_DIFF_THRESHOLD
       of baseline -> PASS (acceptable float noise from CUDA non-det).
    4. Otherwise -> FAIL with details.

When no baseline is mounted, the run still emits a report (status =
BASELINE_GENERATED) which a human commits as the new baseline.
"""

import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Paths (Kaggle conventions). The "/kaggle/input/<slug>/" directories are
# mounted automatically when a Dataset is attached to the kernel.
# ---------------------------------------------------------------------------

MODELS_DIR = Path("/kaggle/input/attendence-v3-models")
BASELINES_DIR = Path("/kaggle/input/attendence-v3-baselines")
WORKING_DIR = Path("/kaggle/working")

YOLO_MODEL_PATH = MODELS_DIR / "yolo12n.onnx"
LVFACE_MODEL_PATH = MODELS_DIR / "lvface_model.onnx"
BASELINE_PATH = BASELINES_DIR / "inference_baseline.json"
OUTPUT_PATH = WORKING_DIR / "inference_current.json"

# ---------------------------------------------------------------------------
# Test inputs (deterministic, seeded). NCHW float32 in [0, 1].
# ---------------------------------------------------------------------------

SEED = 42
NUM_YOLO_FRAMES = 2
NUM_LVFACE_FRAMES = 4

YOLO_INPUT_SHAPE = (1, 3, 640, 640)   # standard YOLO input size
LVFACE_INPUT_SHAPE = (1, 3, 112, 112) # fixed per LVFace config.pbtxt

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

# Any per-stat drift below this is treated as acceptable float noise.
# CUDA reductions are not bit-deterministic across driver/cuDNN versions,
# but real model drift would move stats by orders of magnitude more.
MAX_ABS_DIFF_THRESHOLD = 1e-4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_bytes(buf: bytes) -> str:
    h = hashlib.sha256()
    h.update(buf)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize_array(arr: np.ndarray) -> dict:
    """Compact, comparable summary of an ndarray. Captures everything we need
    to do both bit-exact (sha256) and within-tolerance (stats) comparison."""
    flat = arr.flatten()
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": sha256_bytes(arr.tobytes()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "first_16": [float(x) for x in flat[:16]],
        "last_16": [float(x) for x in flat[-16:]],
    }


def now_iso() -> str:
    return datetime.datetime.now(tz=datetime.timezone.utc).isoformat()


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Runtime setup
# ---------------------------------------------------------------------------

log("== Attendence-V3 nightly inference ==")
log(f"timestamp: {now_iso()}")
log(f"onnxruntime version: {ort.__version__}")

providers_available = ort.get_available_providers()
log(f"Available providers: {providers_available}")

if "CUDAExecutionProvider" in providers_available:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
elif "TensorrtExecutionProvider" in providers_available:
    providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
else:
    providers = ["CPUExecutionProvider"]
log(f"Using providers: {providers}")

session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL


# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------

if not YOLO_MODEL_PATH.exists():
    log(f"FATAL: YOLO model not found at {YOLO_MODEL_PATH}")
    sys.exit(2)
if not LVFACE_MODEL_PATH.exists():
    log(f"FATAL: LVFace model not found at {LVFACE_MODEL_PATH}")
    sys.exit(2)

log(f"Hashing {YOLO_MODEL_PATH} ...")
yolo_sha = sha256_file(YOLO_MODEL_PATH)
log(f"  -> {yolo_sha}")
log(f"Hashing {LVFACE_MODEL_PATH} ...")
lvface_sha = sha256_file(LVFACE_MODEL_PATH)
log(f"  -> {lvface_sha}")

log("Loading YOLO session ...")
yolo_session = ort.InferenceSession(
    str(YOLO_MODEL_PATH), sess_options=session_options, providers=providers
)
yolo_input_name = yolo_session.get_inputs()[0].name
yolo_output_names = [o.name for o in yolo_session.get_outputs()]
log(f"  inputs={[i.name for i in yolo_session.get_inputs()]} outputs={yolo_output_names}")

log("Loading LVFace session ...")
lvface_session = ort.InferenceSession(
    str(LVFACE_MODEL_PATH), sess_options=session_options, providers=providers
)
lvface_input_name = lvface_session.get_inputs()[0].name
lvface_output_names = [o.name for o in lvface_session.get_outputs()]
log(f"  inputs={[i.name for i in lvface_session.get_inputs()]} outputs={lvface_output_names}")


# ---------------------------------------------------------------------------
# Run inference on deterministic inputs
# ---------------------------------------------------------------------------

rng = np.random.default_rng(SEED)
tests = {}

log(f"Generating {NUM_YOLO_FRAMES} YOLO test inputs (seed={SEED}) ...")
for i in range(NUM_YOLO_FRAMES):
    inp = rng.uniform(0.0, 1.0, size=YOLO_INPUT_SHAPE).astype(np.float32)
    test_name = f"yolo_frame_{i}"

    # Warm-up pass for the first iteration to stabilize timing.
    if i == 0:
        yolo_session.run(yolo_output_names, {yolo_input_name: inp})

    t0 = time.perf_counter()
    outputs = yolo_session.run(yolo_output_names, {yolo_input_name: inp})
    latency_ms = (time.perf_counter() - t0) * 1000.0

    tests[test_name] = {
        "model": "yolo12n",
        "input_index": i,
        "input_summary": summarize_array(inp),
        "latency_ms": latency_ms,
        "outputs": {
            name: summarize_array(out) for name, out in zip(yolo_output_names, outputs)
        },
    }
    log(f"  {test_name}: {latency_ms:.2f} ms; outputs={[o.shape for o in outputs]}")

log(f"Generating {NUM_LVFACE_FRAMES} LVFace test inputs (seed={SEED}) ...")
for i in range(NUM_LVFACE_FRAMES):
    inp = rng.uniform(0.0, 1.0, size=LVFACE_INPUT_SHAPE).astype(np.float32)
    test_name = f"lvface_frame_{i}"

    if i == 0:
        lvface_session.run(lvface_output_names, {lvface_input_name: inp})

    t0 = time.perf_counter()
    outputs = lvface_session.run(lvface_output_names, {lvface_input_name: inp})
    latency_ms = (time.perf_counter() - t0) * 1000.0

    tests[test_name] = {
        "model": "lvface",
        "input_index": i,
        "input_summary": summarize_array(inp),
        "latency_ms": latency_ms,
        "outputs": {
            name: summarize_array(out) for name, out in zip(lvface_output_names, outputs)
        },
    }
    log(f"  {test_name}: {latency_ms:.2f} ms; outputs={[o.shape for o in outputs]}")


# ---------------------------------------------------------------------------
# Assemble current report
# ---------------------------------------------------------------------------

report = {
    "metadata": {
        "timestamp_utc": now_iso(),
        "onnxruntime_version": ort.__version__,
        "providers_used": providers,
        "providers_available": providers_available,
        "models": {
            "yolo12n": {"path": str(YOLO_MODEL_PATH), "sha256": yolo_sha},
            "lvface": {"path": str(LVFACE_MODEL_PATH), "sha256": lvface_sha},
        },
        "config": {
            "seed": SEED,
            "yolo_input_shape": list(YOLO_INPUT_SHAPE),
            "lvface_input_shape": list(LVFACE_INPUT_SHAPE),
            "num_yolo_frames": NUM_YOLO_FRAMES,
            "num_lvface_frames": NUM_LVFACE_FRAMES,
            "max_abs_diff_threshold": MAX_ABS_DIFF_THRESHOLD,
        },
    },
    "tests": tests,
    "comparison": None,
    "status": "BASELINE_GENERATED",
}


# ---------------------------------------------------------------------------
# Compare to baseline if present
# ---------------------------------------------------------------------------

def compare_to_baseline(current: dict, baseline: dict) -> dict:
    checks = []
    overall_status = "PASS"

    # Model file hashes (most important: any model bytes change is a real event)
    for model_name in ("yolo12n", "lvface"):
        b_sha = baseline["metadata"]["models"][model_name]["sha256"]
        c_sha = current["metadata"]["models"][model_name]["sha256"]
        if b_sha != c_sha:
            checks.append({
                "test": f"model_file_{model_name}",
                "status": "FAIL",
                "reason": f"Model file sha256 changed: {b_sha[:12]} -> {c_sha[:12]}",
            })
            overall_status = "FAIL"
        else:
            checks.append({
                "test": f"model_file_{model_name}",
                "status": "PASS",
                "reason": "Model file unchanged.",
            })

    # Per-test output comparison
    for test_name, test in current["tests"].items():
        if test_name not in baseline["tests"]:
            checks.append({
                "test": test_name,
                "status": "MISSING_BASELINE",
                "reason": "No baseline entry for this test.",
            })
            continue

        b_test = baseline["tests"][test_name]
        for out_name, out_summary in test["outputs"].items():
            if out_name not in b_test["outputs"]:
                checks.append({
                    "test": f"{test_name}.{out_name}",
                    "status": "MISSING_BASELINE_OUTPUT",
                    "reason": f"Baseline has no output {out_name}",
                })
                continue

            b_summary = b_test["outputs"][out_name]

            # Shape check (any mismatch is a real fail).
            if list(out_summary["shape"]) != list(b_summary["shape"]):
                checks.append({
                    "test": f"{test_name}.{out_name}",
                    "status": "FAIL",
                    "reason": f"Shape mismatch: baseline {b_summary['shape']} vs current {out_summary['shape']}",
                })
                overall_status = "FAIL"
                continue

            # Tight check: bit-identical via sha256.
            if out_summary["sha256"] == b_summary["sha256"]:
                checks.append({
                    "test": f"{test_name}.{out_name}",
                    "status": "PASS",
                    "reason": "Bit-identical to baseline.",
                })
                continue

            # Loose check: stats within tolerance.
            drift_report = {}
            failed = False
            for stat in ("min", "max", "mean", "std"):
                drift = abs(out_summary[stat] - b_summary[stat])
                drift_report[stat] = drift
                if drift > MAX_ABS_DIFF_THRESHOLD:
                    failed = True
            if failed:
                checks.append({
                    "test": f"{test_name}.{out_name}",
                    "status": "FAIL",
                    "reason": f"Stats drifted beyond {MAX_ABS_DIFF_THRESHOLD}",
                    "drift": drift_report,
                })
                overall_status = "FAIL"
            else:
                checks.append({
                    "test": f"{test_name}.{out_name}",
                    "status": "DRIFT_WITHIN_TOLERANCE",
                    "reason": "Stats match within tolerance though sha256 differs.",
                    "drift": drift_report,
                })

    return {
        "baseline_timestamp": baseline["metadata"]["timestamp_utc"],
        "checks": checks,
        "overall_status": overall_status,
    }


if BASELINE_PATH.exists():
    log(f"Loading baseline from {BASELINE_PATH}")
    with open(BASELINE_PATH) as fp:
        baseline = json.load(fp)
    cmp_result = compare_to_baseline(report, baseline)
    report["comparison"] = cmp_result
    report["status"] = cmp_result["overall_status"]
    log(f"== Comparison status: {cmp_result['overall_status']} ==")
    for check in cmp_result["checks"]:
        log(f"  [{check['status']}] {check['test']}: {check.get('reason', '')}")
else:
    log(f"No baseline at {BASELINE_PATH}; this run will become the new baseline.")


# ---------------------------------------------------------------------------
# Write report
# ---------------------------------------------------------------------------

WORKING_DIR.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w") as fp:
    json.dump(report, fp, indent=2)
log(f"Wrote {OUTPUT_PATH}")
log(f"FINAL_STATUS: {report['status']}")
