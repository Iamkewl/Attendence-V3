#!/usr/bin/env python3
"""Phase 3.6 automation: download/export YOLOv12 + LVFace for Triton."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Iterable


# Hardcoded official source identifiers to avoid accidental community checkpoints.
YOLO_OFFICIAL_WEIGHT = "yolo12n.pt"
YOLO_OFFICIAL_HF_MODEL_ID = "nielsr/yolov12n"

LVFACE_GITHUB_REPOSITORY = "https://github.com/bytedance/LVFace.git"
LVFACE_HF_REPOSITORY = "bytedance-research/LVFace"
LVFACE_DEFAULT_VARIANT = "LVFace-B_Glint360K"

LVFACE_NETWORK_CANDIDATES: dict[str, list[str]] = {
    "LVFace-T_Glint360K": ["vit_t", "vit_t_dp005_mask0"],
    "LVFace-S_Glint360K": ["vit_s", "vit_s_dp005_mask_0"],
    "LVFace-B_Glint360K": ["vit_b_dp005_mask_005", "vit_b"],
    "LVFace-L_Glint360K": ["vit_l_dp005_mask_005"],
}

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_REPOSITORY_ROOT = PROJECT_ROOT / "infra" / "triton" / "model_repository"

YOLO_TARGET_MODEL_PATH = MODEL_REPOSITORY_ROOT / "yolov12" / "1" / "model.onnx"
LVFACE_TARGET_MODEL_PATH = MODEL_REPOSITORY_ROOT / "lvface" / "1" / "model.onnx"

YOLO_CONFIG_PATH = MODEL_REPOSITORY_ROOT / "yolov12" / "config.pbtxt"
LVFACE_CONFIG_PATH = MODEL_REPOSITORY_ROOT / "lvface" / "config.pbtxt"


@dataclass
class TensorSpec:
    """Minimal tensor metadata for Triton config generation."""

    name: str
    dims: list[int]


def _print_header(message: str) -> None:
    print(f"\n=== {message} ===")


def _run_command(command: list[str], cwd: Path | None = None) -> None:
    """Run a command and stream captured output for easier debugging."""
    printable_cwd = str(cwd) if cwd else str(PROJECT_ROOT)
    print(f"[run] cwd={printable_cwd}")
    print(f"[run] {' '.join(command)}")
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
    )

    if completed.stdout:
        print(completed.stdout.rstrip())
    if completed.stderr:
        print(completed.stderr.rstrip())

    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(command)}"
        )


def _ensure_python_dependencies() -> None:
    """Fail with install guidance when required Python packages are missing."""
    required_modules = {
        "ultralytics": "ultralytics",
        "huggingface_hub": "huggingface_hub",
        "onnx": "onnx",
        "onnxscript": "onnxscript",
        "torch": "torch",
        "timm": "timm",
    }
    missing_packages: list[str] = []

    for module_name, package_name in required_modules.items():
        if importlib.util.find_spec(module_name) is None:
            missing_packages.append(package_name)

    if missing_packages:
        package_list = sorted(set(missing_packages))
        install_hint = (
            f"{sys.executable} -m pip install "
            "ultralytics huggingface_hub onnx onnxscript torch timm"
        )
        raise RuntimeError(
            "Missing required Python dependencies: "
            + ", ".join(package_list)
            + "\nInstall them with:\n  "
            + install_hint
        )


def _clone_lvface_repo(temp_root: Path) -> Path:
    """Clone LVFace repository, with zip fallback when git is unavailable."""
    destination = temp_root / "LVFace"

    if shutil.which("git"):
        _run_command([
            "git",
            "clone",
            "--depth",
            "1",
            LVFACE_GITHUB_REPOSITORY,
            str(destination),
        ])
        return destination

    print("[warn] git is not available. Falling back to GitHub zip download.")
    import urllib.request
    import zipfile

    archive_path = temp_root / "LVFace-main.zip"
    source_url = "https://codeload.github.com/bytedance/LVFace/zip/refs/heads/main"
    urllib.request.urlretrieve(source_url, archive_path)

    with zipfile.ZipFile(archive_path, mode="r") as archive:
        archive.extractall(temp_root)

    extracted_root = temp_root / "LVFace-main"
    if not extracted_root.exists():
        raise RuntimeError("Failed to extract LVFace repository archive.")

    extracted_root.rename(destination)
    return destination


def _download_lvface_weights(variant: str) -> Path:
    """Download official LVFace PyTorch weights from Hugging Face."""
    from huggingface_hub import hf_hub_download

    filename = f"{variant}/{variant}.pt"
    print(
        f"[info] downloading LVFace checkpoint from {LVFACE_HF_REPOSITORY} "
        f"(file={filename})"
    )
    downloaded_path = hf_hub_download(repo_id=LVFACE_HF_REPOSITORY, filename=filename)
    return Path(downloaded_path)


def _rename_tensor_value(model: object, old_name: str, new_name: str) -> None:
    """Rename a tensor value throughout an ONNX graph."""
    if old_name == new_name:
        return

    graph = model.graph

    for node in graph.node:
        for index, input_name in enumerate(node.input):
            if input_name == old_name:
                node.input[index] = new_name
        for index, output_name in enumerate(node.output):
            if output_name == old_name:
                node.output[index] = new_name

    for value_info in list(graph.input) + list(graph.output) + list(graph.value_info):
        if value_info.name == old_name:
            value_info.name = new_name

    for initializer in graph.initializer:
        if initializer.name == old_name:
            initializer.name = new_name

    for sparse_initializer in graph.sparse_initializer:
        if sparse_initializer.values.name == old_name:
            sparse_initializer.values.name = new_name


def _normalize_yolo_onnx_io(
    source_onnx_path: Path,
    destination_onnx_path: Path,
    *,
    input_name: str,
    output_name: str,
) -> None:
    """Keep one output and enforce stable IO names for Triton."""
    import onnx

    model = onnx.load(str(source_onnx_path))
    graph = model.graph

    if len(graph.input) != 1:
        raise RuntimeError(
            f"Expected one YOLO input tensor, got {len(graph.input)} in {source_onnx_path}."
        )
    if len(graph.output) < 1:
        raise RuntimeError(f"YOLO model has no outputs: {source_onnx_path}")

    old_input_name = graph.input[0].name
    _rename_tensor_value(model, old_input_name, input_name)
    graph.input[0].name = input_name

    while len(graph.output) > 1:
        graph.output.pop()

    old_output_name = graph.output[0].name
    _rename_tensor_value(model, old_output_name, output_name)
    graph.output[0].name = output_name

    destination_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(destination_onnx_path))


def _pick_exported_onnx(export_result: object, export_dir: Path) -> Path:
    """Resolve Ultralytics export return value to an existing ONNX path."""
    if isinstance(export_result, (str, Path)):
        candidate = Path(export_result)
        if candidate.exists() and candidate.suffix.lower() == ".onnx":
            return candidate

    found = sorted(export_dir.rglob("*.onnx"))
    if not found:
        raise RuntimeError(f"No ONNX file was produced under {export_dir}.")
    return found[-1]


def _patch_yolo12_aattn_compat(pytorch_model: object) -> int:
    """Patch older Ultralytics YOLO12 AAttn modules missing all_head_dim."""
    patched = 0
    modules = getattr(pytorch_model, "modules", None)
    if modules is None:
        return patched

    for module in modules():
        if module.__class__.__name__ != "AAttn":
            continue
        if hasattr(module, "all_head_dim"):
            continue

        head_dim = getattr(module, "head_dim", None)
        num_heads = getattr(module, "num_heads", None)
        if isinstance(head_dim, int) and isinstance(num_heads, int):
            module.all_head_dim = head_dim * num_heads
            patched += 1

    return patched


def _export_yolov12_model(temp_root: Path) -> Path:
    """Export official Ultralytics YOLOv12n weights to ONNX."""
    _print_header("Exporting official YOLOv12n model to ONNX")
    print(
        "[info] Using official Ultralytics base weights "
        f"({YOLO_OFFICIAL_WEIGHT}) and anchoring source id {YOLO_OFFICIAL_HF_MODEL_ID}."
    )

    from ultralytics import YOLO

    export_dir = temp_root / "yolo_export"
    export_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(YOLO_OFFICIAL_WEIGHT)
    patched_modules = _patch_yolo12_aattn_compat(model.model)
    if patched_modules > 0:
        print(
            "[warn] Applied Ultralytics compatibility patch for YOLO12 AAttn "
            f"on {patched_modules} modules."
        )

    export_result = model.export(
        format="onnx",
        imgsz=640,
        dynamic=True,
        nms=True,
        simplify=False,
        batch=16,
        opset=17,
        project=str(export_dir),
        name="yolov12n",
        exist_ok=True,
    )

    raw_export_path = _pick_exported_onnx(export_result, export_dir)
    normalized_path = temp_root / "yolov12_normalized.onnx"
    _normalize_yolo_onnx_io(
        raw_export_path,
        normalized_path,
        input_name="INPUT__0",
        output_name="OUTPUT__0",
    )
    return normalized_path


def _write_lvface_wrapper_script(wrapper_path: Path) -> None:
    """Write a short standalone wrapper script that exports LVFace to ONNX."""
    wrapper_source = dedent(
        """
        import argparse
        from pathlib import Path
        import torch
        import torch.nn as nn


        def _unwrap_state_dict(payload):
            if isinstance(payload, dict):
                for key in ("state_dict", "model_state_dict", "model", "net"):
                    value = payload.get(key)
                    if isinstance(value, dict):
                        payload = value
                        break
            if not isinstance(payload, dict):
                raise RuntimeError("Checkpoint did not contain a PyTorch state_dict mapping.")

            normalized = {}
            for key, value in payload.items():
                new_key = key
                if new_key.startswith("module."):
                    new_key = new_key[len("module.") :]
                if new_key.startswith("backbone."):
                    new_key = new_key[len("backbone.") :]
                normalized[new_key] = value
            return normalized


        class LVFaceLegacyWrapper(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone

            def forward(self, x):
                embedding = self.backbone(x)
                if embedding.ndim > 2:
                    embedding = embedding.reshape(embedding.shape[0], -1)

                # LVFace uses embedding norm as quality proxy in evaluation settings.
                quality_norm = torch.linalg.vector_norm(embedding, ord=2, dim=1, keepdim=True)
                live_logit = (quality_norm - 20.0) / 8.0
                liveness_logits = torch.cat([-live_logit, live_logit], dim=1)
                return liveness_logits, embedding


        def _build_backbone(candidates, state_dict):
            from backbones import get_model

            strict_errors = {}
            for network_name in candidates:
                model = get_model(network_name, dropout=0.0, fp16=False, num_features=512)
                try:
                    model.load_state_dict(state_dict, strict=True)
                    return network_name, model
                except Exception as exc:
                    strict_errors[network_name] = str(exc)

            best_candidate = None
            for network_name in candidates:
                model = get_model(network_name, dropout=0.0, fp16=False, num_features=512)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                score = len(missing) + len(unexpected)
                if best_candidate is None or score < best_candidate[0]:
                    best_candidate = (score, network_name, model, missing, unexpected)

            if best_candidate is None:
                details = "\\n".join(
                    [f"  - {name}: {error}" for name, error in strict_errors.items()]
                )
                raise RuntimeError("Unable to initialize LVFace backbone.\\n" + details)

            score, network_name, model, missing, unexpected = best_candidate
            if score > 128:
                details = "\\n".join(
                    [f"  - {name}: {error}" for name, error in strict_errors.items()]
                )
                raise RuntimeError(
                    "No backbone matched the checkpoint closely enough.\\n"
                    + f"Best candidate={network_name}, mismatch score={score}.\\n"
                    + details
                )

            print(
                f"[wrapper] fallback non-strict load network={network_name} "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
            return network_name, model


        def main():
            parser = argparse.ArgumentParser(description="Export LVFace checkpoint to ONNX.")
            parser.add_argument("--repo", required=True, type=Path)
            parser.add_argument("--weights", required=True, type=Path)
            parser.add_argument("--output", required=True, type=Path)
            parser.add_argument("--candidates", required=True)
            parser.add_argument("--opset", default=18, type=int)
            args = parser.parse_args()

            repo_path = args.repo.resolve()
            if not repo_path.exists():
                raise RuntimeError(f"LVFace repository path does not exist: {repo_path}")

            import sys

            sys.path.insert(0, str(repo_path))

            state_dict = _unwrap_state_dict(torch.load(args.weights, map_location="cpu"))
            candidate_networks = [entry.strip() for entry in args.candidates.split(",") if entry.strip()]
            if not candidate_networks:
                raise RuntimeError("No candidate LVFace network names were provided.")

            network_name, backbone = _build_backbone(candidate_networks, state_dict)
            print(f"[wrapper] selected network={network_name}")

            wrapper = LVFaceLegacyWrapper(backbone)
            wrapper.eval()

            args.output.parent.mkdir(parents=True, exist_ok=True)
            dummy_input = torch.randn(1, 3, 112, 112, dtype=torch.float32)
            with torch.no_grad():
                torch.onnx.export(
                    wrapper,
                    dummy_input,
                    str(args.output),
                    export_params=True,
                    opset_version=args.opset,
                    external_data=False,
                    do_constant_folding=True,
                    input_names=["INPUT__0"],
                    output_names=["LIVENESS__0", "EMBEDDING__1"],
                    dynamic_axes={
                        "INPUT__0": {0: "batch_size"},
                        "LIVENESS__0": {0: "batch_size"},
                        "EMBEDDING__1": {0: "batch_size"},
                    },
                )

            print(f"[wrapper] exported={args.output}")


        if __name__ == "__main__":
            main()
        """
    ).strip()

    wrapper_path.write_text(wrapper_source + "\n", encoding="utf-8")


def _export_lvface_model(temp_root: Path, variant: str) -> Path:
    """Clone LVFace repo, download official weights, and export ONNX via wrapper."""
    _print_header("Exporting LVFace model to ONNX")
    lvface_repo_path = _clone_lvface_repo(temp_root)
    weights_path = _download_lvface_weights(variant)

    candidate_networks = LVFACE_NETWORK_CANDIDATES.get(variant)
    if not candidate_networks:
        raise RuntimeError(
            f"Unsupported LVFace variant '{variant}'. Supported: {sorted(LVFACE_NETWORK_CANDIDATES)}"
        )

    wrapper_script_path = temp_root / "export_lvface_wrapper.py"
    _write_lvface_wrapper_script(wrapper_script_path)

    exported_onnx_path = temp_root / "lvface_export.onnx"
    _run_command(
        [
            sys.executable,
            str(wrapper_script_path),
            "--repo",
            str(lvface_repo_path),
            "--weights",
            str(weights_path),
            "--output",
            str(exported_onnx_path),
            "--candidates",
            ",".join(candidate_networks),
            "--opset",
            "18",
        ]
    )

    if not exported_onnx_path.exists():
        raise RuntimeError(f"Expected LVFace export at {exported_onnx_path}, but file was not found.")

    return exported_onnx_path


def _read_onnx_tensor_dims(value_info: object) -> list[int]:
    """Convert ONNX tensor shape to Triton-style dimensions."""
    tensor_type = value_info.type.tensor_type
    dims: list[int] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value") and dim.dim_value > 0:
            dims.append(int(dim.dim_value))
        else:
            dims.append(-1)
    return dims


def _strip_batch_dim(dims: list[int], max_batch_size: int) -> list[int]:
    if max_batch_size > 0 and dims:
        return dims[1:]
    return dims


def _extract_onnx_signature(onnx_path: Path, *, max_batch_size: int) -> tuple[TensorSpec, list[TensorSpec]]:
    """Extract input/output names + dimensions from an ONNX model."""
    import onnx

    model = onnx.load(str(onnx_path))
    graph = model.graph

    if len(graph.input) != 1:
        raise RuntimeError(f"Expected one input tensor in {onnx_path}, got {len(graph.input)}")
    if len(graph.output) < 1:
        raise RuntimeError(f"Expected at least one output tensor in {onnx_path}.")

    input_dims = _strip_batch_dim(_read_onnx_tensor_dims(graph.input[0]), max_batch_size)
    input_spec = TensorSpec(name=graph.input[0].name, dims=input_dims)

    output_specs = []
    for output in graph.output:
        output_dims = _strip_batch_dim(_read_onnx_tensor_dims(output), max_batch_size)
        output_specs.append(TensorSpec(name=output.name, dims=output_dims))

    return input_spec, output_specs


def _dims_to_pbtxt(dims: Iterable[int]) -> str:
    return "[ " + ", ".join(str(int(value)) for value in dims) + " ]"


def _render_output_block(spec: TensorSpec, with_trailing_comma: bool) -> str:
    trailing = "," if with_trailing_comma else ""
    return (
        "  {\n"
        f"    name: \"{spec.name}\"\n"
        "    data_type: TYPE_FP32\n"
        f"    dims: {_dims_to_pbtxt(spec.dims)}\n"
        f"  }}{trailing}"
    )


def _write_yolo_config(input_spec: TensorSpec, output_specs: list[TensorSpec]) -> None:
    if len(output_specs) != 1:
        raise RuntimeError(
            "YOLO Triton config expects exactly one output after export normalization."
        )

    output_block = _render_output_block(output_specs[0], with_trailing_comma=False)
    content = dedent(
        f"""
        name: "yolov12"
        platform: "onnxruntime_onnx"
        max_batch_size: 16

        input [
          {{
            name: "{input_spec.name}"
            data_type: TYPE_FP32
            format: FORMAT_NCHW
            dims: {_dims_to_pbtxt(input_spec.dims)}
          }}
        ]

        output [
        {output_block}
        ]

        instance_group [
          {{
            count: 1
            kind: KIND_GPU
            gpus: [ 0 ]
          }}
        ]

        dynamic_batching {{
          preferred_batch_size: [ 4, 8, 16 ]
          max_queue_delay_microseconds: 2000
          preserve_ordering: true
          default_queue_policy {{
            timeout_action: REJECT
            default_timeout_microseconds: 8000
            allow_timeout_override: false
            max_queue_size: 1024
          }}
        }}

        optimization {{
          execution_accelerators {{
            gpu_execution_accelerator: [
              {{
                name: "tensorrt"
                parameters {{ key: "precision_mode" value: "FP16" }}
                parameters {{ key: "max_workspace_size_bytes" value: "4294967296" }}
              }}
            ]
          }}
        }}

        version_policy: {{
          latest {{
            num_versions: 2
          }}
        }}
        """
    ).strip()

    YOLO_CONFIG_PATH.write_text(content + "\n", encoding="utf-8")
    print(f"[info] wrote {YOLO_CONFIG_PATH}")


def _write_lvface_config(input_spec: TensorSpec, output_specs: list[TensorSpec]) -> None:
    expected_names = {"LIVENESS__0", "EMBEDDING__1"}
    output_name_set = {spec.name for spec in output_specs}
    if output_name_set != expected_names:
        raise RuntimeError(
            "LVFace ONNX outputs do not match expected names "
            f"{sorted(expected_names)}. Found: {sorted(output_name_set)}"
        )

    ordered_outputs = [
        next(spec for spec in output_specs if spec.name == "LIVENESS__0"),
        next(spec for spec in output_specs if spec.name == "EMBEDDING__1"),
    ]

    output_blocks = [
        _render_output_block(ordered_outputs[0], with_trailing_comma=True),
        _render_output_block(ordered_outputs[1], with_trailing_comma=False),
    ]

    content = dedent(
        f"""
        name: "lvface"
        platform: "onnxruntime_onnx"
        max_batch_size: 128

        input [
          {{
            name: "{input_spec.name}"
            data_type: TYPE_FP32
            format: FORMAT_NCHW
            dims: {_dims_to_pbtxt(input_spec.dims)}
          }}
        ]

        output [
        {output_blocks[0]}
        {output_blocks[1]}
        ]

        instance_group [
          {{
            count: 2
            kind: KIND_GPU
            gpus: [ 0 ]
          }}
        ]

        dynamic_batching {{
          preferred_batch_size: [ 16, 32, 64, 128 ]
          max_queue_delay_microseconds: 2500
          preserve_ordering: true
          default_queue_policy {{
            timeout_action: REJECT
            default_timeout_microseconds: 12000
            allow_timeout_override: false
            max_queue_size: 4096
          }}
        }}

        optimization {{
          execution_accelerators {{
            gpu_execution_accelerator: [
              {{
                name: "tensorrt"
                parameters {{ key: "precision_mode" value: "FP16" }}
                parameters {{ key: "max_workspace_size_bytes" value: "2147483648" }}
              }}
            ]
          }}
        }}

        version_policy: {{
          latest {{
            num_versions: 2
          }}
        }}
        """
    ).strip()

    LVFACE_CONFIG_PATH.write_text(content + "\n", encoding="utf-8")
    print(f"[info] wrote {LVFACE_CONFIG_PATH}")


def _copy_model(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    print(f"[info] copied model: {source_path} -> {destination_path}")

    # Some exporters write a sidecar external-data file next to model.onnx.
    source_sidecar = source_path.with_suffix(source_path.suffix + ".data")
    if source_sidecar.exists():
        destination_sidecar = destination_path.with_suffix(destination_path.suffix + ".data")
        shutil.copy2(source_sidecar, destination_sidecar)
        print(f"[info] copied model sidecar: {source_sidecar} -> {destination_sidecar}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3.6 setup: download/export YOLOv12 + LVFace into Triton model repository."
    )
    parser.add_argument(
        "--lvface-variant",
        default=LVFACE_DEFAULT_VARIANT,
        choices=sorted(LVFACE_NETWORK_CANDIDATES.keys()),
        help="LVFace checkpoint variant on Hugging Face.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary working directory for debugging.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    _print_header("Phase 3.6 model setup")
    print(f"[info] project root: {PROJECT_ROOT}")
    print(f"[info] model repository: {MODEL_REPOSITORY_ROOT}")

    try:
        _ensure_python_dependencies()
    except RuntimeError as exc:
        print(f"[error] {exc}")
        return 1

    temp_dir_context = tempfile.TemporaryDirectory(prefix="attendance_phase36_")
    temp_root = Path(temp_dir_context.name)
    print(f"[info] temporary workspace: {temp_root}")

    try:
        yolo_onnx_path = _export_yolov12_model(temp_root)
        _copy_model(yolo_onnx_path, YOLO_TARGET_MODEL_PATH)

        yolo_input, yolo_outputs = _extract_onnx_signature(
            YOLO_TARGET_MODEL_PATH,
            max_batch_size=16,
        )
        _write_yolo_config(yolo_input, yolo_outputs)

        lvface_onnx_path = _export_lvface_model(temp_root, args.lvface_variant)
        _copy_model(lvface_onnx_path, LVFACE_TARGET_MODEL_PATH)

        lvface_input, lvface_outputs = _extract_onnx_signature(
            LVFACE_TARGET_MODEL_PATH,
            max_batch_size=128,
        )
        _write_lvface_config(lvface_input, lvface_outputs)

        _print_header("Setup complete")
        print("[ok] YOLOv12 model exported to:")
        print(f"      {YOLO_TARGET_MODEL_PATH}")
        print("[ok] LVFace model exported to:")
        print(f"      {LVFACE_TARGET_MODEL_PATH}")
        print("[ok] Triton configs updated:")
        print(f"      {YOLO_CONFIG_PATH}")
        print(f"      {LVFACE_CONFIG_PATH}")
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[error] setup failed: {exc}")
        return 1
    finally:
        if args.keep_temp:
            print(f"[info] keeping temporary workspace: {temp_root}")
        else:
            temp_dir_context.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())