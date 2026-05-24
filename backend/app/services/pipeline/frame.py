"""Frame decoding, tensor preparation, face cropping, and detection row normalization."""

from __future__ import annotations

import base64
import binascii
import math

import numpy as np

from app.domain.schemas import ImageTensorPayload


def _decode_frame_tensor(frame: ImageTensorPayload) -> np.ndarray:
    """Decode a base64-encoded frame payload into an HWC float32 tensor."""
    try:
        frame_bytes = base64.b64decode(frame.data_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Frame {frame.frame_id} contains invalid base64 data.") from exc

    if frame.dtype == "uint8":
        decoded = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
            frame.height,
            frame.width,
            frame.channels,
        )
        tensor = decoded.astype(np.float32)
        if frame.normalize:
            tensor /= 255.0
    else:
        tensor = np.frombuffer(frame_bytes, dtype=np.float32).reshape(
            frame.height,
            frame.width,
            frame.channels,
        )
        tensor = tensor.astype(np.float32, copy=False)

    return np.ascontiguousarray(tensor)


def _frame_to_model_input(frame_tensor: np.ndarray) -> np.ndarray:
    """Convert HWC frame tensor to NCHW batch tensor for Triton model input."""
    if frame_tensor.ndim != 3:
        raise ValueError("Input frame tensor must be a rank-3 HWC tensor.")

    channels = frame_tensor.shape[2]
    if channels not in {1, 3, 4}:
        raise ValueError(f"Unsupported channel count {channels}; expected 1, 3, or 4.")

    if channels == 4:
        frame_tensor = frame_tensor[:, :, :3]

    if channels == 1:
        frame_tensor = np.repeat(frame_tensor, 3, axis=2)

    nchw = np.transpose(frame_tensor, (2, 0, 1))[np.newaxis, :, :, :]
    return np.ascontiguousarray(nchw, dtype=np.float32)


def _extract_detection_rows(raw_tensor: np.ndarray) -> np.ndarray:
    """Normalize detector output tensors into [num_detections, features] shape."""
    tensor = np.asarray(raw_tensor, dtype=np.float32)

    if tensor.size == 0:
        return np.empty((0, 6), dtype=np.float32)

    if tensor.ndim == 4:
        tensor = tensor.reshape(tensor.shape[0], -1, tensor.shape[-1])

    if tensor.ndim == 3:
        if tensor.shape[0] == 1:
            tensor = tensor[0]
        else:
            tensor = tensor.reshape(-1, tensor.shape[-1])

    if tensor.ndim == 1:
        if tensor.shape[0] < 6:
            return np.empty((0, 6), dtype=np.float32)
        feature_count = 6
        trimmed_size = tensor.shape[0] - (tensor.shape[0] % feature_count)
        tensor = tensor[:trimmed_size].reshape(-1, feature_count)

    if tensor.ndim != 2:
        return np.empty((0, 6), dtype=np.float32)

    if tensor.shape[1] < 6 and tensor.shape[0] >= 6:
        tensor = tensor.T

    if tensor.shape[1] > 256 and tensor.shape[0] <= 256:
        tensor = tensor.T

    if tensor.shape[1] < 5:
        return np.empty((0, 6), dtype=np.float32)

    return tensor


def _resize_nearest(image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """Resize an HWC image tensor with nearest-neighbor sampling using NumPy indexing."""
    source_height, source_width = image.shape[:2]
    y_indices = np.linspace(0, source_height - 1, target_height, dtype=np.float32).round().astype(int)
    x_indices = np.linspace(0, source_width - 1, target_width, dtype=np.float32).round().astype(int)

    y_indices = np.clip(y_indices, 0, source_height - 1)
    x_indices = np.clip(x_indices, 0, source_width - 1)

    return image[y_indices][:, x_indices, :]


def _crop_face(
    frame_tensor: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    crop_size: int,
) -> np.ndarray:
    """Crop and normalize a face ROI to a fixed-size tensor for LVFace inference."""
    x, y, w, h = bbox
    x0_i = int(max(math.floor(x), 0))
    y0_i = int(max(math.floor(y), 0))
    x1_i = int(min(math.ceil(x + w), frame_tensor.shape[1]))
    y1_i = int(min(math.ceil(y + h), frame_tensor.shape[0]))

    if x1_i <= x0_i or y1_i <= y0_i:
        raise ValueError("Face crop is empty after clipping.")

    face_crop = frame_tensor[y0_i:y1_i, x0_i:x1_i, :]
    resized = _resize_nearest(face_crop, crop_size, crop_size)
    return np.ascontiguousarray(resized.astype(np.float32, copy=False))


def _prepare_face_batch(face_crops: list[np.ndarray]) -> np.ndarray:
    """Convert face crops into an NCHW FP32 batch tensor for Triton inference."""
    if not face_crops:
        raise ValueError("Cannot build face batch from an empty list.")

    batch_hwc = np.stack(face_crops, axis=0).astype(np.float32, copy=False)
    batch_nchw = np.transpose(batch_hwc, (0, 3, 1, 2))
    return np.ascontiguousarray(batch_nchw, dtype=np.float32)
