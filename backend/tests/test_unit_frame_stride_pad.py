"""ATT: detector input must be stride-32 aligned regardless of camera frame size.

Webcam canvases routinely produce dims like 1280x720; YOLOv12's neck Concat
fails on non-multiple-of-32 inputs. The preprocessing pads bottom/right so
existing coordinates stay valid.
"""

from __future__ import annotations

import numpy as np

from app.services.pipeline.frame import _frame_to_model_input


def test_output_dims_are_multiples_of_32_for_common_camera_sizes() -> None:
    for h, w in ((720, 1280), (1080, 1920), (480, 640), (45, 46), (1, 1)):
        out = _frame_to_model_input(np.zeros((h, w, 3), dtype=np.uint8))
        assert out.shape[2] % 32 == 0 and out.shape[3] % 32 == 0, (h, w, out.shape)


def test_padding_preserves_original_pixels_and_channels() -> None:
    frame = np.arange(45 * 46 * 3, dtype=np.uint8).reshape(45, 46, 3)
    out = _frame_to_model_input(frame)
    assert out.dtype == np.float32 and out.shape == (1, 3, 64, 64)
    np.testing.assert_array_equal(out[0, :, :45, :46], frame.transpose(2, 0, 1))
    assert out[0, :, :45, 46:].sum() == 0
    assert out[0, :, 45:, :].sum() == 0


def test_already_aligned_input_is_untouched() -> None:
    frame = np.ones((480, 640, 3), dtype=np.uint8)
    out = _frame_to_model_input(frame)
    assert out.shape == (1, 3, 480, 640)
